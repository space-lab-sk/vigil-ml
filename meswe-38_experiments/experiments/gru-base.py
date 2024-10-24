import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import copy
import logging
import wandb
import argparse


from utils import set_seed, get_torch_device, load_config, \
    count_parameters, apply_glorot_xavier, inspect_gradient_norms
from utils import Logger

from preprocessing import get_k_fold, load_data, get_torch_data
from preprocessing import StandardScaler

from postprocessing import save_gradient_norms_plot, save_predictions_and_true_values_plot, \
    save_predictions_detail_plot, save_scatter_predictions_and_true_values, \
    get_dst_rmse, get_detail_properties, get_r_squared



def apply_smoothing(batch_x, augumentation_rate, smoothing_window=5):

    batch_size, sequence, features = batch_x.shape
    num_samples_to_smooth = int(batch_size * augumentation_rate)

    indices_to_smooth = np.random.choice(batch_size, num_samples_to_smooth, replace=False)
    
    smoothed_batch_x = batch_x.clone()
    for idx in indices_to_smooth:
        df = pd.DataFrame(smoothed_batch_x[idx].cpu().numpy())
        smoothed_data = df.rolling(window=smoothing_window, min_periods=1).mean()
        smoothed_batch_x[idx] = torch.tensor(smoothed_data.values, device=batch_x.device, dtype=torch.float32)

    return smoothed_batch_x


def train_model(model, train_loader, optimizer, criterion, device, AUGUMENTATION_RATE):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device) , targets.to(device)
        optimizer.zero_grad()
        inputs = apply_smoothing(inputs, AUGUMENTATION_RATE)
        outputs = model(inputs)
        targets = targets.squeeze(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    mean_loss = total_loss / len(train_loader)
    return mean_loss

def validate_model(model, val_test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_test_loader:
            inputs, targets = inputs.to(device) , targets.to(device)
            outputs = model(inputs)
            targets = targets.squeeze(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    mean_loss = total_loss / len(val_test_loader)
    return mean_loss, outputs.squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_gru_layers, dropout):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers

        self.gru = nn.GRU(input_size, hidden_size, num_gru_layers, batch_first=True, bidirectional=False, dropout=dropout)

        self.fc1 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h0 = torch.zeros(self.num_gru_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)

        out = self.fc1(out[:, -1, :])

        return out
    

if __name__=="__main__":

    ########################################
    #PART 1: EXPERIMENT CONFIGURATION SETUP
    ########################################

    parser = argparse.ArgumentParser(description='Process a config file location.')
    parser.add_argument("-cfn", '--config_file_name', type=str, help='Path to the input config file')
    parser.add_argument("-dev", "--device", type=str, help="Select device: cuda:0 | cuda:1 | cpu |")
    parser.add_argument("-dt", "--disable_tracking", action="store_false", help="Disable Weights and Biases tracking with its config file")
    args = parser.parse_args()

    config_file_name = args.config_file_name
    tracking_enabled = args.disable_tracking
    device_input = args.device
    #tracking_enabled = True if tracking_enabled is None else False

    config = load_config(f"configs/gru-configs/{config_file_name}")
    EXPERIMENT_NAME = config["logging"]["experiment_name"]
    EXPERIMENT_NOTES = config["logging"]["notes"]
    logger = Logger(EXPERIMENT_NAME)

    set_seed()
    device = get_torch_device(device_input)

    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["learning_rate"]
    NUM_EPOCHS = config["training"]["num_epochs"]
    WEIGHT_DECAY = config["training"]["weight_decay"]
    AUGUMENTATION_RATE = config["training"]["augumentation_rate"]

    INPUT_SIZE = config["model"]["input_size"]
    HIDDEN_CHANNELS = config["model"]["hidden_channels"]
    OUTPUT_SIZE = config["model"]["output_size"]
    NUM_GRU_LAYERS = config["model"]["num_gru_layers"]
    DROPOUT = config["model"]["dropout"]

    TIME_STEPS = config["data"]["time_steps"]
    PREDICTION_WINDOW = config["data"]["prediction_window"]
    K_FOLD = config["data"]["k_fold"]


    ###########################
    #PART 2: PREPARING DATA
    ###########################

    file_ids_train, file_ids_val, file_ids_test = get_k_fold(K_FOLD)

    train_X_unscaled, train_y_unscaled = load_data(file_ids_train, time_steps=TIME_STEPS, sliding_window=PREDICTION_WINDOW)
    val_X_unscaled, val_y_unscaled = load_data(file_ids_val, time_steps=TIME_STEPS, sliding_window=PREDICTION_WINDOW)
    test_X_unscaled, test_y_unscaled = load_data(file_ids_test, time_steps=TIME_STEPS, sliding_window=PREDICTION_WINDOW)

    standard_scaler = StandardScaler(train_X_unscaled, train_y_unscaled)

    train_X = standard_scaler.standardize_X(train_X_unscaled)
    val_X = standard_scaler.standardize_X(val_X_unscaled)
    test_X =standard_scaler.standardize_X(test_X_unscaled)

    train_y = standard_scaler.standardize_y(train_y_unscaled)
    val_y = standard_scaler.standardize_y(val_y_unscaled)
    test_y = standard_scaler.standardize_y(test_y_unscaled)

    train_X, train_y = get_torch_data(train_X, train_y)
    val_X, val_y = get_torch_data(val_X, val_y)
    test_X, test_y = get_torch_data(test_X, test_y)


    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    ###############################################
    #PART 3: DEEP LEARNING PART AND WANDB TRACKING
    ###############################################

    model = GRUModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_CHANNELS, output_size=OUTPUT_SIZE, num_gru_layers=NUM_GRU_LAYERS, dropout=DROPOUT)
    model.to(device)
    apply_glorot_xavier(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    if tracking_enabled:
        name=f"{EXPERIMENT_NAME} - {PREDICTION_WINDOW} Steps"
        wandb.init(
            project="MESWE-38-experiments",
            name=name,
            notes=EXPERIMENT_NOTES,
            entity="majirky-technical-university-of-ko-ice",
            anonymous="allow"
        )

        config = wandb.config
        config.inputs = INPUT_SIZE
        config.hidden_size = HIDDEN_CHANNELS
        config.num_gru_layers = NUM_GRU_LAYERS
        config.learning_rate = LEARNING_RATE
        config.batch_size = BATCH_SIZE
        config.criterion = str(criterion)
        config.optimizer = str(optimizer)
        config.time_steps = TIME_STEPS
        config.epochs = NUM_EPOCHS
        config.prediction_window = PREDICTION_WINDOW
        config.k_fold = K_FOLD
        config.weight_decay = WEIGHT_DECAY
        config.augumentation_rate = AUGUMENTATION_RATE

        wandb.watch(model, log="all", log_freq=1)


    print(model)
    logger.log_message(model)

    total_params = count_parameters(model)
    print(f"Total parameters of the model: {total_params}")
    logger.log_message(f"Total parameters of the model: {total_params}")


    ############################
    #PART 4: TRAINING LOOP
    ############################

    print(f"--------------TRAINING LOOP--------------")
    logger.log_message(f"--------------TRAINING LOOP--------------")
    
    losses = []
    val_losses = []
    gradient_norms = []

    best_val = 10000.0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device, AUGUMENTATION_RATE)
        val_loss, _ = validate_model(model, val_loader, criterion, device)

        losses.append(train_loss)
        val_losses.append(val_loss)

        #==============grad norms============
        total_norm = inspect_gradient_norms(model)
        gradient_norms.append(total_norm)
        #==========================

        print(f'{epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        logger.log_message(f'{epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        if tracking_enabled:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"model with val loss {val_loss} saved...")
            logger.log_message(f"model with val loss {val_loss} saved...")

    print('Training completed saving....')
    logger.log_message('Training completed saving....')
    torch.save(best_model_state, f'models/{EXPERIMENT_NAME}.pth')

    save_gradient_norms_plot(gradient_norms, 
                             tracking_enabled,
                             wandb=wandb,
                             save_path=f"logs/log_figures/grad_norms/{EXPERIMENT_NAME}_grad_norms.png")
    

    ############################
    #PART 4: MODEL EVALUATION
    ############################

    model.load_state_dict(best_model_state)
    

    test_loss, test_predictions_standardized = validate_model(model, test_loader, criterion, device)
    test_predictions_standardized = test_predictions_standardized.cpu()
    test_predictions = (test_predictions_standardized * standard_scaler.y_std) + standard_scaler.y_mean
    test_predictions = test_predictions.numpy().tolist()
    print(f"avg. test loss {test_loss}")
    logger.log_message(f"avg. test loss {test_loss}")
    
    #test_y_np = test_y_unscaled
    #test_y_np = np.squeeze(test_y_np, -1)
    y_true_list = test_y_unscaled.tolist()

    # if wandb is not provided, then automaticaly no tracking is executed
    save_predictions_and_true_values_plot(y_true_list, 
                                          test_predictions, 
                                          tracking_enabled, 
                                          wandb=wandb,
                                          save_path=f"logs/log_figures/t_and_p/{EXPERIMENT_NAME}_targets_and_preds.png")
    
    # plot in detail 3 geomagnetic storms period from test set for different k-folds
    for detail_number in range (4):
        detail_start, detail_end, detail_name = get_detail_properties(K_FOLD, detail=detail_number)

        save_predictions_detail_plot(y_true_list, 
                                    test_predictions, 
                                    tracking_enabled, 
                                    wandb=wandb,
                                    save_path=f"logs/log_figures/pred_detail/{EXPERIMENT_NAME}_detail_{detail_number}.png",
                                    detail_start=detail_start,
                                    detail_end=detail_end,
                                    detail_name=detail_name)
    
    
    save_scatter_predictions_and_true_values(test_y_unscaled, 
                                             test_predictions, 
                                             tracking_enabled, 
                                             wandb=wandb,
                                             save_path=f"logs/log_figures/t_and_p_scatter/{EXPERIMENT_NAME}_targets_and_preds_scatter.png")
    
    save_scatter_predictions_and_true_values(test_y_unscaled, 
                                             test_predictions, 
                                             tracking_enabled,
                                             wandb=wandb, 
                                             save_path=f"logs/log_figures/t_and_p_scatter/{EXPERIMENT_NAME}_targets_and_preds_scatter.png")
    
    print(test_y_unscaled.shape)
    print(np.array(test_predictions).shape)
    r_squared = get_r_squared(test_y_unscaled, test_predictions)

    dst_rmse = get_dst_rmse(test_y_unscaled, test_predictions)

    print(f"R^2 on test set: {r_squared:.5f}")
    print(f"Dst RMSE on test set between targets and predictions: {dst_rmse:.5f}")
    print(f"lowest val. loss: {best_val:.5f}")
    logger.log_message(f"R^2 on test set: {r_squared:.5f}")
    logger.log_message(f"Dst RMSE on test set between targets and predictions: {dst_rmse:.5f}")
    logger.log_message(f"lowest val. loss: {best_val:.5f}")
    if tracking_enabled:
        lowest_val_loss_messeage = f"{best_val:.5f}" # for wandb
        wandb.run.summary['lowest_val_loss'] = lowest_val_loss_messeage
        wandb.run.summary['R_squared'] = r_squared
        wandb.run.summary['dst_rmse'] = dst_rmse
        wandb.finish()