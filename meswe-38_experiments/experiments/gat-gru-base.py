import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import wandb

from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import argparse
from utils import set_seed, get_torch_device, load_config, \
    count_parameters, apply_glorot_xavier, inspect_gradient_norms
from utils import Logger
from preprocessing import get_k_fold, load_data, get_torch_data
from preprocessing import StandardScaler
from preprocessing import TimeSeriesDataset
from postprocessing import save_gradient_norms_plot, save_predictions_and_true_values_plot, \
    get_r_squared, save_scatter_predictions_and_true_values, \
    save_predictions_detail_plot, get_detail_properties, \
    get_dst_rmse


def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    mean_loss = total_loss / len(train_loader)
    return mean_loss

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
    mean_loss = total_loss / len(val_loader)
    return mean_loss, out

class TimeSeriesGNN(nn.Module):
    def __init__(self, num_features, hidden_channels, output_size, num_gru_layers=2, num_heads=4):
        super(TimeSeriesGNN, self).__init__()
        self.num_gru_layers = num_gru_layers
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        self.gat1 = GATConv(num_features, hidden_channels, heads=num_heads, concat=True, dropout=0.0)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=True, dropout=0.0)

        self.gru = nn.GRU(hidden_channels, hidden_channels, num_gru_layers, batch_first=True, bidirectional=False)

        self.linear = nn.Linear(hidden_channels, output_size)

    def forward(self, x, edge_index, batch=None):

        # x in  2d shape [batch_size * seq_len, features], holes in edge index makes sure that sequences are not interconected between batches,
        #  that could create information leak of target variable
        
        x = F.elu(self.gat1(x, edge_index)) 
        x = F.dropout(x, p=0.0, training=self.training)

        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=0.0, training=self.training)

        x = x.view(batch.max().item() + 1, -1, x.size(-1))  # back to 3d [batch_size, seq_len, features] shape for gru

        h0 = torch.zeros(self.num_gru_layers, x.size(0), self.hidden_channels).to(x.device)
        
        x, _ = self.gru(x, h0)

        x = x[:, -1, :]  # taking last state of gru, [batch_size, features]

        x = self.linear(x)
        
        # out in shape [batch_size, 1]
        return x
    


if __name__=="__main__":


    ########################################
    #PART 1: EXPERIMENT CONFIGURATION SETUP
    ########################################

    parser = argparse.ArgumentParser(description='Process a config file location.')
    parser.add_argument("-cfn", '--config_file_name', type=str, help='Path to the input config file')
    parser.add_argument("-dt", "--disable_tracking", action="store_false", help="Disable Weights and Biases tracking with its config file")
    args = parser.parse_args()

    config_file_name = args.config_file_name
    tracking_enabled = args.disable_tracking
    #tracking_enabled = True if tracking_enabled is None else False

    config = load_config(f"configs/gat-gru-configs/{config_file_name}")
    EXPERIMENT_NAME = config["logging"]["experiment_name"]
    EXPERIMENT_NOTES = config["logging"]["notes"]
    logger = Logger(EXPERIMENT_NAME)

    set_seed()
    device = get_torch_device()

    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["learning_rate"]
    NUM_EPOCHS = config["training"]["num_epochs"]
    WEIGHT_DECAY = config["training"]["weight_decay"]

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

    train_dataset = TimeSeriesDataset(train_X, train_y)
    val_dataset = TimeSeriesDataset(val_X, val_y)
    test_dataset = TimeSeriesDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    ###############################################
    #PART 3: DEEP LEARNING PART AND WANDB TRACKING
    ###############################################

    model = TimeSeriesGNN(num_features=INPUT_SIZE, hidden_channels=HIDDEN_CHANNELS, output_size=OUTPUT_SIZE, num_gru_layers=NUM_GRU_LAYERS)
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
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
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

    save_predictions_and_true_values_plot(y_true_list, 
                                          test_predictions, 
                                          tracking_enabled,
                                          wandb=wandb,
                                          save_path=f"logs/log_figures/t_and_p/{EXPERIMENT_NAME}_targets_and_preds.png")
    
    # TODO: get detailes
    detail_1_start, detail_1_end, detail_1_name = get_detail_properties(K_FOLD, detail=1)
    detail_2_start, detail_2_end, detail_2_name = get_detail_properties(K_FOLD, detail=2)

    save_predictions_detail_plot(y_true_list, 
                                 test_predictions, 
                                 tracking_enabled, 
                                 wandb=wandb,
                                 save_path=f"logs/log_figures/pred_detail/{EXPERIMENT_NAME}_detail_1.png",
                                 detail_start=20,
                                 detail_end=120,
                                 detail_name="Event XX")
    
    save_scatter_predictions_and_true_values(test_y_unscaled, 
                                             test_predictions, 
                                             tracking_enabled, 
                                             wandb=wandb,
                                             save_path=f"logs/log_figures/t_and_p_scatter/{EXPERIMENT_NAME}_targets_and_preds_scatter.png")
    
    r_squared = get_r_squared(test_y_unscaled, test_predictions)

    dst_rmse = get_dst_rmse(test_y_unscaled, test_predictions)

    print(f"R^2 on test set: {r_squared:.5f}")
    print(f"Dst RMSE on test set between targets and predictions: {dst_rmse:.5f}")
    print(f"lowest val. loss: {best_val:.5f}")
    logger.log_message(f"R^2 on test set: {r_squared:.5f}")
    logger.log_message(f"Dst RMSE on test set between targets and predictions: {dst_rmse:.5f}")
    logger.log_message(f"lowest val. loss: {best_val:.5f}")
    #if tracking_enabled:
        #lowest_val_loss_messeage = f"{best_val:.5f}" # for wandb
        #wandb.run.summary['lowest_val_loss'] = lowest_val_loss_messeage
        #wandb.run.summary['R_squared'] = r_squared
        #wandb.run.summary['dst_rmse'] = dst_rmse
        #wandb.finish()


    







    
