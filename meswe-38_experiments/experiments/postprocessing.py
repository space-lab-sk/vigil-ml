import matplotlib.pyplot as plt
import numpy as np


def get_dst_rmse(test_y, predictions):

    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    if isinstance(test_y, list):
        test_y = np.array(test_y)

    rmse = np.sqrt(np.mean(np.square(test_y-predictions)))
    return rmse


def get_r_squared(test_y: np.ndarray, predictions: np.ndarray):

    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    if isinstance(test_y, list):
        test_y = np.array(test_y)

    if predictions.ndim > 1:
        predictions = predictions.reshape(-1)

    if test_y.ndim > 1:
        test_y = test_y.reshape(-1)

    mean_true_values = np.mean(test_y)
    sst = np.sum((test_y - mean_true_values) ** 2)
    ssr = np.sum((test_y - predictions) ** 2)

    r_squared = 1 - (ssr / sst)
    return r_squared


def save_gradient_norms_plot(gradient_norms: list[float], tracking_enabled: bool, save_path:str, wandb=None,):
    plt.plot(gradient_norms)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm over Time')
    if wandb is not None and tracking_enabled:
        wandb.log({"Gradient Norm over Time": wandb.Image(plt)})
    plt.savefig(save_path)


def save_predictions_and_true_values_plot(y_true: list[float], predictions: list[float], tracking_enabled: bool, save_path:str, wandb=None):
    plt.figure(figsize=(20, 5))
    plt.plot(y_true, label="True values", linewidth=1, color="green")
    plt.plot(predictions, label="Prediction", color='orange', linewidth=1)

    plt.legend()
    plt.xlabel('index')
    plt.ylabel('Values')
    plt.grid(True)
    #plt.show()
    if wandb is not None and tracking_enabled:
        wandb.log({"Predicted and True Values": wandb.Image(plt)})
    
    plt.savefig(save_path)


def save_predictions_detail_plot(y_true: list[float], 
                                 predictions: list[float], 
                                 tracking_enabled: bool, 
                                 save_path:str, 
                                 detail_start: int, 
                                 detail_end: int,
                                 detail_name: str,
                                 wandb=None):
    
    plt.figure(figsize=(20, 5))
    plt.plot(y_true[detail_start:detail_end], label="True values", linewidth=0.5, color="green", marker='o', markersize=3)
    plt.plot(predictions[detail_start:detail_end], label="Prediction", linewidth=0.5, color="orange", marker='o', markersize=3)
    plt.legend()
    plt.xlabel('index')
    plt.ylabel('Values')
    plt.title(detail_name)
    plt.grid(True)
    #plt.show()
    if wandb is not None and tracking_enabled:
        wandb.log({detail_name: wandb.Image(plt)})
    
    plt.savefig(save_path)




def save_scatter_predictions_and_true_values(test_y: np.ndarray, predictions: np.ndarray, tracking_enabled: bool, save_path:str, wandb=None):

    if isinstance(test_y, list):
        test_y = np.array(test_y)
    
    if isinstance(predictions, list):
        predictions = np.array(predictions)

    test_y = test_y.flatten()
    predictions = predictions.flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(test_y, predictions, label="Predictions", alpha=0.5, color="orange")
    min_val = min(test_y.min(), predictions.min())
    max_val = max(test_y.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', label="Ideal")
    plt.xlabel("True Values [nT]")
    plt.ylabel("Predicted Values [nT]")
    plt.title("Predicted and True Values Scatter")
    plt.legend()
    #plt.show()
    if tracking_enabled:
        wandb.log({"Predicted and True Values Scatter": wandb.Image(plt)})
    
    plt.savefig(save_path)



def get_detail_properties(K_FOLD, detail):
    
    detail_start = 20
    detail_end = 100
    detail_name = "Event XX"

    return detail_start, detail_end, detail_name