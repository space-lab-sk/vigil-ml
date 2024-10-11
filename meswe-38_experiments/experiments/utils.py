import torch
import numpy as np
import yaml
import os
import time
from datetime import datetime


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed():
    torch.manual_seed(42)
    np.random.seed(42)
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:.10f}'.format})
    torch.set_float32_matmul_precision("high")


def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


class Logger:
    def __init__(self, log_name):
        os.makedirs("logs", exist_ok=True)
        log_filename = f"logs/{log_name}.log"
        self.log_file = open(log_filename, "a")
        self.log_name = log_name
        
        self.log_file.write(f"{self.log_name}\n")
        self.log_file.write(f"logging started: {datetime.now()}\n")
        self.log_file.write("-" * 80 + "\n")
        self.log_file.flush() 
        
    def log_message(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"[{timestamp}] {msg}\n")
        self.log_file.flush() 
        
    def __del__(self):
        self.log_file.write(f"logging ended: {datetime.now()}\n")
        self.log_file.write("-" * 80 + "\n")
        self.log_file.flush() 
        self.log_file.close()