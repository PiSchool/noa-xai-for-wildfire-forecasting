import random
import os
import numpy as np
import torch

def seed_everything(seed):
    #python seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy seed
    np.random.seed(seed)
    # pytorch seed for all devices (CPU & CUDA)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
