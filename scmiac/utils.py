import pickle
import numpy as np
import torch

def savepkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def readpkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data



def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
