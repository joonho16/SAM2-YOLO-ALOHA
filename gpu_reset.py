from numba import cuda
import torch

torch.cuda.empty_cache()
device = cuda.get_current_device()
device.reset()