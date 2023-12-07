import torch

print(torch.__version__)
torch._utils

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device to Use: {}".format(device))