import torch

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")

def init_gpu_mac(use_gpu=True, gpu_id=0):
    global device
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using NVIDIA GPU (CUDA) id {gpu_id}")
    elif use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        if use_gpu:
            print("GPU not detected (neither CUDA nor MPS). Defaulting to CPU.")
        else:
            print("GPU usage disabled. Using CPU.")

# 别忘了在程序开始时调用它
# init_gpu(use_gpu=True)

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
