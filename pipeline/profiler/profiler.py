import torch 
import psutil 

def print_memory_usage(max_memory):
    print(f"\nMax GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**3} GB")
    print(f"Max GPU memory cached: {torch.cuda.max_memory_reserved() / 1024**3} GB")
    print(f"Max CPU memory allocated: {max_memory} GB")

def get_cpu_usage(max_memory):
    process = psutil.Process()
    max_memory = max(max_memory, process.memory_info().rss / 1024**3)
    return(max_memory)


def model_memory_cost(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**3
    print('Model size: {:.3f}GB'.format(size_all_mb))