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