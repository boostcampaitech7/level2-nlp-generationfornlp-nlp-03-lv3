import torch
import gc

# GPU 메모리 강제 초기화
torch.cuda.empty_cache()
gc.collect()
