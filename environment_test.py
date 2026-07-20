import torch
print("--------------------------------------------------")
print(f"PyTorch 成功运行！当前版本: {torch.__version__}")
print(f"CUDA (GPU加速) 是否可用: {torch.cuda.is_available()}")
print("--------------------------------------------------")