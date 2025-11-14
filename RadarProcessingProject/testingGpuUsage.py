import torch, platform, sys
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
