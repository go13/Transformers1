import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Print the CUDA device:
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # Print the version of CUDA
    cuda_version = torch.version.cuda
    print(f"CUDA Version: {cuda_version}")
else:
    print("CUDA is not available. Please check your installation.")