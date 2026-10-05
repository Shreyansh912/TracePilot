"""
scripts/verify_env.py
Verifies PyTorch CUDA hardware acceleration, VRAM allocation, and basic tensor math.
"""
import sys
import torch
import numpy as np


def verify_environment() -> None:
    print(f"[*] Python Version: {sys.version.split()[0]}")
    print(f"[*] PyTorch Version: {torch.__version__}")
    print(f"[*] NumPy Version: {np.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"[*] CUDA Available: {cuda_available}")

    if not cuda_available:
        print("\n[!] WARNING: CUDA is NOT available to PyTorch.")
        print("    PyTorch was installed without CUDA support or your NVIDIA driver is missing.")
        print("    Verify your NVIDIA driver via 'nvidia-smi' in PowerShell.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cuda_capability = torch.cuda.get_device_capability(0)

    print(f"[*] Detected Devices: {device_count}")
    print(f"[*] Primary GPU: {device_name}")
    print(f"[*] Compute Capability: {cuda_capability[0]}.{cuda_capability[1]}")
    print(f"[*] Total Detected VRAM: {total_vram:.2f} GB")

    # Hardware Tensor Math Smoke Test
    print("\n[*] Executing GPU Tensor Allocation & Matmul Verification...")
    device = torch.device("cuda:0")

    # Allocate two 2048 x 2048 FP32 matrices directly on GPU
    # Matrix A: (2048, 2048) * 4 bytes ≈ 16 MB
    a = torch.randn(2048, 2048, device=device, dtype=torch.float32)
    b = torch.randn(2048, 2048, device=device, dtype=torch.float32)

    # Perform matrix multiplication
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Block host until GPU operation finishes

    allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
    cached_vram = torch.cuda.memory_reserved(0) / (1024 ** 2)

    print(f"[+] Matrix multiplication successful. Output shape: {tuple(c.shape)}")
    print(f"[+] Active Allocated VRAM: {allocated_vram:.2f} MB")
    print(f"[+] Cached Reserved VRAM: {cached_vram:.2f} MB")
    print("\n[SUCCESS] Hardware environment is ready for Transformer training.")


if __name__ == "__main__":
    verify_environment()