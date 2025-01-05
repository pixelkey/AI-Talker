import os
import torch

def setup_cuda_environment():
    """Set up CUDA environment and test DeepSpeed availability"""
    # Set CUDA_HOME if not set
    if not os.environ.get('CUDA_HOME'):
        cuda_paths = [
            '/usr/lib/nvidia-cuda-toolkit',  # Debian/Ubuntu CUDA toolkit location
            '/usr/local/cuda',
            '/usr/cuda',
            '/opt/cuda'
        ]
        for path in cuda_paths:
            if os.path.exists(path):
                os.environ['CUDA_HOME'] = path
                print(f"\nSet CUDA_HOME to {path}")
                break

    # Test DeepSpeed availability
    try:
        import deepspeed
        print(f"\nDeepSpeed version: {deepspeed.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA arch list: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not set')}")
    except ImportError as e:
        print(f"\nDeepSpeed not available: {str(e)}")
    except Exception as e:
        print(f"\nError testing DeepSpeed: {str(e)}")
