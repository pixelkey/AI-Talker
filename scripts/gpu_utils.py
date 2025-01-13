import subprocess
import logging

logger = logging.getLogger(__name__)

def get_gpu_temperature():
    """Get the current GPU temperature using nvidia-smi."""
    try:
        # Try to get GPU temperature using nvidia-smi
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=temperature.gpu', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            temp = float(result.stdout.strip())
            return temp
        return None
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        return None

def is_gpu_too_hot(max_temp=80):  # 75°C is a reasonable threshold
    """Check if the GPU temperature exceeds the threshold."""
    temp = get_gpu_temperature()
    if temp is not None:
        logger.info(f"Current GPU temperature: {temp}°C (threshold: {max_temp}°C)")
        return temp > max_temp
    logger.warning("Could not get GPU temperature")
    return False  # If we can't get temperature, assume it's safe to continue
