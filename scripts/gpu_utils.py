import subprocess
import logging
import time

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
    """Check if the GPU temperature exceeds the threshold.
    If temperature is too high, check every second for up to 10 seconds for cooling."""
    for attempt in range(10):
        temp = get_gpu_temperature()
        if temp is None:
            logger.warning("Could not get GPU temperature")
            return False
            
        is_hot = temp > max_temp
        logger.info(f"GPU temperature (attempt {attempt + 1}/10): {temp}°C (threshold: {max_temp}°C)")
        
        if not is_hot:
            return False
            
        if attempt < 9:  # Don't sleep on the last attempt
            time.sleep(1)
            logger.info("Waiting 1 second for cooling...")
            
    return True  # Still too hot after all attempts
