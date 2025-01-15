#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting installation process for Talker project...${NC}"

# Function to update git repository
update_repository() {
    echo -e "${YELLOW}Checking for repository updates...${NC}"
    
    # Check if we're in a git repository
    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo -e "${RED}Not a git repository. Skipping update.${NC}"
        return 1
    fi

    # Stash any local changes
    echo -e "${YELLOW}Stashing local changes if any...${NC}"
    git stash

    # Get the current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    
    # Fetch all changes
    echo -e "${YELLOW}Fetching updates...${NC}"
    if ! git fetch origin; then
        echo -e "${RED}Failed to fetch updates${NC}"
        git stash pop
        return 1
    fi

    # Update the current branch
    echo -e "${YELLOW}Updating branch: $CURRENT_BRANCH${NC}"
    if git pull origin $CURRENT_BRANCH; then
        echo -e "${GREEN}Repository updated successfully${NC}"
        
        # Apply stashed changes if any
        git stash pop > /dev/null 2>&1
    else
        echo -e "${RED}Failed to update repository${NC}"
        # Apply stashed changes if any
        git stash pop > /dev/null 2>&1
        return 1
    fi
}

# Function to check GPU and memory
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}NVIDIA GPU detected${NC}"
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
        echo -e "GPU Memory: ${GPU_MEM}MB"
        return 0
    else
        echo -e "${YELLOW}No NVIDIA GPU detected, running in CPU mode${NC}"
        return 1
    fi
}

# Update repository first
update_repository

# Check for root privileges
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if (( $(echo "$PYTHON_VERSION >= 3.11" | bc -l) )); then
        echo -e "${GREEN}Python $PYTHON_VERSION is already installed${NC}"
    else
        echo -e "${YELLOW}Python $PYTHON_VERSION found, but 3.11 or higher is required${NC}"
        
        # Try installing from default repositories first
        if apt-cache show python3.11 &> /dev/null; then
            echo -e "${GREEN}Python 3.11 found in default repositories${NC}"
            apt-get install -y python3.11 python3.11-venv python3.11-dev
        # Try backports if available
        elif grep -r "backports" /etc/apt/sources.list* &> /dev/null; then
            echo -e "${GREEN}Trying to install from backports...${NC}"
            apt-get -t $(lsb_release -cs)-backports install -y python3.11 python3.11-venv python3.11-dev
        # If not available in official repos, use deadsnakes as fallback
        else
            echo -e "${YELLOW}Python 3.11 not found in official repositories${NC}"
            echo -e "${YELLOW}Adding deadsnakes PPA (trusted Python repository maintained since 2009)${NC}"
            apt-get install -y software-properties-common
            add-apt-repository -y ppa:deadsnakes/ppa
            apt-get update
            apt-get install -y python3.11 python3.11-venv python3.11-dev
        fi
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
        update-alternatives --set python3 /usr/bin/python3.11
    fi
else
    echo -e "${YELLOW}Python 3 not found, checking repositories...${NC}"
    # Same logic as above for new installation
    if apt-cache show python3.11 &> /dev/null; then
        echo -e "${GREEN}Python 3.11 found in default repositories${NC}"
        apt-get install -y python3.11 python3.11-venv python3.11-dev
    elif grep -r "backports" /etc/apt/sources.list* &> /dev/null; then
        echo -e "${GREEN}Trying to install from backports...${NC}"
        apt-get -t $(lsb_release -cs)-backports install -y python3.11 python3.11-venv python3.11-dev
    else
        echo -e "${YELLOW}Python 3.11 not found in official repositories${NC}"
        echo -e "${YELLOW}Adding deadsnakes PPA (trusted Python repository maintained since 2009)${NC}"
        apt-get install -y software-properties-common
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update
        apt-get install -y python3.11 python3.11-venv python3.11-dev
    fi
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    update-alternatives --set python3 /usr/bin/python3.11
fi

# Install other system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    python3-pyaudio \
    curl \
    git

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Installing Ollama...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Function to start Ollama
start_ollama() {
    # Check if we're in a container or non-systemd environment
    if [ ! -d "/run/systemd/system" ]; then
        echo -e "${YELLOW}Running in container/non-systemd environment${NC}"
        # Start Ollama server in the background
        ollama serve > ollama.log 2>&1 &
        OLLAMA_PID=$!
        echo $OLLAMA_PID > ollama.pid
    else
        echo -e "${YELLOW}Starting Ollama service via systemd...${NC}"
        systemctl start ollama
    fi
}

# Function to check if Ollama is running
check_ollama() {
    curl -s http://localhost:11434/api/version &>/dev/null
    return $?
}

# Ensure Ollama is running
echo -e "${YELLOW}Ensuring Ollama is running...${NC}"
if ! check_ollama; then
    echo -e "${YELLOW}Starting Ollama...${NC}"
    start_ollama
    
    # Wait for service to fully start
    echo -e "${YELLOW}Waiting for Ollama to initialize...${NC}"
    sleep 5  # Initial wait

    # Check with retries
    MAX_RETRIES=5
    RETRY_COUNT=0
    while ! check_ollama && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo -e "${YELLOW}Waiting for Ollama to become responsive... ($(($RETRY_COUNT + 1))/$MAX_RETRIES)${NC}"
        sleep 5
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done

    if ! check_ollama; then
        echo -e "${RED}Failed to connect to Ollama after $MAX_RETRIES retries${NC}"
        echo -e "${YELLOW}Check ollama.log for details${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Ollama is already running${NC}"
fi

# Download required models
echo -e "${YELLOW}Downloading required models...${NC}"
if ! ollama pull llama3.2:latest; then
    echo -e "${RED}Failed to download llama3.2 model${NC}"
    exit 1
fi

if ! ollama pull nomic-embed-text:latest; then
    echo -e "${RED}Failed to download nomic-embed-text model${NC}"
    exit 1
fi

# Setup Python virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install requirements
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Check for GPU and install CUDA/DeepSpeed if available
if check_gpu; then
    echo -e "${YELLOW}Setting up CUDA environment...${NC}"
    # Check common CUDA paths (including Paperspace default)
    CUDA_PATHS=("/usr/local/cuda" "/usr/cuda" "/opt/cuda")
    for path in "${CUDA_PATHS[@]}"; do
        if [ -d "$path" ]; then
            export CUDA_HOME=$path
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
            echo -e "${GREEN}CUDA found at: $CUDA_HOME${NC}"
            break
        fi
    done

    if [ -n "$CUDA_HOME" ]; then
        echo -e "${YELLOW}Installing DeepSpeed...${NC}"
        pip install deepspeed
    fi
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp .env-example .env
    # Update model source to local
    sed -i 's/MODEL_SOURCE=openai/MODEL_SOURCE=local/' .env
    # Update model names
    echo "LLM_MODEL=llama3.2:latest" >> .env
    echo "EMBEDDING_MODEL=nomic-embed-text:latest" >> .env
fi

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}To activate the virtual environment, run: source venv/bin/activate${NC}"
exit 0
