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

    # Check for untracked files first
    UNTRACKED_FILES=$(git ls-files --others --exclude-standard)
    if [ ! -z "$UNTRACKED_FILES" ]; then
        echo -e "${YELLOW}Untracked files detected that might be overwritten:${NC}"
        echo "$UNTRACKED_FILES"
        echo -e "${YELLOW}Would you like to remove these files and continue? (y/N)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Removing untracked files...${NC}"
            git clean -f
        else
            echo -e "${YELLOW}Update aborted by user${NC}"
            return 1
        fi
    fi

    # Stash any tracked changes
    echo -e "${YELLOW}Stashing any tracked changes...${NC}"
    git stash

    # Get the current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    
    # Fetch and merge
    echo -e "${YELLOW}Fetching updates...${NC}"
    if git fetch origin && git reset --hard origin/$CURRENT_BRANCH; then
        echo -e "${GREEN}Repository updated successfully${NC}"
        # Apply stashed changes if any
        git stash pop > /dev/null 2>&1 || true
    else
        echo -e "${RED}Failed to update repository${NC}"
        # Apply stashed changes if any
        git stash pop > /dev/null 2>&1 || true
        return 1
    fi
}

# Check Python version without using bc
check_python_version() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; ver=sys.version_info; print(f"{ver.major}.{ver.minor}")')
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -gt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]); then
            echo -e "${GREEN}Python $PYTHON_VERSION is already installed${NC}"
            return 0
        else
            echo -e "${YELLOW}Python $PYTHON_VERSION found, but 3.11 or higher is required${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}Python 3 not found${NC}"
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
if ! check_python_version; then
    # Add deadsnakes PPA for Python 3.11
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.11 python3.11-venv python3.11-dev
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
echo -e "${YELLOW}Checking Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating new virtual environment...${NC}"
    python3 -m venv venv
elif [ "$1" == "--fresh" ]; then
    echo -e "${YELLOW}--fresh flag detected. Removing existing virtual environment...${NC}"
    rm -rf venv
    echo -e "${YELLOW}Creating new virtual environment...${NC}"
    python3 -m venv venv
else
    echo -e "${GREEN}Using existing virtual environment${NC}"
    echo -e "${YELLOW}(Use --fresh flag to create a new environment if needed)${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Verify virtual environment is active
if [[ "$VIRTUAL_ENV" != *"venv"* ]]; then
    echo -e "${RED}Failed to activate virtual environment${NC}"
    exit 1
fi

# Upgrade pip and install requirements
echo -e "${YELLOW}Installing Python dependencies...${NC}"
python3 -m pip install --upgrade pip wheel setuptools

# Check for GPU and set up CUDA environment
if check_gpu; then
    echo -e "${YELLOW}Setting up CUDA environment...${NC}"
    
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    echo -e "${GREEN}Detected CUDA version: $CUDA_VERSION${NC}"

    # Set CUDA environment variables
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# Install all dependencies from requirements.txt
echo -e "${YELLOW}Installing dependencies individually from requirements.txt...${NC}"

# Read requirements.txt line by line and install each package
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Remove leading/trailing whitespace
    line=$(echo "$line" | xargs)
    
    echo -e "${YELLOW}Installing $line...${NC}"
    if ! python3 -m pip install --no-cache-dir "$line"; then
        echo -e "${RED}Failed to install $line${NC}"
        echo -e "${YELLOW}Retrying with --no-deps flag...${NC}"
        if ! python3 -m pip install --no-cache-dir --no-deps "$line"; then
            echo -e "${RED}Failed to install $line even with --no-deps. Continuing with next package...${NC}"
        fi
    fi
done < requirements.txt

# Install NLTK data
echo -e "${YELLOW}Installing NLTK data...${NC}"
python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

# Verify critical packages are installed
echo -e "${YELLOW}Verifying installations...${NC}"
REQUIRED_PACKAGES=("python-dotenv" "langchain" "langchain-community" "langchain-openai" "openai" "gradio" "ollama" "faiss-cpu" "nltk" "tortoise-tts" "deepspeed")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -m pip show "$package" > /dev/null 2>&1; then
        echo -e "${RED}Critical package $package is not installed${NC}"
        exit 1
    fi
done

# Create required directories
echo -e "${YELLOW}Creating required directories...${NC}"
mkdir -p ingest embeddings
touch ingest/.gitkeep

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
