#!/bin/bash

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
RUN_SCRIPT="$PROJECT_DIR/scripts/main.py"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$RUN_SCRIPT"; then
    echo "AI-Talker cron jobs already exist. Skipping..."
    exit 0
fi

# Create a temporary file with existing crontab entries
crontab -l 2>/dev/null > /tmp/current_cron

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Create a wrapper script that sets up the environment properly
cat > "$PROJECT_DIR/run_ai_talker.sh" << EOF
#!/bin/bash
# Script to run AI-Talker with proper environment
cd "$PROJECT_DIR"
export PATH=\$PATH:/usr/bin:/usr/local/bin
export PYTHONPATH="$PROJECT_DIR"
export HOME="$(echo ~)"

# Activate virtual environment and run the script
source "$PROJECT_DIR/venv/bin/activate"
python "$RUN_SCRIPT" >> "$PROJECT_DIR/logs/cron.log" 2>&1
EOF

# Create a stop script
cat > "$PROJECT_DIR/stop_ai_talker.sh" << EOF
#!/bin/bash
# Script to stop AI-Talker
pkill -f "$RUN_SCRIPT" >> "$PROJECT_DIR/logs/cron.log" 2>&1
EOF

# Make scripts executable
chmod +x "$PROJECT_DIR/run_ai_talker.sh" "$PROJECT_DIR/stop_ai_talker.sh"

# Add cron jobs:
# 1. Start AI-Talker at 9:00 AM every day
echo "0 9 * * * $PROJECT_DIR/run_ai_talker.sh # AI-TALKER-START" >> /tmp/current_cron

# 2. Stop AI-Talker at 9:30 PM every day
echo "30 21 * * * $PROJECT_DIR/stop_ai_talker.sh # AI-TALKER-STOP" >> /tmp/current_cron

# Install the new crontab
crontab /tmp/current_cron

# Clean up
rm /tmp/current_cron

echo "AI-Talker cron jobs have been added successfully:"
echo "- Start at 9:00 AM"
echo "- Stop at 9:30 PM"
echo "- Wrapper scripts created with proper environment settings"
