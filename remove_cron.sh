#!/bin/bash

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$PROJECT_DIR/scripts/main.py"

# Create a temporary file
crontab -l 2>/dev/null | grep -v "AI-TALKER-" > /tmp/current_cron

# Install new crontab without the AI-Talker entries
crontab /tmp/current_cron

# Clean up
rm /tmp/current_cron

# Remove wrapper scripts if they exist
rm -f "$PROJECT_DIR/run_ai_talker.sh" "$PROJECT_DIR/stop_ai_talker.sh"

echo "AI-Talker cron jobs have been removed"
