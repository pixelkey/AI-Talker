#!/bin/bash
# Script to run AI-Talker with proper environment
cd "/home/andrew/projects/app/python/ai-talker"
export PATH=$PATH:/usr/bin:/usr/local/bin
export PYTHONPATH="/home/andrew/projects/app/python/ai-talker"
export HOME="/home/andrew"

# Activate virtual environment and run the script
source "/home/andrew/projects/app/python/ai-talker/venv/bin/activate"
python "/home/andrew/projects/app/python/ai-talker/scripts/main.py" >> "/home/andrew/projects/app/python/ai-talker/logs/cron.log" 2>&1
