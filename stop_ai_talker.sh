#!/bin/bash
# Script to stop AI-Talker
pkill -f "/home/andrew/projects/app/python/ai-talker/scripts/main.py" >> "/home/andrew/projects/app/python/ai-talker/logs/cron.log" 2>&1
