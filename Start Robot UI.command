#!/bin/bash

# -------------------------------------------------------
# FIBO Robot Cafe Studio — one-click launcher
# Double-click this file in Finder to start everything.
# Close the Terminal window (or press Ctrl-C) to stop.
# -------------------------------------------------------

PROJECT="/Users/dome/robot-ui"
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

cd "$PROJECT" || { echo "ERROR: Cannot find $PROJECT"; read -r; exit 1; }

echo "============================================"
echo "  FIBO Robot Cafe Studio"
echo "  Starting UI (port 3001) + ROS relay (9090)"
echo "============================================"
echo ""

npm run dev:full
