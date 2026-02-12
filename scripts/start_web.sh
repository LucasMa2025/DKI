#!/bin/bash

# DKI Web Server Startup Script (API + UI)
# 
# Usage:
#   ./scripts/start_web.sh              # Start web server
#   ./scripts/start_web.sh --port 8080  # Custom port
#   ./scripts/start_web.sh --help       # Show help
#
# Redis is automatically enabled/disabled based on config/config.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default values
HOST="0.0.0.0"
PORT=8000
CONFIG="config/config.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "DKI Web Server Startup Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --host HOST     Server host (default: 0.0.0.0)"
            echo "  --port PORT     Server port (default: 8000)"
            echo "  --config PATH   Config file path (default: config/config.yaml)"
            echo "  --help, -h      Show this help"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if config exists
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Read Redis status from config
REDIS_ENABLED=$(python3 -c "
import yaml
try:
    with open('$CONFIG', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    enabled = config.get('redis', {}).get('enabled', False)
    print('enabled' if enabled else 'disabled')
except Exception as e:
    print('disabled')
")

echo "=========================================="
echo "  DKI Web Server"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Host:   $HOST"
echo "  Port:   $PORT"
echo "  Config: $CONFIG"
echo "  Redis:  $REDIS_ENABLED"
echo ""
echo "=========================================="
echo ""

# Start the server
exec python main.py web \
    --host "$HOST" \
    --port "$PORT"
