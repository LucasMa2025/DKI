#!/bin/bash

# DKI API Server Startup Script
# 
# Usage:
#   ./scripts/start_api.sh              # Start API server
#   ./scripts/start_api.sh --port 8080  # Custom port
#   ./scripts/start_api.sh --help       # Show help
#
# Redis is automatically enabled/disabled based on config/config.yaml
# No need for --redis flag - just set redis.enabled: true in config

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
            echo "DKI API Server Startup Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --host HOST     Server host (default: 0.0.0.0)"
            echo "  --port PORT     Server port (default: 8000)"
            echo "  --config PATH   Config file path (default: config/config.yaml)"
            echo "  --help, -h      Show this help"
            echo ""
            echo "Redis Configuration:"
            echo "  Redis is automatically enabled based on config.yaml"
            echo "  Set redis.enabled: true in config/config.yaml to enable"
            echo ""
            echo "Examples:"
            echo "  $0                          # Start with defaults"
            echo "  $0 --port 8080              # Custom port"
            echo "  $0 --config custom.yaml     # Custom config"
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
echo "  DKI API Server"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Host:   $HOST"
echo "  Port:   $PORT"
echo "  Config: $CONFIG"
echo "  Redis:  $REDIS_ENABLED"
echo ""
echo "Endpoints:"
echo "  Health: http://$HOST:$PORT/health"
echo "  Chat:   http://$HOST:$PORT/v1/dki/chat"
echo "  Stats:  http://$HOST:$PORT/api/stats"
echo "  Docs:   http://$HOST:$PORT/docs"
echo ""
echo "=========================================="
echo ""

# Start the server
exec python main.py api \
    --host "$HOST" \
    --port "$PORT" \
    --config "$CONFIG"
