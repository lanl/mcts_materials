#!/bin/bash
# Load Materials Project API key from .env file
#
# Usage:
#   source study/set_api_key.sh
#   # OR
#   . study/set_api_key.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    echo "✓ Loaded MP_API_KEY from .env"
    echo "  MP_API_KEY=${MP_API_KEY:0:10}... (${#MP_API_KEY} chars)"
else
    echo "ERROR: .env file not found at $ENV_FILE"
    echo ""
    echo "Create .env file with:"
    echo "  echo 'MP_API_KEY=your-key-here' > $ENV_FILE"
    echo ""
    echo "Or set directly:"
    echo "  export MP_API_KEY='your-key-here'"
    return 1
fi
