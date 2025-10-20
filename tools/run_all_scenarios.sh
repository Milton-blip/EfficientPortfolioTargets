#!/bin/zsh
# Run EfficientPortfolioTargets scenarios for a supplied holdings file

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <holdings_filename>"
  exit 1
fi

HOLDINGS_FILE="$1"
PYTHON_PATH="./.venv/bin/python"
SCRIPT="PA2.py"

if [ ! -x "$PYTHON_PATH" ]; then
  echo "Python virtual environment not found: $PYTHON_PATH"
  exit 1
fi

if [ ! -f "$SCRIPT" ]; then
  echo "Cannot find $SCRIPT in current directory."
  exit 1
fi

echo "=== Running scenarios for holdings file: $HOLDINGS_FILE ==="

SCENARIOS=(0.06 0.08 0.10 0.12 0.14)
for VOL in $SCENARIOS; do
  echo "\n--- Scenario: target vol $VOL ---"
  $PYTHON_PATH $SCRIPT --target-vol $VOL --holdings "$HOLDINGS_FILE"
done

echo "\n--- Aggregate (unconstrained) scenario ---"
$PYTHON_PATH $SCRIPT --holdings "$HOLDINGS_FILE"

echo "\n✅ All scenarios complete."