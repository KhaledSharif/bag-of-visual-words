#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Running tests with coverage..."
# Run pytest and capture output while also displaying it
pytest 2>&1 | tee pytest_output.txt

# Extract test count from output
TEST_COUNT=$(grep -oP '\d+(?= passed)' pytest_output.txt | head -1)

# Extract coverage percentage from the TOTAL line (handles decimals like 76.43%)
COVERAGE=$(grep "TOTAL" pytest_output.txt | grep -oP '\d+\.\d+%|\d+%' | tail -1)

# Clean up temporary file
rm pytest_output.txt

# Provide default values if extraction failed
if [ -z "$TEST_COUNT" ]; then
    TEST_COUNT="unknown"
fi

if [ -z "$COVERAGE" ]; then
    COVERAGE="unknown%"
fi

echo ""
echo "=========================================="
echo "Extracted values:"
echo "  Test count: $TEST_COUNT"
echo "  Coverage: $COVERAGE"
echo "  Commit message would be: \"$TEST_COUNT tests $COVERAGE coverage\""
echo "=========================================="
