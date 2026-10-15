#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:80}"

echo "Checking health endpoint at ${BASE_URL}/health..."
curl -sf "${BASE_URL}/health" > /dev/null

echo "Checking predict endpoint at ${BASE_URL}/predict..."
RESPONSE=$(curl -sf -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.0380759,
    "sex": 0.0506801,
    "bmi": 0.0616962,
    "bp": 0.0218723,
    "s1": -0.0442235,
    "s2": -0.0348208,
    "s3": -0.0434008,
    "s4": -0.0025923,
    "s5": 0.0199075,
    "s6": -0.0176461
  }')

echo "Prediction response: $RESPONSE"
echo "Smoke test passed successfully!"
