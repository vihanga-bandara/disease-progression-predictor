#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:80}"
JSON="Content-Type: application/json"
PAYLOAD='{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
MISSING_S6='{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}'

health=$(curl -sf "$BASE_URL/health")
echo "health: $health"
echo "$health" | grep -q '"status":"ok"'

prediction=$(curl -sf -X POST "$BASE_URL/predict" -H "$JSON" -d "$PAYLOAD")
echo "predict: $prediction"
echo "$prediction" | grep -Eq '"prediction":-?[0-9]'

status=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/predict" -H "$JSON" -d "$MISSING_S6")
echo "missing s6: HTTP $status"
test "$status" = "422"

echo "smoke test passed"
