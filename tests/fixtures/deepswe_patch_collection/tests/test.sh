#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
cd /app
patch=/logs/artifacts/model.patch
if [ ! -s "$patch" ]; then
  echo "missing or empty model.patch"
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi
git config --global --add safe.directory /app
git apply --check "$patch"
git apply "$patch"
bytes=$(wc -c < "$patch" | tr -d ' ')
echo "[verifier] model.patch applied (${bytes} bytes)"
if [ "$(cat fixture_source.txt)" = "PATCHED_BY_FORGELOOP" ]; then
  echo "verifier observed ForgeLoop source state"
  echo 1 > /logs/verifier/reward.txt
else
  echo "verifier did not observe ForgeLoop source state"
  echo 0 > /logs/verifier/reward.txt
fi
