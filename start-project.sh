#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/.logs"
mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "pnpm is required but not found in PATH."
  echo "Install pnpm and retry."
  exit 1
fi

for file in scripts/start-ai.sh scripts/start-backend.sh scripts/start-ingestion.sh scripts/start-scheduler.sh; do
  if [ ! -f "${ROOT_DIR}/${file}" ]; then
    echo "Missing ${file}. Run this script from project root."
    exit 1
  fi
done

PIDS=()

start_component() {
  local name="$1"
  local cmd="$2"
  local log_file="${LOG_DIR}/${name}.log"
  echo "Starting ${name}..."
  bash -lc "${cmd}" >"${log_file}" 2>&1 &
  PIDS+=("$!")
  echo "  pid=$! log=${log_file}"
}

cleanup() {
  echo ""
  echo "Stopping all components..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait || true
  echo "Stopped."
}

trap cleanup INT TERM EXIT

echo "Booting Governance Intel Platform..."
echo ""

start_component "ai" "cd \"${ROOT_DIR}\" && ./scripts/start-ai.sh"
start_component "api" "cd \"${ROOT_DIR}\" && ./scripts/start-backend.sh"
start_component "ingestion" "cd \"${ROOT_DIR}\" && ./scripts/start-ingestion.sh"
start_component "scheduler" "cd \"${ROOT_DIR}\" && ./scripts/start-scheduler.sh"
start_component "web" "cd \"${ROOT_DIR}\" && pnpm --dir apps/web dev"

echo ""
echo "All components launched."
echo "Web:       http://localhost:3000"
echo "API:       http://localhost:8000"
echo "AI:        http://localhost:8100"
echo "Scheduler: http://localhost:8200"
echo "Ingestion: http://localhost:8300"
echo ""
echo "Logs are in ${LOG_DIR}/"
echo "Press Ctrl+C to stop everything."
echo ""

wait
