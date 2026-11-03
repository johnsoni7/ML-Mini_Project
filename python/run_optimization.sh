#!/bin/zsh
# Start the unattended experiment controller, keeping the laptop awake. Keep it plugged in with the lid open.
cd "$(dirname "$0")/.."
export TF_USE_LEGACY_KERAS=1
exec caffeinate -dimsu .venv/bin/python python/auto_optimize.py --stop-at "${1:-2026-11-03T04:15:00+05:30}" --run-until "${4:-2026-11-03T05:20:00+05:30}" --slots "${2:-6}" --threads "${3:-2}"
