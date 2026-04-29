#!/usr/bin/env bash
set -euo pipefail

MIHOMO_PID="${MIHOMO_PID:-/tmp/mihomo.pid}"

if [ ! -f "$MIHOMO_PID" ]; then
  echo "mihomo pid file not found"
  exit 0
fi

pid="$(cat "$MIHOMO_PID")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "stopped mihomo pid=$pid"
else
  echo "mihomo pid=$pid is not running"
fi

rm -f "$MIHOMO_PID"
