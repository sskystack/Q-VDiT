#!/usr/bin/env bash
set -euo pipefail

MIHOMO_BIN="${MIHOMO_BIN:-/usr/local/bin/mihomo}"
MIHOMO_HOME="${MIHOMO_HOME:-/root/.config/mihomo}"
MIHOMO_CONFIG="${MIHOMO_CONFIG:-$MIHOMO_HOME/config.yaml}"
MIHOMO_PID="${MIHOMO_PID:-/tmp/mihomo.pid}"
MIHOMO_LOG="${MIHOMO_LOG:-/tmp/mihomo.log}"

mkdir -p "$MIHOMO_HOME"

if [ ! -x "$MIHOMO_BIN" ]; then
  echo "mihomo binary not found: $MIHOMO_BIN" >&2
  exit 1
fi

if [ ! -f "$MIHOMO_CONFIG" ]; then
  echo "mihomo config not found: $MIHOMO_CONFIG" >&2
  exit 1
fi

if [ -f "$MIHOMO_PID" ] && kill -0 "$(cat "$MIHOMO_PID")" 2>/dev/null; then
  echo "mihomo already running with pid $(cat "$MIHOMO_PID")"
  exit 0
fi

nohup "$MIHOMO_BIN" -d "$MIHOMO_HOME" -f "$MIHOMO_CONFIG" >"$MIHOMO_LOG" 2>&1 &
echo $! >"$MIHOMO_PID"
echo "mihomo started, pid=$(cat "$MIHOMO_PID"), log=$MIHOMO_LOG"
