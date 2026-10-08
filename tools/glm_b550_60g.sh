#!/bin/sh
# B550's assembled REAP Q23 model: loopback by default, 60 GiB RAM, no swap, 2 GiB GPU reserve.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
unit=strata-glm-b550.service
host=127.0.0.1
case "${1:-start}" in
    stop) exec systemctl --user stop "$unit" ;;
    status) exec systemctl --user status "$unit" ;;
    start) ;;
    start-lan) host=0.0.0.0 ;;
    *) echo "usage: $0 [start|start-lan|stop|status]" >&2; exit 2 ;;
esac
keyfile="$root/build-hip-glm-v11/server-api.env"
set --
if [ -f "$keyfile" ]; then
    set -- "--property=EnvironmentFile=$keyfile"
elif [ "$host" = 0.0.0.0 ]; then
    echo "start-lan requires STRATA_API_KEY in $keyfile (mode 600)" >&2
    exit 2
fi
exec systemd-run "$@" --user --unit="$unit" \
    --property=WorkingDirectory="$root" \
    --property=MemoryMax=60G --property=MemorySwapMax=0 \
    --property=OOMPolicy=kill --property=KillMode=control-group \
    --property=IOAccounting=yes --property=TimeoutStopSec=30 \
    "$root/.venv-glm-hip/bin/python" "$root/tools/glm_guarded_serve.py" \
    "$root/configs/glm53f-reap50-q23-b550-60g-9070xt-experimental.json" \
    --ram-gib 60 --reserve-mib 2048 --host "$host" --port 8080 \
    --telemetry "$root/build-hip-glm-v11/server-memory.json"
