#!/usr/bin/env bash
# Connection settings stay in ignored local configuration. See connection.env.example.
connection_config=${DS41_CONNECTION_CONFIG:-../../priv/ds41f_connection.env}
if [ -f "$connection_config" ]; then source "$connection_config"; fi
: "${GPU_HOST:?Set GPU_HOST in the local connection config}"
: "${GPU_ADDR:?Set GPU_ADDR in the local connection config}"
: "${B_BIN:?Set B_BIN in the local connection config}"
export GPU_HOST GPU_ADDR B_BIN
