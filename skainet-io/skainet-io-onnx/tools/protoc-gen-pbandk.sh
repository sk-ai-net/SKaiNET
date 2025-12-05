#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec java -jar "$DIR/protoc-gen-pbandk.jar" "$@"
