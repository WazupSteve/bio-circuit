#!/usr/bin/env bash
# verifiers 0.2.1 pins `mcp<2` for itself but generates its harness script with
# `mcp` UNPINNED, so uv resolves 2.0.0 into the harness's isolated environment.
# A 2.x client cannot complete the handshake against the 1.x server verifiers
# runs, and every rollout dies at 0 turns with a HarnessError whose traceback
# points at session.initialize().
#
# Downgrade mcp inside each harness environment. uv reuses them rather than
# re-resolving, because 1.x still satisfies the unpinned requirement.
#
#   bash scripts/fix_harness_mcp.sh
#
# Re-run after clearing the uv cache, or after any run that rebuilds one.
set -euo pipefail
found=0
for env in "$HOME/.cache/uv/environments-v2"/*/; do
    [ -x "$env/bin/python" ] || continue
    "$env/bin/python" -c 'import mcp' 2>/dev/null || continue
    version=$("$env/bin/python" -c 'import importlib.metadata as m; print(m.version("mcp"))')
    case "$version" in
        2.*)
            echo "downgrading mcp $version in $(basename "$env")"
            uv pip install --quiet --python "$env/bin/python" 'mcp<2'
            found=$((found + 1))
            ;;
    esac
done
if [ "$found" -eq 0 ]; then
    echo "no harness environment on mcp 2.x — nothing to do"
else
    echo "patched $found environment(s)"
fi
