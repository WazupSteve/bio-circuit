#!/usr/bin/env bash
# One-command self-hosted subject model (replaces the Modal deployment).
#
#   scripts/subject.sh up                 # create Prime pod, install, serve, tunnel
#   scripts/subject.sh up ubuntu@host     # use an existing GPU box you can SSH into
#   scripts/subject.sh down               # terminate the Prime pod
#   SUBJECT_INSTANCE=kimi SUBJECT_PORT=8001 scripts/subject.sh up
#
# When it finishes, `export SUBJECT_URL=http://localhost:8000` and run as usual.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="$ROOT/environments/circuit_supernodes"
INSTANCE="${SUBJECT_INSTANCE:-}"
[[ "$INSTANCE" =~ ^[a-zA-Z0-9_-]*$ ]] || {
  echo "SUBJECT_INSTANCE may contain only letters, numbers, underscores, and hyphens" >&2
  exit 2
}
STATE="$ROOT/.subject${INSTANCE:+.$INSTANCE}" # holds pod id + ssh target
SSH_KEY="${SUBJECT_SSH_KEY:-$HOME/.ssh/primeintellect_ed25519}"
GPU="${SUBJECT_GPU:-A100_80GB}"
PORT="${SUBJECT_PORT:-8000}"
NAME="biocircuit-subject${INSTANCE:+-$INSTANCE}"

prime() { (cd "$ROOT" && uv run prime --plain "$@"); }
# Pods are ephemeral and IPs are recycled, so host keys go to a scratch file.
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30)
ssh_() { ssh "${SSH_OPTS[@]}" "$@"; }

create_pod() {
  echo "» creating $GPU pod on Prime" >&2
  local id output pod_id
  id=$(prime availability list --gpu-type "$GPU" --gpu-count 1 --output json \
       | jq -r '.gpu_resources | sort_by(.price_value) | .[0].id')
  if [[ -z "$id" || "$id" == "null" ]]; then
    echo "No $GPU pod is currently available on Prime" >&2
    return 1
  fi
  output=$(prime pods create --id "$id" --name "$NAME" --gpu-count 1 --disk-size 200 -y) || return 1
  pod_id=$(grep -oE '[0-9a-f]{32}' <<< "$output" | head -1 || true)
  if [[ -z "$pod_id" ]]; then
    echo "$output" >&2
    echo "Prime did not return a pod ID" >&2
    return 1
  fi
  echo "$pod_id" > "$STATE.pod"
  prime config set-ssh-key-path "$SSH_KEY" >/dev/null
  echo "» waiting for pod $(cat "$STATE.pod")" >&2
  until prime pods status "$(cat "$STATE.pod")" | grep -q ACTIVE; do sleep 15; done
  prime pods status "$(cat "$STATE.pod")" | awk '/^SSH/{print $2}'
}

up() {
  local target="${1:-}"
  if [[ -z "$target" ]] && ! target=$(create_pod); then
    rm -f "$STATE.pod"
    return 1
  fi
  echo "$target" > "$STATE.ssh"
  echo "» syncing code to $target"
  ssh_ "$target" 'curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; mkdir -p ~/bio-circuit'
  rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude .venv --exclude '*.jsonl.gz' --exclude outputs \
    "$ENV_DIR/" "$target:~/bio-circuit/circuit_supernodes/"
  echo "» installing GPU deps + starting server (first boot downloads ~57 GB of weights)"
  ssh_ "$target" 'cd ~/bio-circuit/circuit_supernodes
    [ -d .venv ] || ~/.local/bin/uv venv -p 3.11 -q .venv
    ~/.local/bin/uv pip install -q -p .venv/bin/python "circuit-tracer==0.5.0" "nnsight==0.6.1" \
      "torch==2.8.0" "transformer-lens==2.16.1" "huggingface_hub[hf_transfer]<1" \
      fastapi uvicorn typer pydantic tenacity httpx
    [ -f ~/subject.pid ] && kill "$(cat ~/subject.pid)" 2>/dev/null; sleep 2
    HF_HUB_ENABLE_HF_TRANSFER=1 nohup .venv/bin/python -m modal_app.server --port '"$PORT"' > ~/subject.log 2>&1 &
    echo $! > ~/subject.pid'
  echo "» opening tunnel localhost:$PORT"
  close_tunnel
  ssh "${SSH_OPTS[@]}" -N -L "$PORT:localhost:$PORT" "$target" >/dev/null 2>&1 &
  disown
  echo $! > "$STATE.tunnel"
  until curl -sf -m 3 "localhost:$PORT/health" >/dev/null; do
    ssh_ "$target" 'grep -qE "Traceback|Error" ~/subject.log' && { ssh_ "$target" 'tail -40 ~/subject.log'; exit 1; }
    sleep 20
  done
  echo
  echo "✓ subject ready (tunnel pid $(cat "$STATE.tunnel")).   export SUBJECT_URL=http://localhost:$PORT"
}

close_tunnel() {
  [[ -f "$STATE.tunnel" ]] && { kill "$(cat "$STATE.tunnel")" 2>/dev/null || true; rm -f "$STATE.tunnel"; }
  return 0
}

down() {
  close_tunnel
  if [[ -f "$STATE.pod" ]]; then
    prime pods terminate "$(cat "$STATE.pod")" -y
    rm -f "$STATE.pod"
  fi
  rm -f "$STATE.ssh"
  echo "✓ torn down"
}

case "${1:-}" in
  up) up "${2:-}" ;;
  down) down ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
