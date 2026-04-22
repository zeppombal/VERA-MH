#!/usr/bin/env bash
# Run README "Recommended Settings" via run_pipeline.py (generation → judge → score).
# See README.md § Recommended Settings: 100 personas, 30 turns, two user-agent suites
# (GPT 5.2 and Claude Opus 4.5) with one conversation per persona each (200 total),
# judged with GPT-4o and Claude Sonnet 4.5.
#
# After both pipelines complete, merges the two evaluation folders into one pooled
# score bundle (results.csv + scores/) via scripts/pool_vera_scores.py.
#
# Usage:
#   ./scripts/run_recommended_vera_pipeline.sh <provider-agent-model> [extra run_pipeline.py args...]
#
# Examples:
#   ./scripts/run_recommended_vera_pipeline.sh gpt-4o
#   ./scripts/run_recommended_vera_pipeline.sh claude-sonnet-4-5-20250929 --max-concurrent 10
#
# Optional environment:
#   VERA_OUTPUT_PARENT      Parent directory for new p_* folders (default: output)
#   VERA_MAX_CONCURRENT     Passed as --max-concurrent when set
#   VERA_MAX_PERSONAS     Passed as --max-personas when set (for dry runs; omit for all personas)
#   VERA_POOL_OUTPUT        Parent directory for pooled j_pooled__* folder (default: same as VERA_OUTPUT_PARENT)
#   VERA_POOL_SKIP_RISK     If non-empty, skip pooled risk-level analysis (--skip-risk-analysis)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <provider-agent-model> [extra run_pipeline.py arguments...]" >&2
  exit 2
fi

PROVIDER_AGENT="$1"
shift

OUTPUT_PARENT="${VERA_OUTPUT_PARENT:-output_test}"
# USER_GPT="${VERA_USER_GPT:-gpt-5.2}"
USER_GPT="${VERA_USER_GPT:-gpt-5.4-nano}"
# USER_CLAUDE="${VERA_USER_CLAUDE:-claude-opus-4-5-20251101}"
USER_CLAUDE="${VERA_USER_CLAUDE:-claude-haiku-4-5-20251001}"
# JUDGE_GPT="${VERA_JUDGE_GPT:-gpt-4o}"
JUDGE_GPT="${VERA_JUDGE_GPT:-gpt-4o-mini}"
# JUDGE_CLAUDE="${VERA_JUDGE_CLAUDE:-claude-sonnet-4-5-20250929}"
JUDGE_CLAUDE="${VERA_JUDGE_CLAUDE:-claude-haiku-4-5-20251001}"

POOL_PARENT="${VERA_POOL_OUTPUT:-$OUTPUT_PARENT}"

# TODO set max concurrency

COMMON_ARGS=(
  --provider-agent "$PROVIDER_AGENT"
  --turns 4
  --runs 1
  --judge-model "$JUDGE_GPT" "$JUDGE_CLAUDE"
  --output "$OUTPUT_PARENT"
)

if [[ -n "${VERA_MAX_CONCURRENT:-}" ]]; then
  COMMON_ARGS+=(--max-concurrent "$VERA_MAX_CONCURRENT")
fi
if [[ -n "${VERA_MAX_PERSONAS:-}" ]]; then
  COMMON_ARGS+=(--max-personas "$VERA_MAX_PERSONAS")
fi

POOL_ARGS=(-o "$POOL_PARENT")
if [[ -n "${VERA_POOL_SKIP_RISK:-}" ]]; then
  POOL_ARGS+=(--skip-risk-analysis)
fi

run_pipeline_capture_eval() {
  local log
  log="$(mktemp)"
  # Send pipeline transcript to stderr so command substitution only captures the path.
  python3 run_pipeline.py "$@" 2>&1 | tee "$log" >&2
  local st="${PIPESTATUS[0]}"
  if [[ "$st" -ne 0 ]]; then
    rm -f "$log"
    return "$st"
  fi
  local ev
  ev="$(python3 "$SCRIPT_DIR/pool_vera_scores.py" --extract-from-log "$log")" || {
    rm -f "$log"
    return 1
  }
  rm -f "$log"
  printf '%s\n' "$ev"
}

echo "== VERA recommended pipeline: user $USER_GPT → provider $PROVIDER_AGENT =="
EVAL_GPT="$(run_pipeline_capture_eval --user-agent "$USER_GPT" "${COMMON_ARGS[@]}" "$@")"

echo ""
echo "== VERA recommended pipeline: user $USER_CLAUDE → provider $PROVIDER_AGENT =="
EVAL_CLAUDE="$(run_pipeline_capture_eval --user-agent "$USER_CLAUDE" "${COMMON_ARGS[@]}" "$@")"

echo ""
echo "== Pooling evaluation scores into $POOL_PARENT =="
python3 "$SCRIPT_DIR/pool_vera_scores.py" "${POOL_ARGS[@]}" "$EVAL_GPT" "$EVAL_CLAUDE"
