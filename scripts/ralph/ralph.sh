#!/usr/bin/env bash
# Ralph autonomous development loop.
# Usage:
#   scripts/ralph/ralph.sh [--agent codex|claude|amp|kimi] [--unsafe] [--dry-run] [max_iterations]
#   scripts/ralph/ralph.sh --agent NAME --agent-command PATH [--unsafe] [--dry-run] [max_iterations]
#
# Permission modes:
#   Default runs each built-in agent unattended but bounded:
#     codex: --sandbox workspace-write --ask-for-approval never
#            (writes confined to the workspace; nothing prompts)
#     claude: --permission-mode acceptEdits
#     amp: no bypass flags
#     kimi: -p already applies the auto permission policy (no approval
#           prompts); static deny rules still apply. This is NOT a sandbox:
#           the agent runs with the user's privileges.
#   --unsafe forwards the agent's bypass-approvals flag:
#     codex: --dangerously-bypass-approvals-and-sandbox
#     claude: --dangerously-skip-permissions
#     amp: --dangerously-allow-all
#     kimi: no-op (print mode is already auto-approved, and --yolo/--auto
#           are rejected when combined with -p)
#   Custom --agent-command adapters receive RALPH_UNSAFE=0|1 and decide for
#   themselves. Use --unsafe only inside an externally sandboxed environment
#   (container, VM, or disposable checkout).

set -euo pipefail

AGENT="${RALPH_AGENT:-codex}"
AGENT_COMMAND="${RALPH_AGENT_COMMAND:-}"
MAX_ITERATIONS=10
DRY_RUN=0
UNSAFE=0

usage() {
  cat <<'EOF'
Usage: scripts/ralph/ralph.sh [options] [max_iterations]

Options:
  --agent NAME          Built-in agent (codex, claude, amp, kimi) or custom label
  --tool NAME           Backward-compatible alias for --agent
  --agent-command PATH  Executable adapter for any other CLI agent
  --unsafe              Forward the agent's bypass-approvals flag (see header)
  --dry-run             Validate files and the selected agent without launching
  -h, --help            Show this help

A custom adapter is invoked from the repository root with the AGENTS.md prompt
file as its first argument. It must run one non-interactive agent turn and write
the agent output to stdout or stderr.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent|--tool)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; exit 2; }
      AGENT="$2"
      shift 2
      ;;
    --agent=*|--tool=*)
      AGENT="${1#*=}"
      shift
      ;;
    --agent-command)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; exit 2; }
      AGENT_COMMAND="$2"
      shift 2
      ;;
    --agent-command=*)
      AGENT_COMMAND="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --unsafe)
      UNSAFE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      else
        echo "Error: unknown argument '$1'" >&2
        usage >&2
        exit 2
      fi
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PRD_FILE="$REPO_ROOT/prd.json"
PROGRESS_FILE="$REPO_ROOT/progress.txt"
ARCHIVE_DIR="$REPO_ROOT/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"
PROMPT_FILE="$SCRIPT_DIR/AGENTS.md"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command '$1' was not found for agent '$AGENT'." >&2
    exit 1
  }
}

validate_agent() {
  if [[ -n "$AGENT_COMMAND" ]]; then
    require_command "$AGENT_COMMAND"
    return
  fi

  case "$AGENT" in
    codex|claude|amp|kimi)
      require_command "$AGENT"
      ;;
    *)
      echo "Error: unknown agent '$AGENT'." >&2
      echo "Use codex, claude, amp, kimi, or provide --agent-command PATH." >&2
      exit 1
      ;;
  esac
}

run_agent() {
  export RALPH_AGENT="$AGENT"
  export RALPH_REPO_ROOT="$REPO_ROOT"
  export RALPH_PROMPT_FILE="$PROMPT_FILE"
  export RALPH_ITERATION="$1"
  export RALPH_MAX_ITERATIONS="$MAX_ITERATIONS"
  export RALPH_UNSAFE="$UNSAFE"

  if [[ -n "$AGENT_COMMAND" ]]; then
    "$AGENT_COMMAND" "$PROMPT_FILE"
    return
  fi

  case "$AGENT" in
    codex)
      # --ask-for-approval is a top-level flag; placing it after `exec` is
      # rejected with "unexpected argument" (verified on 0.147.0-alpha).
      if [[ "$UNSAFE" -eq 1 ]]; then
        codex exec --dangerously-bypass-approvals-and-sandbox - < "$PROMPT_FILE"
      else
        codex --sandbox workspace-write --ask-for-approval never exec - < "$PROMPT_FILE"
      fi
      ;;
    claude)
      if [[ "$UNSAFE" -eq 1 ]]; then
        claude --dangerously-skip-permissions --print < "$PROMPT_FILE"
      else
        claude --permission-mode acceptEdits --print < "$PROMPT_FILE"
      fi
      ;;
    amp)
      if [[ "$UNSAFE" -eq 1 ]]; then
        amp --dangerously-allow-all < "$PROMPT_FILE"
      else
        amp < "$PROMPT_FILE"
      fi
      ;;
    kimi)
      # -p already applies the auto permission policy; --yolo and --auto are
      # rejected when combined with -p, so there is no bypass flag to forward.
      kimi -p "$(cat "$PROMPT_FILE")"
      ;;
  esac
}

read_branch_name() {
  [[ -f "$PRD_FILE" ]] || return 0
  if command -v jq >/dev/null; then
    jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || true
  elif command -v node >/dev/null; then
    node -e "const fs=require('fs'); try { const data=JSON.parse(fs.readFileSync(process.argv[1], 'utf8')); console.log(data.branchName || ''); } catch { process.exit(0); }" "$PRD_FILE"
  elif command -v python3 >/dev/null; then
    python3 -c "import json,sys; print((json.load(open(sys.argv[1])).get('branchName') or ''))" "$PRD_FILE" 2>/dev/null || true
  fi
}

has_json_reader() {
  command -v jq >/dev/null || command -v node >/dev/null || command -v python3 >/dev/null
}

all_stories_pass() {
  [[ -f "$PRD_FILE" ]] || return 1
  if command -v jq >/dev/null; then
    jq -e '(.userStories | length) > 0 and all(.userStories[]; .passes == true)' "$PRD_FILE" >/dev/null 2>&1
  elif command -v node >/dev/null; then
    node -e "const fs=require('fs'); try { const d=JSON.parse(fs.readFileSync(process.argv[1], 'utf8')); const s=d.userStories || []; process.exit(s.length > 0 && s.every(x => x.passes === true) ? 0 : 1); } catch { process.exit(1); }" "$PRD_FILE"
  elif command -v python3 >/dev/null; then
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); s=d.get('userStories') or []; sys.exit(0 if s and all(x.get('passes') is True for x in s) else 1)" "$PRD_FILE" 2>/dev/null
  else
    return 1
  fi
}

cd "$REPO_ROOT"
CURRENT_BRANCH="$(read_branch_name)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Ralph dry run"
  echo "  repo root: $REPO_ROOT"
  echo "  prd file: $PRD_FILE"
  echo "  progress file: $PROGRESS_FILE"
  echo "  archive dir: $ARCHIVE_DIR"
  echo "  branch: ${CURRENT_BRANCH:-<missing>}"
  echo "  agent: $AGENT"
  echo "  agent command: ${AGENT_COMMAND:-<built-in>}"
  echo "  unsafe mode: $UNSAFE"

  has_json_reader || { echo "Error: jq, node, or python3 is required to read prd.json" >&2; exit 1; }
  [[ -f "$PRD_FILE" ]] || { echo "Error: missing $PRD_FILE" >&2; exit 1; }
  [[ -n "$CURRENT_BRANCH" ]] || { echo "Error: prd.json branchName is missing" >&2; exit 1; }
  [[ -f "$PROGRESS_FILE" ]] || { echo "Error: missing $PROGRESS_FILE" >&2; exit 1; }
  [[ -d "$ARCHIVE_DIR" ]] || { echo "Error: missing $ARCHIVE_DIR" >&2; exit 1; }
  [[ -f "$PROMPT_FILE" ]] || { echo "Error: missing $PROMPT_FILE" >&2; exit 1; }
  validate_agent

  echo "Dry run OK"
  exit 0
fi

validate_agent

# A branch change means the previous run's prd.json was already overwritten.
# Its content is unrecoverable at this point, so do NOT archive the current
# (new) file under the old run's name. Archiving must happen before prd.json
# is replaced; see the ralph skill's "Archiving Previous Runs" section.
if [[ -f "$PRD_FILE" && -f "$LAST_BRANCH_FILE" ]]; then
  LAST_BRANCH="$(cat "$LAST_BRANCH_FILE" 2>/dev/null || true)"
  if [[ -n "$CURRENT_BRANCH" && -n "$LAST_BRANCH" && "$CURRENT_BRANCH" != "$LAST_BRANCH" ]]; then
    echo "Warning: prd.json branchName ($CURRENT_BRANCH) differs from the last run ($LAST_BRANCH)." >&2
    echo "The previous run's prd.json has already been overwritten and cannot be archived now." >&2
    echo "Archive prd.json and progress.txt BEFORE replacing them with a new run." >&2
  fi
fi

if [[ -n "$CURRENT_BRANCH" ]]; then
  echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
fi

if [[ ! -f "$PROGRESS_FILE" ]]; then
  {
    echo "# Ralph Progress Log"
    echo "Started: $(date)"
    echo "---"
  } > "$PROGRESS_FILE"
fi

echo "Starting Ralph - Agent: $AGENT - Max iterations: $MAX_ITERATIONS - Unsafe: $UNSAFE"

for i in $(seq 1 "$MAX_ITERATIONS"); do
  echo ""
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($AGENT)"
  echo "==============================================================="

  # Stream output live AND capture it via a plain temp file; process
  # substitution (tee >(...)) fails in restricted sandboxes without /dev/fd.
  LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/ralph-output.XXXXXX")"
  set +e
  run_agent "$i" 2>&1 | tee "$LOG_FILE"
  AGENT_STATUS=${PIPESTATUS[0]}
  set -e
  OUTPUT="$(<"$LOG_FILE")"
  rm -f "$LOG_FILE"

  if [[ "$AGENT" == "codex" && -z "$AGENT_COMMAND" ]]; then
    if printf '%s\n' "$OUTPUT" | grep -q "^assistant"; then
      OUTPUT="$(printf '%s\n' "$OUTPUT" | awk 'BEGIN{found=0} /^assistant/{found=1;next} {if(found) print}')"
    elif printf '%s\n' "$OUTPUT" | grep -q "^tokens used"; then
      OUTPUT="$(printf '%s\n' "$OUTPUT" | awk 'BEGIN{found=0} /^tokens used/{found=1} {if(found) print}')"
    fi
  fi

  if [[ "$AGENT_STATUS" -ne 0 ]]; then
    echo "" >&2
    echo "Error: agent '$AGENT' exited with status $AGENT_STATUS on iteration $i." >&2
    echo "Stopping instead of silently retrying. Inspect the output above, fix the cause, and re-run." >&2
    exit "$AGENT_STATUS"
  fi

  # The token only counts when it sits on a line by itself (surrounding
  # whitespace tolerated), so agents mentioning it inside a sentence do not
  # trigger a false completion.
  if printf '%s\n' "$OUTPUT" | grep -qE '^[[:space:]]*<promise>COMPLETE</promise>[[:space:]]*$'; then
    if all_stories_pass; then
      echo ""
      echo "Ralph completed all tasks!"
      echo "Completed at iteration $i of $MAX_ITERATIONS"
      exit 0
    fi
    echo "" >&2
    echo "Warning: agent emitted the COMPLETE token but prd.json still has stories with passes != true." >&2
    echo "Ignoring the token and continuing. Stories must not be marked passed without validation evidence." >&2
  fi

  echo "Iteration $i complete. Continuing..."
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all tasks."
echo "Check $PROGRESS_FILE for status."
exit 1
