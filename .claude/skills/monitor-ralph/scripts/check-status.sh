#!/bin/bash
# Check Ralph/Claude process status

MODE="${1:-ralph}"

case "$MODE" in
  ralph)
    # Check if Ralph loop is running
    if pgrep -f "ralph_loop.sh" > /dev/null; then
      RALPH_PID=$(pgrep -f "ralph_loop.sh")
      RUNTIME=$(ps -p "$RALPH_PID" -o etime= 2>/dev/null | tr -d ' ')
      echo "✅ Ralph is running (PID: $RALPH_PID, Runtime: $RUNTIME)"

      # Check loop count from status.json
      if [[ -f "status.json" ]]; then
        LOOP_COUNT=$(jq -r '.loop_count // 0' status.json 2>/dev/null)
        CALLS=$(jq -r '.calls_made_this_hour // 0' status.json 2>/dev/null)
        echo "   Loop: #$LOOP_COUNT, API Calls: $CALLS"
      fi
      exit 0
    else
      echo "❌ Ralph is not running"
      exit 1
    fi
    ;;

  claude)
    # Check if Claude process is running
    if pgrep -f "claude" > /dev/null; then
      CLAUDE_PID=$(pgrep -f "claude" | head -1)
      echo "✅ Claude is running (PID: $CLAUDE_PID)"
      exit 0
    else
      echo "❌ Claude is not running"
      exit 1
    fi
    ;;

  skills)
    # Check if required skills exist
    MISSING=0

    if [[ ! -f ".claude/skills/validate-design/SKILL.md" ]]; then
      echo "❌ validate-design skill missing"
      MISSING=1
    else
      echo "✅ validate-design skill exists"
    fi

    if [[ ! -f "PROMPT.md" ]]; then
      echo "❌ PROMPT.md missing"
      MISSING=1
    else
      SKILL_REFS=$(grep -c "Skill(skill=\"validate-design\")" PROMPT.md 2>/dev/null || echo "0")
      if [[ $SKILL_REFS -ge 3 ]]; then
        echo "✅ PROMPT.md references skill ($SKILL_REFS times)"
      else
        echo "⚠️  PROMPT.md has only $SKILL_REFS skill references (expected >= 3)"
        MISSING=1
      fi
    fi

    exit $MISSING
    ;;

  stop)
    # Stop Ralph
    if pkill -f "ralph_loop.sh"; then
      echo "✅ Stopped Ralph"
      exit 0
    else
      echo "ℹ️  Ralph was not running"
      exit 1
    fi
    ;;

  *)
    echo "Usage: $0 {ralph|claude|skills|stop}"
    exit 1
    ;;
esac
