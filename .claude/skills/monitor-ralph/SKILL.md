---
name: monitor-ralph
description: Monitors Ralph autonomous development loops in real-time, catches validation violations before they're committed, performs root cause analysis when failures occur, and ensures Ralph follows the validate-design workflow. Proactively stops Ralph if bad patterns are detected.
allowed-tools: Read, Bash, Grep
---

# Monitor Ralph Autonomous Loop

## Purpose

Act as a vigilant assistant during Ralph autonomous development cycles:
1. Monitor execution and provide real-time commentary
2. Catch validation violations (pylint W0212, etc.) before commits
3. Root cause analysis when Ralph fails or produces bad code
4. Test fixes before restarting Ralph
5. Ensure Ralph invokes validate-design skill before code changes

## When to Invoke

**Invoke this skill when:**
- About to start a Ralph loop (pre-flight check)
- Ralph loop is running (monitor progress)
- Ralph loop completed (analyze results)
- Ralph produced unexpected output (debug)
- User asks "what's Ralph doing?" or "did Ralph work?"

## Core Monitoring Principles

### Principle 1: Be Proactive, Not Reactive

Stop Ralph IMMEDIATELY if you detect:
- Ralph using `._to_array()` or other protected methods
- Ralph NOT invoking `Skill(skill="validate-design")`
- Ralph making same mistake repeatedly (loop > 2)
- Validation would fail but Ralph hasn't caught it yet

### Principle 2: Always Do Root Cause Analysis

When something goes wrong:
1. Don't just describe the symptom
2. Trace back to WHY it happened
3. Identify what SHOULD have prevented it
4. Propose how to FIX the prevention mechanism

### Principle 3: Test Before Recommending

Before telling user to run Ralph again:
1. Manually test the fix (if applicable)
2. Verify skill files exist and are correct
3. Check that PROMPT.md has proper instructions
4. Ensure previous bad commits are reverted

## Monitoring Workflow

### Phase 1: Pre-Flight Check

Before Ralph starts, verify:

```bash
# Check skill exists
bash .claude/skills/monitor-ralph/scripts/check-status.sh skills

# Check PROMPT.md has skill invocations
grep -c "Skill(skill=\"validate-design\")" PROMPT.md
# Expect: >= 3 (should be in multiple places)

# Check validation config
grep "pylint.*tests" pyproject.toml || echo "WARNING: validation may not check tests/"

# Verify Ralph files committed
git status | grep -E "PROMPT.md|@fix_plan.md|@AGENT.md|validate-design"
# Expect: all committed, not in working directory
```

**Decision Point:**
- ✅ All checks pass → Proceed with Ralph
- ❌ Any check fails → Fix issues first, don't start Ralph

### Phase 2: Monitor During Execution

While Ralph runs, periodically check (every 30-60s):

```bash
# Get Ralph status
bash .claude/skills/monitor-ralph/scripts/check-status.sh ralph

# Check latest output
bash .claude/skills/monitor-ralph/scripts/latest-output.sh | tail -50
```

**Watch for these patterns:**

1. **Skill Invocation Check:**
   ```bash
   grep -i "Skill(skill=\"validate-design\")" logs/claude_output_*.log
   ```
   - ✅ Found → Ralph is following instructions
   - ❌ Not found → STOP RALPH IMMEDIATELY

2. **Protected Access Check:**
   ```bash
   bash .claude/skills/monitor-ralph/scripts/detect-violations.sh
   ```
   - ✅ No violations → Good
   - ❌ Violations found → STOP RALPH, revert commits

3. **Loop Progress:**
   ```bash
   tail -20 logs/ralph.log
   ```
   - Look for: files changed, commits made, validation status
   - Comment on what Ralph is doing

**Commentary Template:**
```
📊 Ralph Status (Loop #N, Xs elapsed):
- Phase: TEST_REPAIR / DEADCODE_ELIMINATION
- Skill invoked: YES/NO ⚠️
- Files modified: N
- Last action: <description>
- ✅ Good: <positive observations>
- ⚠️ Watch: <things to monitor>
- 🚨 Alert: <issues requiring immediate attention>
```

### Phase 3: Post-Execution Analysis

After Ralph completes a loop:

```bash
# Get final output
OUTPUT_FILE=$(ls -t logs/claude_output_*.log | head -1)
cat "$OUTPUT_FILE" | jq -r '.result' || cat "$OUTPUT_FILE"

# Check git commits
git log --oneline -5

# Validate latest commit
git show --stat HEAD
bash .claude/skills/monitor-ralph/scripts/detect-violations.sh HEAD
```

**Analysis Questions:**

1. **Did Ralph invoke the skill?**
   ```bash
   grep -c "Skill(skill=\"validate-design\")" "$OUTPUT_FILE"
   ```
   - If 0: ROOT CAUSE: Why didn't Ralph use the skill?

2. **Did Ralph follow skill's guidance?**
   - Read skill's recommendation from output
   - Check if Ralph's code matches recommendation
   - If mismatch: ROOT CAUSE: Why did Ralph ignore skill?

3. **Would validation pass?**
   ```bash
   uv run pylint tests/ 2>&1 | grep -c "W0212"
   ```
   - If > 0: ROOT CAUSE: Why did validation fail to catch this?

4. **Are commits clean?**
   ```bash
   git log --oneline -3
   # Should show: proper conventional commit messages
   # Should NOT show: "WIP", "fix", "try again"
   ```

**Output Format:**

```
RALPH LOOP ANALYSIS - Loop #N
==============================

EXECUTION SUMMARY:
- Duration: Xm Ys
- Loops completed: N
- Files modified: N
- Commits made: N
- Exit status: <reason>

SKILL USAGE:
- validate-design invoked: YES/NO
- Skill recommendation: <summary>
- Ralph followed guidance: YES/NO/PARTIAL

VALIDATION STATUS:
- pylint tests/: PASS/FAIL (N W0212 warnings)
- poe validate: PASS/FAIL
- Tests: PASS/FAIL (N failures)

COMMITS ANALYSIS:
- <commit hash> - <message> - ✅/❌ <assessment>

ROOT CAUSE (if failure):
- What failed: <description>
- Why it failed: <root cause>
- What should have prevented it: <missing safeguard>
- How to fix: <specific steps>

RECOMMENDATION:
- Revert commits: YES/NO - <which ones>
- Fix needed: <description>
- Safe to continue: YES/NO
- Next steps: <specific actions>
```

## Common Failure Patterns

### Pattern 1: Ralph Didn't Invoke Skill

**Symptom:** No "Skill(skill=" in output
**Root Cause:** PROMPT.md not loaded or instructions unclear
**Fix:**
1. Verify PROMPT.md committed: `git log --all -- PROMPT.md`
2. Check PROMPT.md has "MANDATORY" language
3. Restart Ralph

### Pattern 2: Ralph Used Protected Methods

**Symptom:** `._to_array()` in code, W0212 violations
**Root Cause:** Either skill not invoked OR Ralph ignored skill
**Fix:**
1. Check if skill was invoked
2. If yes: skill's guidance may be unclear - improve SKILL.md
3. If no: PROMPT.md not enforcing - strengthen language
4. Revert bad commits: `git reset --hard <good-commit>`

### Pattern 3: Validation Incomplete

**Symptom:** `poe validate` passes but `pylint tests/` fails
**Root Cause:** Validation config only checks `poc_homography/`
**Fix:**
1. Update `pyproject.toml`: `pylint = "pylint poc_homography tests"`
2. Update PROMPT.md to run `pylint tests/` separately
3. Update validate-design skill to check both

### Pattern 4: Files Not Committed

**Symptom:** `git reset` loses skill/PROMPT files
**Root Cause:** Files created but not committed before git operations
**Fix:**
1. Always commit infrastructure files immediately
2. Add to `.gitignore` if they shouldn't be committed
3. Test with `git status` before running Ralph

### Pattern 5: Skill Gives Wrong Guidance

**Symptom:** Skill recommends approach that later fails validation
**Root Cause:** Skill's knowledge incomplete or outdated
**Fix:**
1. Update `.claude/skills/validate-design/SKILL.md`
2. Add the failure case to examples
3. Test skill manually before re-running Ralph

## Root Cause Analysis Framework

When something goes wrong, use this framework:

### Step 1: Establish Facts
```bash
# What actually happened?
git log --oneline -5
git diff HEAD~1
uv run pylint tests/ 2>&1 | grep W0212
```

### Step 2: Trace Causality
- What action caused the failure?
- What should have prevented it?
- Why didn't the prevention work?

### Step 3: Identify Systemic Issue
- Is this a one-off mistake or systemic?
- Have we seen this pattern before?
- What's the root cause of the root cause?

### Step 4: Propose Multi-Layer Fix
- **Immediate:** Fix the current issue
- **Preventive:** Add safeguards to prevent recurrence
- **Systemic:** Address underlying architectural issue

**Example:**
```
FAILURE: Ralph used ._to_array() despite skill

Step 1 - Facts:
- Commit abc123 has ._to_array() calls
- pylint shows 32 W0212 warnings
- Output shows: No "Skill(skill=" found

Step 2 - Causality:
- Ralph didn't invoke skill
- PROMPT.md says "MANDATORY" but Ralph ignored
- Why? Possibly unclear or skill invocation syntax wrong

Step 3 - Systemic:
- This is NOT first time - happened in previous loop too
- Pattern: Ralph tends to skip "optional" seeming steps
- Root cause: PROMPT.md language not strong enough

Step 4 - Multi-layer fix:
- Immediate: Revert commit, add /validate-design to top of PROMPT
- Preventive: Add validation check in Ralph loop to verify skill usage
- Systemic: Create pre-commit hook to block W0212 violations
```

## Testing Fixes

Before recommending to restart Ralph:

### Test 1: Skill Works
```bash
cd ~/workspace/goclever/poc-homography/.worktrees/issue-172-ddd-refactor/
claude -p "Test scenario: matrix1 - matrix2 fails. Fix with ._to_array()? Use Skill(skill='validate-design')"
# Expect: Skill rejects ._to_array() approach
```

### Test 2: Files Committed
```bash
git status
# Expect: Clean working directory, all files committed
```

### Test 3: Validation Config
```bash
uv run poe validate
uv run pylint tests/
# Both must be run to catch all issues
```

### Test 4: PROMPT.md Has Instructions
```bash
grep -c "Skill(skill=\"validate-design\")" PROMPT.md
# Expect: >= 3 mentions
```

## Stopping Ralph

When to stop Ralph immediately:

1. **Protected method usage detected**
2. **Skill not being invoked (2+ loops)**
3. **Same mistake repeated (3+ times)**
4. **Circuit breaker opened**
5. **Validation would fail but Ralph hasn't noticed**

How to stop:
```bash
bash .claude/skills/monitor-ralph/scripts/check-status.sh stop
# Or manually:
pkill -f ralph_loop.sh
```

## Reference

See [reference.md](reference.md) for:
- Detailed failure case studies
- Complete monitoring checklist
- Advanced debugging techniques
