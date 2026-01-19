# Monitor Ralph - Reference Guide

## Case Studies

### Case Study 1: The Protected Access Loop (2026-01-19)

**Situation:** Ralph ran Loop #1 and fixed 62 tests by adding `._to_array()` calls.

**What Happened:**
- Ralph completed successfully, all tests passed
- Created commit with message: "test: fix rotation matrix tests after VO-based refactoring"
- Validation (`poe validate`) reported success
- But hidden violation: 32 W0212 warnings in `tests/` directory

**Root Cause Chain:**
1. **Immediate:** Ralph used protected methods (`._to_array()`)
2. **Why not caught:** `poe validate` only checks `poc_homography/`, not `tests/`
3. **Why Ralph did it:** validate-design skill wasn't invoked
4. **Why skill not invoked:** Skill and PROMPT.md weren't committed to git
5. **Why not committed:** Created in `/tmp`, then `git reset --hard` wiped them

**Lessons:**
- Always commit infrastructure files immediately
- Validation must check BOTH production and test code
- Skills are useless if not in the project when Ralph runs
- Test manually before running autonomous loops

**Fixes Applied:**
1. Created skill in `.claude/skills/validate-design/` (in project)
2. Committed PROMPT.md, skill files immediately
3. Added monitor-ralph skill to catch these issues
4. Updated PROMPT.md to emphasize `Skill(skill="validate-design")` invocation

### Case Study 2: Skill Invocation Syntax (2026-01-19)

**Situation:** PROMPT.md said `/validate-design` instead of `Skill(skill="validate-design")`

**What Happened:**
- Used slash-command syntax (wrong for skills)
- Proper syntax is: `Skill(skill="name")`
- Ralph couldn't find the skill

**Root Cause:**
- Unfamiliarity with Claude Code skill invocation syntax
- Didn't consult skill-expert documentation

**Fix:**
- Updated all PROMPT.md references to use `Skill(skill="validate-design")`
- Verified with manual test that skill works

## Monitoring Checklist

### Before Starting Ralph

- [ ] Skills exist: `.claude/skills/validate-design/SKILL.md`
- [ ] PROMPT.md exists and committed
- [ ] PROMPT.md references skill >= 3 times
- [ ] @fix_plan.md exists
- [ ] @AGENT.md exists
- [ ] All Ralph files committed (not in working dir)
- [ ] Previous bad commits reverted
- [ ] Validation config checks both prod and test code

**Script:** `bash .claude/skills/monitor-ralph/scripts/check-status.sh skills`

### During Ralph Execution (every 30-60s)

- [ ] Ralph still running
- [ ] Loop progressing (check logs/ralph.log)
- [ ] Check latest output for skill invocation
- [ ] No violations detected yet
- [ ] Commits being made (if applicable)

**Scripts:**
```bash
bash .claude/skills/monitor-ralph/scripts/check-status.sh ralph
bash .claude/skills/monitor-ralph/scripts/latest-output.sh | tail -50
bash .claude/skills/monitor-ralph/scripts/detect-violations.sh
```

### After Loop Completes

- [ ] Check if skill was invoked
- [ ] Verify commits don't have violations
- [ ] Run full validation (including tests/)
- [ ] Review commit messages
- [ ] Analyze whether Ralph followed skill guidance

**Scripts:**
```bash
bash .claude/skills/monitor-ralph/scripts/detect-violations.sh HEAD
uv run pylint tests/
git show --stat HEAD
```

## Common Patterns to Watch

### Pattern: Ralph Skips "Optional" Steps

**Symptom:** PROMPT.md says "MANDATORY" but Ralph ignores it

**Why:** Language not strong enough, or buried in text

**Fix:**
- Put critical instructions at TOP of file
- Use emojis/formatting: "🚨 MANDATORY"
- Repeat instruction at multiple points
- Make it first instruction in relevant section

### Pattern: Files Not Persisted

**Symptom:** Files exist, then disappear after git operation

**Why:** Created but not committed before `git reset`

**Fix:**
- Commit infrastructure files IMMEDIATELY after creation
- Verify with `git status` before risky operations
- Use `git stash` if need to preserve uncommitted work

### Pattern: Validation Gaps

**Symptom:** Validation passes but code has violations

**Why:** Validation config incomplete

**Fix:**
- Always run `pylint tests/` separately
- Update `pyproject.toml` to include tests in validation
- Document in @AGENT.md that BOTH must be run

### Pattern: Skill Exists But Not Used

**Symptom:** Skill file present, but Ralph doesn't invoke it

**Why:** PROMPT.md instructions unclear or syntax wrong

**Fix:**
- Use exact syntax: `Skill(skill="name")`
- Make invocation mandatory (not optional)
- Test manually first: `claude -p "Use Skill(skill='name')"`

## Debugging Commands

### Check Ralph's Current State
```bash
# Is Ralph running?
pgrep -f ralph_loop.sh

# What loop is it on?
cat status.json | jq

# Latest log entry
tail -10 logs/ralph.log

# Circuit breaker status
cat .circuit_breaker_state 2>/dev/null | jq
```

### Analyze Latest Output
```bash
# Get most recent output file
LATEST=$(ls -t logs/claude_output_*.log | head -1)

# Check for skill invocation
grep -i "Skill(skill" "$LATEST"

# Check for violations
grep -i "_to_array\|protected" "$LATEST"

# Parse JSON output
cat "$LATEST" | jq -r '.result' | head -50
```

### Validation Health Check
```bash
# Production code
uv run poe validate

# Test code (NOT in poe validate!)
uv run pylint tests/

# Check specific file
uv run pylint tests/test_rotation_matrix_consistency.py | grep W0212

# Count violations
uv run pylint tests/ 2>&1 | grep -c W0212
```

### Git Forensics
```bash
# Recent commits
git log --oneline -10

# What changed in last commit
git show --stat HEAD

# Violations in last commit
bash .claude/skills/monitor-ralph/scripts/detect-violations.sh HEAD

# When was file last modified
git log --all -- PROMPT.md | head -10
```

## Emergency Procedures

### Ralph is Doing Something Wrong

1. **Stop immediately:**
   ```bash
   pkill -f ralph_loop.sh
   ```

2. **Check what it did:**
   ```bash
   git log --oneline -5
   git diff HEAD~2
   ```

3. **Revert if bad:**
   ```bash
   git reset --hard <last-good-commit>
   ```

4. **Root cause analysis:**
   - Why did Ralph do this?
   - What should have prevented it?
   - How to fix the prevention?

### Skills Not Working

1. **Verify skill exists:**
   ```bash
   ls -la .claude/skills/*/SKILL.md
   ```

2. **Test skill manually:**
   ```bash
   claude -p "Use Skill(skill='validate-design') to analyze X"
   ```

3. **Check syntax in PROMPT.md:**
   ```bash
   grep "Skill(skill=" PROMPT.md
   # Should use Skill(skill="name"), not /name
   ```

4. **Verify committed:**
   ```bash
   git log --all -- .claude/skills/
   ```

### Validation Confusion

1. **Understand what's checked:**
   ```bash
   # What does poe validate run?
   grep -A 5 "\[tool.poe.tasks.validate\]" pyproject.toml
   ```

2. **Run each validator separately:**
   ```bash
   uv run ruff check poc_homography
   uv run pyright poc_homography
   uv run pylint poc_homography  # Production only!
   uv run pylint tests/           # Tests (separate!)
   uv run vulture
   ```

3. **Update config if needed:**
   - Edit `pyproject.toml` to include tests in pylint task
   - Or document that two commands must be run

## Success Indicators

You know monitoring is working when:

- ✅ You catch violations BEFORE commit
- ✅ You can explain WHY something went wrong
- ✅ You test fixes BEFORE recommending restart
- ✅ Ralph consistently invokes validate-design skill
- ✅ No W0212 violations make it into commits
- ✅ Root cause analysis reveals systemic issues
- ✅ Fixes address root causes, not just symptoms
