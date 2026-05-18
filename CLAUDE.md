# Working agreement (Claude Code ↔ me)

This file is about **how we work together**, not what we're building. For project content, see PROJECT.md.

## Context

I'm a solo developer building an MVP. This is my first time using Claude Code on a real project — go slow with me.

---

## Documentation system

This repo uses three living documents. Knowing the difference is important — putting content in the wrong file degrades all three.

### PROJECT.md — "what we're building"
A neutral, factual snapshot of the project: vision, architecture, milestones, data contracts, current status, known pitfalls. Reads like onboarding documentation for any new developer (human or AI).

**Update rules:**
- After completing a milestone: tick the checkbox under "Development milestones" and update "Current status"
- When discovering a new pitfall or design constraint: add to "Implementation pitfalls" or "Known issues"
- When making a meaningful architecture decision: reflect it in the relevant section
- **Don't add** AI-collaboration instructions ("don't do X", "please remember Y") — those belong in CLAUDE.md
- **Don't add** implementation-detail specifics (loss weights, specific model versions, algorithm pseudocode) — those belong in code/comments
- **Don't change** scope (milestones, architecture) without asking me first

### CLAUDE.md — "how we work together" (this file)
Collaboration agreement: workflow, style preferences, boundaries. Auto-loaded by Claude Code at session start.

**Update rules:**
- When we discover a new working pattern that helps: add it here
- When a recurring annoyance comes up: codify the rule here
- Ask before making changes — this is the contract between us

### NOTES.md — "running log"
Dated, append-only log of what happened, decisions made, parameters tried, open questions. Like a lab notebook.

**Update rules:**
- Append at the end of each working session
- Use date-stamped entries: `## YYYY-MM-DD`
- Record: what was accomplished, key decisions and *why*, parameters tried (for Stage 2 tuning especially), open questions
- Never edit past entries — append corrections as new entries

---

## Session lifecycle

### At the start of a session
- Confirm CLAUDE.md is loaded (it auto-loads)
- Read PROJECT.md, especially "Current status" and the section for the current milestone
- Read the most recent entries in NOTES.md to know what just happened
- Summarize back to me: current milestone, last session's outcome, anything pending

### During work
- Outline the plan in plain English before writing code; wait for my approval
- Write small, focused changes I can review — not large code dumps
- When debugging, ask for the actual error before guessing
- Suggest the simplest solution that works for MVP
- Explain *why* before *how* for non-trivial choices

### Before ending a session
- Update PROJECT.md if a milestone was completed (checkbox + Current status)
- Append a new dated entry to NOTES.md summarizing what happened, decisions made, open questions
- Suggest a one-line git commit message

---

## Boundaries

- Don't create new files unless necessary; prefer editing existing ones
- Don't propose splitting single-file stages into submodules — the flat layout is intentional
- Don't refactor unrelated code while doing a task
- Don't add abstraction layers, base classes, or interfaces "for future flexibility"
- Don't write unit tests unless I ask
- Don't silently install packages — ask first
- Don't change PROJECT.md scope (milestones, architecture) without asking
- Implement only the current milestone's scope; if a future feature seems "easy to add now", propose it, don't sneak it in

---

## Reuse over reimplementation

Always prefer reusing existing open-source models/libraries over writing algorithms from scratch.

Priority order for any algorithmic capability:
1. Official pip package with clean API (best)
2. Official GitHub repo's demo/inference script, wrapped thinly
3. Modifying source from a forked repo (avoid unless necessary)
4. Implementing from scratch (only if no alternative exists)

When suggesting a new dependency, tell me:
- What it does
- License (especially: is it commercial-friendly?)
- Whether it's pip-installable or repo-clone required
- A simpler alternative if any

---

## Code style preferences

- Python 3.10 syntax: `list[str]`, `dict[str, int]`, `X | None` — not `List/Dict/Optional`
- Type hints on function signatures; lighter inside function bodies
- `pathlib.Path` over `os.path`
- `loguru` over `print` for anything worth keeping
- numpy arrays for math; `torch.Tensor` only inside model code
- Constants in `config.py`, not magic numbers scattered in code
- Docstrings on public functions; one-liners are fine for simple stuff

---

## When unsure

Ask me. Especially about:
- Architecture / module boundary decisions
- License-sensitive dependencies
- Algorithm choices that affect output quality
- Anything touching the data contract between stages (Stage1Result / Stage2Result / Stage3Result)

I'd rather answer a question than undo a wrong assumption.

---

## Useful commands

- Run pipeline: `python scripts/run.py <video> --height H --weight W`
- View latest run: `ls data/runs/`