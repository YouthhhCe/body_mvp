# Working style for this project

## About me & this project
I'm a solo developer building an MVP for a 3D body reconstruction pipeline. See PROJECT.md for full context. I'm using Claude Code for the first time on this project — go slow with me.

## Core principles
1. **Working end-to-end > clean architecture.** I'd rather have ugly code that runs than beautiful code that doesn't.
2. **Small steps, visible results.** Each change should produce something I can see or run.
3. **Ask before assuming.** I'd rather answer a question than undo a wrong assumption.

## Do
- Before writing code: outline the plan in plain English and wait for my approval
- Suggest the simplest solution that works for an MVP
- Explain *why* before writing code, especially for non-trivial decisions
- Write small, focused changes I can review in one sitting
- When adding a dependency, tell me what it does and why we need it
- When debugging, ask for the actual error message before guessing
- After completing a milestone, update PROJECT.md status and NOTES.md log

## Don't
- Don't create new files unless necessary — prefer editing existing ones
- Don't split single files into modules "for organization" — see PROJECT.md, each stage stays in one file
- Don't refactor unrelated code while doing a task
- Don't add abstraction layers, base classes, or interfaces "for future flexibility"
- Don't write unit tests unless I explicitly ask
- Don't generate large code blocks all at once — break into reviewable chunks
- Don't silently install packages — ask first
- Don't change PROJECT.md milestones or scope without asking

## Code style
- Type hints on function signatures, but skip overly verbose typing inside functions
- Docstrings on public functions, single-line OK for simple ones
- Prefer `pathlib.Path` over `os.path`
- Prefer `loguru` over `print` for anything that should persist
- Use `numpy` arrays for math, `torch.Tensor` only inside model code
- Constants in `config.py`, not magic numbers in code

## When unsure
Ask me. Especially about:
- Architecture decisions
- License-sensitive dependencies  
- Algorithm choices that affect quality
- Anything that touches the data contract between stages

## Useful commands
- Run pipeline: `python scripts/run.py <video> --height H --weight W`
- Download models: `bash scripts/download_models.sh`
- View latest run: `ls data/runs/`