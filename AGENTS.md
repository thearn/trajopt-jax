# trajopt-jax

Develop on a branch; `git land` merges to main through the landing gate (lib/git-hooks). Tests:
`~/.venvs/trajopt-jax/bin/python -m pytest -q`.

## Branches and landing

`main` is gated (`[tool.quality]` in `pyproject.toml`, lib/git-hooks): develop on a branch (`git switch -c <topic>`, or
`git worktree add ../../.wt/trajopt-jax-<topic> -b <topic>`), and land with `git land` when the task is done and the suite is
green. `git land --check` shows what landing requires: every per-unit rule absolute (functions at most 60 statements
and cognitive complexity 40, modules at most 500, no duplicated block, module-level pass-through or bare `# noqa`),
`pytest -m gate`, `code-spec --check` and the prose check. Never bypass the gate without the user.
