# AGENT Instructions

## Project structure

- Application code lives in `src/`.
- Tests live in `tests/` and mirror the package layout.
- Treat `uv.lock` as generated output. Update it with uv, not by hand.

## Python workflow

- Use `uv add` and `uv remove` for dependencies. Do not use pip.
- Run Python commands and tools with `uv run`.
- Read `pyproject.toml` before changing dependencies or tool settings.

## Quality checks

- Run tests with `uv run --group test pytest`.
- Run linting with `uv run ruff check`.
- Check formatting with `uv run ruff format`.
- Run type checking with `uv run ty check`.

## Change boundaries

- Ask before adding a production dependency.
- Do not edit generated files or secrets.
- Keep changes limited to the requested task.
