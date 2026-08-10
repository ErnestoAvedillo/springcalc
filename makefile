# Recursive (=) so it is re-read after `bump` rewrites pyproject.toml,
# instead of being frozen to the pre-bump version at parse time.
VERSION = $(shell uv run python -c "import pathlib, tomllib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")

.PHONY: bump build pypi test

bump:
	uv version --bump patch

build: bump
	rm -rf dist
	uv run python -m build

pypi: build
	uv run python -m twine upload --repository pypi dist/springcalc-$(VERSION)*

test: build
	uv run python -m twine upload --repository testpypi dist/springcalc-$(VERSION)*
