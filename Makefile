lint:
	python -m black rl_sampling tests; python3 -m isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --skip __init__.py  --line-width 88 -rc rl_sampling tests

flake8:
	python -m flake8 --exclude=tests/bdd/features/steps,rl_sampling/externals rl_sampling tests

testing:
	bash ./scripts/tests/coverage.sh
