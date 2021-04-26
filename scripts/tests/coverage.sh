#!/usr/bin/env bash

set -e
set -x

coverage run -m pytest --disable-pytest-warnings tests/units
# Uncomment when bdd tests added coverage run --append -m behave --tags="@common" tests/bdd/features

coverage report -m
coverage html -d coverage_html
coverage xml -o coverage-reports/coverage-ci.xml
