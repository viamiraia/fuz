# Justfile for fuz project

# Default recipe (list all recipes)
default:
    @just --list

# Development tasks
test:
    uv run pytest -v

mypy:
    uv run mypy pkg

# Demo tasks
demo:
    uv run marimo run pkg/fuz-demo/fuz/demo/optimal_bayesian_ranking_ch1.py

emo:
    uv run marimo edit --no-token

html: html-ch1 html-ch2

html-ch1:
    uv run marimo export html pkg/fuz-demo/fuz/demo/optimal_bayesian_ranking_ch1.py -o docs/_static/ch1.html

html-ch2:
    uv run marimo export html pkg/fuz-demo/fuz/demo/optimal_bayesian_ranking_ch2.py -o docs/_static/ch2.html

wasm: wasm-ch1 wasm-ch2

wasm-ch1:
    uv run marimo export html-wasm pkg/fuz-demo/fuz/demo/optimal_bayesian_ranking_ch1.py -o docs/_static/wasm/ch1.html --mode run

wasm-ch2:
    uv run marimo export html-wasm pkg/fuz-demo/fuz/demo/optimal_bayesian_ranking_ch2.py -o docs/_static/wasm/ch2.html --mode run

wasm-test: wasm-ch1-test wasm-ch2-test

wasm-ch1-test:
    uv run marimo export html-wasm pkg/fuz-demo/fuz/demo/optimal_bayesian_ranking_ch1.py -o docs/_private/wasm/ch1.html --mode edit

wasm-ch2-test:
    uv run marimo export html-wasm pkg/fuz-demo/fuz/demo/optimal_bayesian_ranking_ch2.py -o docs/_private/wasm/ch2.html --mode edit

# Documentation tasks
doc: sphinx html wasm

sphinx:
    cd docs && make html

testdoc:
    cd docs/_build/html && python -m http.server 8000

lint:
    uv run ruff check pkg

format:
    uv run ruff format pkg
