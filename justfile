# Justfile for fuz project

# Default recipe (list all recipes)
default:
    @just --list

# Development tasks
test:
    uv run pytest -v

mypy:
    uv run mypy packages/

# Demo tasks
demo:
    uv run marimo run packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py

emo:
    uv run marimo edit --headless --no-token packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py

html: html-ch1 html-ch2

html-ch1:
    uv run marimo export html packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py -o docs/_static/ch1.html

html-ch2:
    uv run marimo export html packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch2.py -o docs/_static/ch2.html

wasm: wasm-ch1 wasm-ch2

wasm-ch1:
    uv run marimo export html-wasm packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py -o docs/_static/wasm/ch1.html --mode run

wasm-ch2:
    uv run marimo export html-wasm packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch2.py -o docs/_static/wasm/ch2.html --mode run

wasm-test: wasm-ch1-test wasm-ch2-test

wasm-ch1-test:
    uv run marimo export html-wasm packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py -o docs/_private/wasm/ch1.html --mode edit

wasm-ch2-test:
    uv run marimo export html-wasm packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch2.py -o docs/_private/wasm/ch2.html --mode edit

# Documentation tasks
doc: sphinx html wasm

sphinx:
    cd docs && make html

testdoc:
    cd docs/_build/html && python -m http.server 8000
