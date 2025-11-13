# Fuz Migration Plan: Pixi → UV Monorepo

**Date**: 2025-11-13
**Branch**: uv-migration
**Status**: Planning phase

---

## Executive Summary

Migrating from Pixi environment manager to UV with a major refactor from single package to monorepo with 5 sub-packages. This is a complete migration (no pixi remnants).

---

## Current Codebase Structure

### File Tree

```
fuz/
├── __init__.py                     # Only contains __version__ = '0.1.3'
├── _version.py                     # Version file
├── types.py                        # Type aliases and protocols
├── lint.py                         # Log-space numerical operations
├── rank.py                         # Ranking functions
├── pool.py                         # Probability fusion/pooling
├── logic.py                        # Fuzzy logic operations
├── convert.py                      # Conversion utilities
├── utils.py                        # Miscellaneous utilities
├── plot.py                         # Plotting functions
├── marimo.py                       # Marimo helper widgets
├── demo/
│   ├── __init__.py                 # Empty
│   ├── optimal_bayesian_ranking_ch1.py   # Marimo notebook
│   ├── optimal_bayesian_ranking_ch2.py   # Marimo notebook
│   └── optimal_bayesian_ranking_ch3.py   # Marimo notebook
├── dists/
│   ├── __init__.py                 # Exports Beta, Dirichlet, Scored, etc.
│   ├── beta.py                     # Beta distribution class and helpers
│   └── dirichlet.py                # Dirichlet and Scored classes
└── gpu/
    ├── __init__.py                 # Empty
    └── types.py                    # JAX type aliases (placeholder)

tests/
├── __init__.py                     # Empty
└── test_lint.py                    # Hypothesis tests for lint.py
```

### Import Dependency Graph

**Dependency Layers** (bottom to top):
```
Layer 0: types.py (no fuz imports)
Layer 1: lint.py (types), rank.py (no fuz imports), utils.py (no fuz imports), convert.py (types)
Layer 2: pool.py (lint, types), logic.py (lint), dists/* (types)
Layer 3: plot.py (dists), marimo.py (rank)
Layer 4: demos (lint, marimo, dists indirectly)
```

**Module-by-Module Dependencies**:

1. **types.py** - Foundation
   - External: `attrs`, `jaxtyping`, `narwhals`, `numpy`, `scipy.stats`
   - Contains: Type aliases, protocols (FuzDist), constants (RATE_N)
   - **Pure type definitions** - NO implementations

2. **lint.py** - Log-space numerical operations
   - Imports: `fuz.types as ft`
   - External: `numpy`, `plum`
   - Functions: `fillna`, `lsub`, `complex_lsub`, `nanlse`, `complex_nanlse`, `lnorm`, `norm`, log integration functions

3. **pool.py** - Probability fusion
   - Imports: `fuz.lint as flog`, `fuz.types as ft`
   - External: `numpy`, `plum`, `scipy.integrate`
   - Functions: Geometric/multiplicative/weighted pooling for RVs and samples

4. **logic.py** - Fuzzy logic
   - Imports: `fuz.lint as flog`
   - External: `numpy`, `plum`, `scipy.integrate`, `scipy.interpolate`
   - Functions: Fuzzy logic operations (AND, OR, NOT, etc.) on distributions

5. **rank.py** - Ranking functions
   - Imports: **NONE from fuz**
   - External: `numpy`, `math`
   - Functions: `bayes_avg`, `ros`, `inf_weight`, `inf_ros`, `rating_to_moons`

6. **convert.py** - Conversions
   - Imports: `fuz.types as ft`
   - External: `numpy`, `math`
   - Functions: `dl_to_ld`, `ld_to_dl`, `rating_to_moons`, `prop_to_stars`, `stars_to_prop`, `prop_to_moons`

7. **utils.py** - Utilities
   - Imports: **NONE from fuz**
   - External: `plum` (in function)
   - Functions: `dl_to_ld`, `ld_to_dl`, `support_autoreload`

8. **plot.py** - Plotting
   - Imports: `fuz.dists as fd`
   - External: `matplotlib`, `mpltern`, `numpy`, `scipy.stats`
   - Functions: `terngrid`, `plot_multinomial`, `plot_multinomial_3d`, `plot_scored_pdf`

9. **marimo.py** - Marimo widgets
   - Imports: `from fuz.rank import rating_to_moons`
   - External: `marimo`
   - Functions: `make_star_widget`

10. **dists/beta.py** - Beta distribution
    - Imports: `fuz.types as ft`
    - External: `numpy`, `scipy`, `attrs`
    - Classes: `Beta` (frozen attrs class implementing FuzDist)
    - Functions: `ab_from_*`, `beta_from_*`, `get_posterior`

11. **dists/dirichlet.py** - Dirichlet distribution
    - Imports: `fuz.types as ft`
    - External: `numpy`, `scipy.stats`, `attrs`
    - Classes: `Dirichlet`, `Scored`

### Known Issues

**Duplicate Code** (to be handled later):
1. `rating_to_moons` appears in both `rank.py` and `convert.py`
2. `dl_to_ld`/`ld_to_dl` appear in both `utils.py` and `convert.py`

---

## Current Pixi Configuration

### Python Version
- Minimum: Python 3.10
- Pixi features: minpy (3.10.*), maxpy (>3.10, <3.14)
- Default: Python 3.14

### Core Dependencies
```
attrs
jaxtyping
narwhals
plum-dispatch
scipy
```

### Feature-Based Dependencies

**Plot Feature**:
- altair, matplotlib, mpltern

**Demo Feature**:
- fastparquet, linkify-it-py, marimo, mdformat-myst, myst-parser
- nbformat, pandas, polars

**Dev Feature**:
- hypothesis, ipykernel, ipympl, ipywidgets, mypy, nox
- pyarrow, pytest, pytest-cov, ruff, scipy-stubs, tqdm
- wat-inspector, duckdb, openai, python-lsp-ruff, python-lsp-server
- sqlglot, ty (from conda-forge/label/ty_alpha - TO BE REMOVED)
- vegafusion, vl-convert-python, websockets

**Doc Feature**:
- numpydoc, pydata-sphinx-theme, soupsieve, sphinx, sphinx-autoapi
- sphinx-autodoc-typehints, sphinx-copybutton, sphinx-design
- sphinx-togglebutton, sphinxcontrib-youtube, sphinx-docsearch (PyPI)

**JAX Feature**:
- jax, jaxlib

**GPU Feature** (CUDA 12 - optional for most contributors):
- jaxlib (>=0.4.35, build=*cuda12*, from conda-forge)

**Temp Feature**:
- expression, seaborn

### Build System
- Backend: hatchling
- Build requires: hatchling, hatch-vcs, hatch-fancy-pypi-readme
- Pixi uses: pixi-build with pixi-build-python backend

### Pixi Environments
1. **default**: ['demo', 'plot']
2. **dev**: ['dev', 'demo', 'plot', 'minpy', 'jax', 'temp', 'doc']
3. **devgpu**: ['dev', 'demo', 'plot', 'minpy', 'jax', 'gpu', 'temp', 'doc']
4. **doc**: ['dev', 'demo', 'plot', 'maxpy', 'jax', 'doc']

### Pixi Tasks

**Demo tasks**:
- `demo`: Run marimo main demo
- `emo`: Run marimo in edit mode (headless, no token)
- `html`: Export all chapters to HTML (depends on html-ch1, html-ch2)
- `html-ch1`: Export chapter 1 to static HTML
- `html-ch2`: Export chapter 2 to static HTML
- `wasm`: Export all chapters to WASM (depends on wasm-ch1, wasm-ch2)
- `wasm-ch1`: Export chapter 1 to WASM (run mode)
- `wasm-ch2`: Export chapter 2 to WASM (run mode)
- `wasm-ch1-test`: Export chapter 1 to WASM (edit mode, private folder)
- `wasm-ch2-test`: Export chapter 2 to WASM (edit mode, private folder)
- `wasm-test`: Test WASM exports (depends on wasm-ch1-test, wasm-ch2-test)

**Dev tasks**:
- `mypy`: Run type checking
- `test`: Run pytest with verbose output

**Doc tasks**:
- `doc`: Build documentation (depends on sphinx, html, wasm)
- `sphinx`: Build Sphinx docs (runs `make html` in docs directory)
- `testdoc`: Serve documentation locally on port 8000

---

## Target Monorepo Structure

### Directory Layout

```
fuz/ (monorepo root + metapackage)
├── packages/
│   ├── fuz-core/           # Foundation: types, dists, utils
│   │   ├── pyproject.toml
│   │   ├── fuz/
│   │   │   └── core/
│   │   │       ├── __init__.py
│   │   │       ├── types.py
│   │   │       ├── convert.py
│   │   │       ├── utils.py
│   │   │       ├── marimo.py
│   │   │       └── dists/
│   │   │           ├── __init__.py
│   │   │           ├── beta.py
│   │   │           └── dirichlet.py
│   │   └── tests/
│   │
│   ├── fuz-lint/           # Log-space numerics
│   │   ├── pyproject.toml
│   │   ├── fuz/
│   │   │   └── lint/
│   │   │       ├── __init__.py
│   │   │       └── lint.py
│   │   └── tests/
│   │       └── test_lint.py
│   │
│   ├── fuz-pool/           # Probability fusion + fuzzy logic
│   │   ├── pyproject.toml
│   │   ├── fuz/
│   │   │   └── pool/
│   │   │       ├── __init__.py
│   │   │       ├── pool.py
│   │   │       └── logic.py
│   │   └── tests/
│   │
│   ├── fuz-rank/           # Ranking + plotting
│   │   ├── pyproject.toml
│   │   ├── fuz/
│   │   │   └── rank/
│   │   │       ├── __init__.py
│   │   │       ├── rank.py
│   │   │       └── plot.py
│   │   └── tests/
│   │
│   └── fuz-demos/          # Interactive notebooks
│       ├── pyproject.toml
│       ├── fuz/
│       │   └── demos/
│       │       ├── __init__.py
│       │       ├── optimal_bayesian_ranking_ch1.py
│       │       ├── optimal_bayesian_ranking_ch2.py
│       │       └── optimal_bayesian_ranking_ch3.py
│       └── tests/
│
├── tests/                  # Integration tests
│   └── test_integration.py
│
├── docs/                   # Documentation (existing)
├── justfile                # Task runner (replaces pixi tasks)
├── pyproject.toml          # Workspace root + metapackage
├── uv.lock                 # Lock file (committed)
├── .python-version         # Python version (3.10)
├── .gitignore              # Updated
└── README.md               # Updated instructions
```

### Package Dependencies

**Dependency Chain**:
```
fuz-core (foundation - no internal deps)
    ↓
fuz-lint (depends on: fuz-core)
    ↓
fuz-pool (depends on: fuz-core, fuz-lint)

fuz-rank (depends on: fuz-core)

fuz-demos (depends on: fuz-core, fuz-lint, fuz-rank)
```

**Metapackage `fuz`**:
- Installing `fuz` installs all 5 packages as a bundle
- Users can also install individual packages (e.g., `pip install fuz-lint`)

### Test Organization

**Both per-package and integration**:
1. Each package has its own `tests/` directory for unit tests
2. Root `tests/` directory for integration tests across packages

---

## Migration Plan

### Phase 1: Create UV Workspace Structure

**1.1 Create directory structure**
```bash
mkdir -p packages/fuz-core/fuz/core/dists
mkdir -p packages/fuz-lint/fuz/lint
mkdir -p packages/fuz-pool/fuz/pool
mkdir -p packages/fuz-rank/fuz/rank
mkdir -p packages/fuz-demos/fuz/demos
mkdir -p packages/fuz-core/tests
mkdir -p packages/fuz-lint/tests
mkdir -p packages/fuz-pool/tests
mkdir -p packages/fuz-rank/tests
mkdir -p packages/fuz-demos/tests
```

**1.2 Create root workspace `pyproject.toml`**
```toml
[project]
name = "fuz"
version = "0.1.3"
description = "Fuzzy probability utilities - metapackage"
requires-python = ">=3.10,<3.14"
dependencies = [
    "fuz-core",
    "fuz-lint",
    "fuz-pool",
    "fuz-rank",
    "fuz-demos",
]

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
fuz-core = { workspace = true }
fuz-lint = { workspace = true }
fuz-pool = { workspace = true }
fuz-rank = { workspace = true }
fuz-demos = { workspace = true }

[build-system]
requires = ["hatchling", "hatch-vcs", "hatch-fancy-pypi-readme"]
build-backend = "hatchling.build"
```

**1.3 Create `pyproject.toml` for each package**

See detailed specs in Phase 3 below.

**1.4 Create `.python-version`**
```
3.10
```

### Phase 2: Move and Reorganize Files

**2.1 File movements**

Map:
```
fuz/types.py          → packages/fuz-core/fuz/core/types.py
fuz/convert.py        → packages/fuz-core/fuz/core/convert.py
fuz/utils.py          → packages/fuz-core/fuz/core/utils.py
fuz/marimo.py         → packages/fuz-core/fuz/core/marimo.py
fuz/dists/*           → packages/fuz-core/fuz/core/dists/*

fuz/lint.py           → packages/fuz-lint/fuz/lint/lint.py

fuz/pool.py           → packages/fuz-pool/fuz/pool/pool.py
fuz/logic.py          → packages/fuz-pool/fuz/pool/logic.py

fuz/rank.py           → packages/fuz-rank/fuz/rank/rank.py
fuz/plot.py           → packages/fuz-rank/fuz/rank/plot.py

fuz/demo/*            → packages/fuz-demos/fuz/demos/*

tests/test_lint.py    → packages/fuz-lint/tests/test_lint.py
```

**2.2 Update imports across all files**

Examples:
```python
# Old imports
from fuz.types import NPVec
import fuz.lint as flog
from fuz.dists import Beta

# New imports
from fuz.core.types import NPVec
import fuz.lint as flog  # (fuz.lint.__init__ re-exports from fuz.lint.lint)
from fuz.core.dists import Beta
```

**2.3 Create proper `__init__.py` files**

Each package's `__init__.py` should expose public API.

Example `packages/fuz-lint/fuz/lint/__init__.py`:
```python
"""Log-space numerical helpers."""

from fuz.lint.lint import (
    fillna,
    lsub,
    complex_lsub,
    nanlse,
    complex_nanlse,
    limag_sign,
    lnorm,
    norm,
    log_trap,
    ltrap,
    lsimp13,
    lsimp38,
    lsimp_irreg,
)

__all__ = [
    "fillna",
    "lsub",
    "complex_lsub",
    "nanlse",
    "complex_nanlse",
    "limag_sign",
    "lnorm",
    "norm",
    "log_trap",
    "ltrap",
    "lsimp13",
    "lsimp38",
    "lsimp_irreg",
]
```

### Phase 3: Convert Dependencies to pyproject.toml

**3.1 fuz-core dependencies**

```toml
[project]
name = "fuz-core"
version = "0.1.3"
description = "Core types and distributions for fuz"
requires-python = ">=3.10,<3.14"
dependencies = [
    "attrs",
    "jaxtyping",
    "narwhals",
    "numpy",
    "scipy",
    "plum-dispatch",
]

[project.optional-dependencies]
marimo = ["marimo"]
```

**3.2 fuz-lint dependencies**

```toml
[project]
name = "fuz-lint"
version = "0.1.3"
description = "Log-space numerical operations for fuz"
requires-python = ">=3.10,<3.14"
dependencies = [
    "fuz-core",
    "numpy",
    "plum-dispatch",
]

[tool.uv.sources]
fuz-core = { workspace = true }
```

**3.3 fuz-pool dependencies**

```toml
[project]
name = "fuz-pool"
version = "0.1.3"
description = "Probability fusion and fuzzy logic for fuz"
requires-python = ">=3.10,<3.14"
dependencies = [
    "fuz-core",
    "fuz-lint",
    "numpy",
    "plum-dispatch",
    "scipy",
]

[tool.uv.sources]
fuz-core = { workspace = true }
fuz-lint = { workspace = true }
```

**3.4 fuz-rank dependencies**

```toml
[project]
name = "fuz-rank"
version = "0.1.3"
description = "Ranking and plotting utilities for fuz"
requires-python = ">=3.10,<3.14"
dependencies = [
    "fuz-core",
    "numpy",
]

[project.optional-dependencies]
plot = [
    "altair",
    "matplotlib",
    "mpltern",
]
```

**3.5 fuz-demos dependencies**

```toml
[project]
name = "fuz-demos"
version = "0.1.3"
description = "Interactive demo notebooks for fuz"
requires-python = ">=3.10,<3.14"
dependencies = [
    "fuz-core",
    "fuz-lint",
    "fuz-rank[plot]",
    "marimo",
    "pandas",
    "polars",
    "altair",
    "matplotlib",
    "mpltern",
    "numpy",
]

[tool.uv.sources]
fuz-core = { workspace = true }
fuz-lint = { workspace = true }
fuz-rank = { workspace = true }
```

**3.6 Root metapackage optional dependencies**

```toml
[project.optional-dependencies]
dev = [
    "hypothesis",
    "ipykernel",
    "ipympl",
    "ipywidgets",
    "mypy",
    "nox",
    "pyarrow",
    "pytest",
    "pytest-cov",
    "ruff",
    "scipy-stubs",
    "tqdm",
    "wat-inspector",
    "duckdb",
    "openai",
    "python-lsp-ruff",
    "python-lsp-server",
    "sqlglot",
    "vegafusion",
    "vl-convert-python",
    "websockets",
]

doc = [
    "numpydoc",
    "pydata-sphinx-theme",
    "soupsieve",
    "sphinx",
    "sphinx-autoapi",
    "sphinx-autodoc-typehints",
    "sphinx-copybutton",
    "sphinx-design",
    "sphinx-togglebutton",
    "sphinxcontrib-youtube",
    "sphinx-docsearch",
]

jax = [
    "jax",
    "jaxlib",
]

temp = [
    "expression",
    "seaborn",
]

all = [
    "fuz[dev,doc,jax,temp]",
]
```

**Note**: Removed `ty` package (from conda-forge/label/ty_alpha)

### Phase 4: Create Justfile

Create `justfile` at root with all pixi tasks ported:

```justfile
# Justfile for fuz project

# Default recipe (list all recipes)
default:
    @just --list

# Development tasks
test:
    pytest -v

mypy:
    mypy packages/

# Demo tasks
demo:
    marimo run packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py

emo:
    marimo edit --headless --no-token packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py

html: html-ch1 html-ch2

html-ch1:
    marimo export html packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py -o docs/_static/ch1.html

html-ch2:
    marimo export html packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch2.py -o docs/_static/ch2.html

wasm: wasm-ch1 wasm-ch2

wasm-ch1:
    marimo export html-wasm packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py -o docs/_static/wasm/ch1.html --mode run

wasm-ch2:
    marimo export html-wasm packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch2.py -o docs/_static/wasm/ch2.html --mode run

wasm-test: wasm-ch1-test wasm-ch2-test

wasm-ch1-test:
    marimo export html-wasm packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch1.py -o docs/_private/wasm/ch1.html --mode edit

wasm-ch2-test:
    marimo export html-wasm packages/fuz-demos/fuz/demos/optimal_bayesian_ranking_ch2.py -o docs/_private/wasm/ch2.html --mode edit

# Documentation tasks
doc: sphinx html wasm

sphinx:
    cd docs && make html

testdoc:
    cd docs/_build/html && python -m http.server 8000
```

### Phase 5: Update Configuration Files

**5.1 Update `.gitignore`**

Remove:
```
.pixi
```

Add:
```
.venv/
venv/
```

Keep:
```
uv.lock  # Committed for reproducibility
*.egg-info
```

**5.2 Update `ruff.toml` (if needed)**

Check if any paths need updating for new structure.

### Phase 6: Initialize UV and Test

**6.1 Install just**
```bash
# Windows (using cargo or scoop)
cargo install just
# or
scoop install just
```

**6.2 Initialize UV**
```bash
uv lock
uv sync --all-packages --all-extras
```

**6.3 Run tests**
```bash
just test
# or directly:
uv run pytest -v
```

**6.4 Test justfile tasks**
```bash
just mypy
just demo
just html
```

**6.5 Verify imports**

Test that new import paths work:
```bash
uv run python -c "from fuz.core.types import NPVec; print('✓ fuz-core')"
uv run python -c "import fuz.lint as flog; print('✓ fuz-lint')"
uv run python -c "from fuz.pool import pool; print('✓ fuz-pool')"
uv run python -c "from fuz.rank import bayes_avg; print('✓ fuz-rank')"
uv run python -c "from fuz.demos import optimal_bayesian_ranking_ch1; print('✓ fuz-demos')"
```

### Phase 7: Cleanup

**7.1 Remove pixi files**
```bash
rm pixi.toml pixi.lock
rm -rf .pixi
```

**7.2 Update README.md**

Replace pixi instructions with uv instructions:

```markdown
## Installation

### Using uv (recommended)

Install the entire fuz suite:
```bash
uv pip install fuz
```

Or install specific packages:
```bash
uv pip install fuz-core
uv pip install fuz-lint
uv pip install fuz-pool
uv pip install fuz-rank
uv pip install fuz-demos
```

### Development Setup

Clone and install with all extras:
```bash
git clone https://github.com/yourusername/fuz.git
cd fuz
uv sync --all-packages --all-extras
```

Run tests:
```bash
just test
```

Run demos:
```bash
just demo
```
```

**7.3 Final verification**

- [ ] All tests pass
- [ ] All justfile tasks work
- [ ] Documentation builds correctly
- [ ] No import errors
- [ ] No pixi references remain

---

## Key Design Decisions

1. **Complete migration**: No pixi remnants, UV-only
2. **Monorepo with workspaces**: 5 sub-packages under `packages/`
3. **Metapackage**: `fuz` installs all sub-packages as bundle
4. **Python version**: 3.10-3.13 compatibility (dropped 3.14 for now)
5. **Lock file**: `uv.lock` committed to git for reproducibility
6. **Task runner**: justfile replaces pixi tasks
7. **Test strategy**: Both per-package unit tests + root integration tests
8. **Package structure**: Namespace packages under `fuz.*`
9. **Dependency chain**: Linear: core → lint → pool, with rank and demos branching from core
10. **GPU support**: Deferred until actual GPU code exists (removed gpu/ directory for now)
11. **Removed**: `ty` package (conda-only alpha channel)

---

## Open Questions / Notes

1. **Duplicate code**: Not addressed in this migration
   - `rating_to_moons` in both `rank.py` and `convert.py`
   - `dl_to_ld`/`ld_to_dl` in both `utils.py` and `convert.py`
   - To be cleaned up in a future PR

2. **Version synchronization**: All packages start at 0.1.3
   - Should they be versioned independently going forward?
   - Or stay synchronized?

3. **Publishing strategy**:
   - Publish all 5 packages + metapackage to PyPI?
   - Or keep as monorepo for development only?

4. **Documentation**:
   - Docs currently at root level
   - Should they be restructured per-package or stay unified?

5. **CI/CD**:
   - Will need updates for multi-package testing
   - Need to ensure all package combinations work

---

## Next Steps

1. Review this plan
2. Execute Phase 1-7 systematically
3. Test thoroughly at each phase
4. Create PR for review
5. Update CI/CD pipelines
6. Consider publishing strategy for PyPI
