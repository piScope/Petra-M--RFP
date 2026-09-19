# This directory contains test routines

Run the commands below from the repository root.

## Testing coldplasma model
Run the following to execute cold plasma 1D simulation located in the
examples folder. The test runs the simulation twice using Numba
and C extension backends and compares their outputs.
```bash
python -m pytest tests/test_coldplasma.py -v -s
```

## Testing lkplasma model
Run the following to execute lkplasma 1D simulation located in the
examples folder. The test runs the simulation twice using Numba
and C extension backends and compares their outputs.
```bash
python -m pytest tests/test_lkplasma.py -s -v
```


# General guidance for testing package
## Preparation

Install the project in editable mode with its regular runtime dependencies:

```bash
python -m pip install -e '.[test]'
```

This creates an editable package with the optional `test` dependency, which is
currently `pytest` and `SciPy`. If these test dependencies are already installed in your virtual
environment, use:

```bash
python -m pip install -e .
```

## Running tests


```bash
PYTHONPATH=python python -m pytest tests/test_coldplasma.py tests/test_lkplasma.py -v -s
```

The `PYTHONPATH` setting makes the checkout's source package explicit and
prioritized. With an editable install this is normally equivalent to omitting
it; use a non-editable installation in a clean environment to test an
installed wheel.
