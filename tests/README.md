# How to test the package

## Preparation

Install the project in editable mode with its regular runtime dependencies:

```bash
python -m pip install -e '.[test]'
```

This creates an editable package with the optional `test` dependency, which is
currently `pytest`. If you already have `pytest` installed in your virtual
environment, use:

```bash
python -m pip install -e .
```

## Running tests


```bash
PYTHONPATH=python python -m pytest
```

The `PYTHONPATH` setting makes the checkout's source package explicit and
prioritized. With an editable install this is normally equivalent to omitting
it; use a non-editable installation in a clean environment to test an
installed wheel.

# Tests Cases

## 1 AOT

This test verifies that the Numba ahead-of-time (AOT) RF dispersion
extensions produce the same dielectric tensors as the existing runtime JIT
implementations. It builds the AOT extensions in a temporary directory, so it
does not verify extensions installed by `pip install .`.
