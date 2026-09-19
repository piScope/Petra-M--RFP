# Testing Petra-M RFP

## Public simulation tests

Install the package and test dependencies with `python -m pip install '.[test]'`.
The environment must also have a working Petra-M/MFEM installation and the
SuperLU solver used by these examples. Even a serial run may initialize MPI,
so the environment must permit its local socket operations.

Run the two complete example tests:

```bash
python -m pytest tests/test_coldplasma.py tests/test_lkplasma.py -v -s
```

Each test runs the original `examples/<case>/model.py` using a relative path
from two empty temporary working directories. No model files or existing
`sandbox/` outputs are copied. The subprocesses select EXT and Numba separately
with `PETRAM_RFP_USE_NUMBA_DIELECTRIC`. Historical model paths are left unchanged.
A preflight records the selected backend and facade location. Runs must exit
successfully, print `Normal End`, and produce all expected numerical files.

For installed-package testing, remove the checkout's `python/` directory from
`PYTHONPATH`. To test a checkout instead, build its extensions first and run
with `PYTHONPATH=python`; the runner resolves relative entries before changing
working directories.

Logs and results remain in pytest's temporary directories; `-s` prints their
locations. Each comparison writes `comparison.json` with maximum absolute,
pointwise relative, and scale-normalized errors, the worst absolute-error index,
and the applied tolerances. Failures identify the relevant file or archive key.
Pytest retains temporary directories according to its normal retention policy.

## Compared outputs and tolerances

- Six real/imaginary field vectors: exact finite-element headers and matching
  shapes; numerical comparison with `rtol=1e-5`.
- Two complex port probes: exact sample times; values with `rtol=1e-5`.
- `exported_data.npz`: exact keys, shapes, dtypes, coordinates, and integer
  attributes; complex data with `rtol=1e-8`.
- Mesh: exact text comparison, since the backend does not change mesh generation.

For numerical data, `atol = 1e-10 * max(abs(reference_data))`, using the Numba
result as reference, separately for each vector/probe/export. A zero reference
vector must match exactly. NaN/Inf and empty data fail. Comparisons use the
usual elementwise `abs(EXT - Numba) <= atol + rtol * abs(Numba)` rule.
Serialized metadata (`model_proc.pmfm`, `sol_extended.data`) and log timestamps
are not compared. Compressed NPZ contents are compared, not archive bytes.

Initial installed-package measurements on these examples:

| Quantity | Cold | Hot |
|---|---:|---:|
| Maximum pointwise field relative difference | 0 | 9.80e-7 |
| Maximum probe relative difference | 0 | 8.94e-7 |
| Maximum exported-data relative difference | 0 | 1.85e-10 |

The field/probe tolerance leaves roughly one order of margin above observed
full-solve differences; text fields have only about eight significant digits.
The tighter exported-data tolerance reflects its much closer agreement.
These are proposed regression tolerances for these examples, not a guarantee
of cross-platform solver accuracy. Investigate a failure before relaxing them.
Backend agreement alone does not establish independent physical correctness.

## Developer tests

The full suite includes the simulation tests. For faster kernel/interface checks:

```bash
PYTHONPATH=python python -m pytest -m 'not integration'
```

## Backend and native tests

Run `python setup.py build_ext --inplace` before testing directly from the source
checkout. Tests compare the C kernels with Numba, validate malformed inputs,
and check backend selection in fresh processes. SciPy supplies independent
Bessel checks. No Numba AOT build is used.

For an installed-package check, run from outside the checkout with its `python/`
directory removed from `PYTHONPATH`.
