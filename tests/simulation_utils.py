"""Run the public examples in isolated working directories and compare results."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SOLUTION_FILES = tuple(f'sol{part}_E1{axis}_0' for part in ('r', 'i') for axis in 'xyz')
PROBE_FILES = ('probe_E1y_port1', 'probe_E1z_port1')
EXPECTED_FILES = (*SOLUTION_FILES, *PROBE_FILES, 'solmesh_0', 'exported_data.npz')
# Full-solve field/probe differences reached 9.8e-7 relative in the hot case.
# Keep about one order of margin; exported absorption agreed within 1.9e-10.
TEXT_RTOL = 1e-5
PROBE_RTOL = 1e-5
DATA_RTOL = 1e-8
ATOL_SCALE = 1e-10


def run_model(case, backend, work):
    work.mkdir(parents=True)
    model = REPO / 'examples' / case / 'model.py'
    assert model.is_file(), f'Missing example: {model}'
    env = os.environ.copy()
    env['PETRAM_RFP_USE_NUMBA_DIELECTRIC'] = '1' if backend == 'numba' else '0'
    env['PYTHONUNBUFFERED'] = '1'
    # Honor checkout testing even when PYTHONPATH entries were relative to pytest's cwd.
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = os.pathsep.join(str(Path(p or '.').resolve())
                                          for p in env['PYTHONPATH'].split(os.pathsep))
    log_path = work / 'run.log'
    command = [sys.executable, os.path.relpath(model, work)]
    facade = 'coldplasma' if case == 'coldplasma_1d' else 'lkplasma'
    preflight = (f'from petram.phys.common import {facade} as api; '
                 f'assert api.BACKEND == {backend!r}; '
                 'print("Verified backend:", api.BACKEND, "module:", api.__file__)')
    with log_path.open('w') as log:
        try:
            check = subprocess.run([sys.executable, '-c', preflight], cwd=work, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=180)
            assert check.returncode == 0, f'Backend preflight failed; see {log_path}'
            run = subprocess.run(command, cwd=work, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=600)
        except subprocess.TimeoutExpired as error:
            raise AssertionError(f'{case}/{backend} timed out; see {log_path}') from error
    output = log_path.read_text(errors='replace')
    assert run.returncode == 0, f'{case}/{backend} exit {run.returncode}; see {log_path}\n{output[-4000:]}'
    assert 'Normal End' in output, f'{case}/{backend} did not finish normally; see {log_path}'
    for name in EXPECTED_FILES:
        assert (work / name).is_file(), f'Missing output {work / name}; see {log_path}'


def solution(path):
    header, separator, body = path.read_text().partition('\n\n')
    assert separator and header.startswith('FiniteElementSpace'), f'Invalid solution header: {path}'
    return header, np.atleast_1d(np.loadtxt(io.StringIO(body)))


def probe(path):
    lines = path.read_text().splitlines()
    assert lines[:3] == ['format : 1', '1', 'time'], f'Unexpected probe format: {path}'
    values = [[complex(value.strip()) for value in line.split(',')]
              for line in lines[3:] if line.strip()]
    assert values and all(len(row) == 2 for row in values), f'Empty/invalid probe: {path}'
    return np.array(values, dtype=np.complex128)


def compare_results(ext, numba):
    """Return numerical diagnostics; raise with filename/key on any mismatch."""
    report = {}
    for directory in (ext, numba):
        numerical = {p.name for p in directory.iterdir()
                     if p.name.startswith(('solr_', 'soli_', 'solmesh_', 'probe_'))
                     or p.suffix == '.npz'}
        assert numerical == set(EXPECTED_FILES), f'{directory}: unexpected/missing numerical outputs: {numerical ^ set(EXPECTED_FILES)}'

    def compare(label, actual, expected, rtol):
        assert actual.shape == expected.shape, f'{label}: shape mismatch'
        assert actual.dtype == expected.dtype, f'{label}: dtype mismatch'
        assert actual.size, f'{label}: empty output'
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), f'{label}: nonfinite values'
        scale = float(np.max(np.abs(expected)))
        nonzero = np.abs(expected) > 0
        relative = float(np.max(np.abs(actual[nonzero] - expected[nonzero]) /
                                   np.abs(expected[nonzero]))) if np.any(nonzero) else 0.
        atol = ATOL_SCALE * scale
        delta = np.abs(actual - expected)
        worst = int(np.argmax(delta))
        report[label] = dict(max_abs=float(delta.flat[worst]), max_relative=relative, reference_scale=scale,
                             max_scaled=float(delta.flat[worst]/scale) if scale else float(delta.flat[worst]),
                             worst_index=list(np.unravel_index(worst, delta.shape)),
                             rtol=rtol, atol=atol)
        # Write diagnostics as we go, so a failed comparison also leaves a report.
        (ext.parent / 'comparison.json').write_text(json.dumps(report, indent=2, default=int))
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                                   equal_nan=False, err_msg=label)

    for name in SOLUTION_FILES:
        ah, a = solution(ext / name)
        bh, b = solution(numba / name)
        assert ah == bh, f'{name}: finite-element headers differ'
        compare(name, a, b, TEXT_RTOL)
    for name in PROBE_FILES:
        a, b = probe(ext / name), probe(numba / name)
        np.testing.assert_array_equal(a[:, 0], b[:, 0], err_msg=f'{name}: sample times')
        compare(name, a[:, 1], b[:, 1], PROBE_RTOL)
    # Both backends generate the same mesh, with no numerical solver involved.
    assert (ext / 'solmesh_0').read_text() == (numba / 'solmesh_0').read_text(), 'Mesh differs'
    with np.load(ext / 'exported_data.npz', allow_pickle=False) as a, \
            np.load(numba / 'exported_data.npz', allow_pickle=False) as b:
        required = {'PointCloud1_ptx', 'PointCloud1_data', 'PointCloud1_attr'}
        assert set(a.files) == set(b.files) == required, 'NPZ keys differ or expected data missing'
        for key in sorted(required):
            if key.endswith('_data'):
                compare('exported_data.npz:' + key, a[key], b[key], DATA_RTOL)
            else:
                assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype, key
                assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all(), key
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    return report


def run_and_compare(case, work):
    for backend in ('ext', 'numba'):
        run_model(case, backend, work / backend)
    report = compare_results(work / 'ext', work / 'numba')
    print(f'{case}: results and logs retained in {work}')
    print(json.dumps(report, indent=2, default=int))
