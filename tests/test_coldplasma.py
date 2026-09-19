"""Full cold-plasma simulation regression."""
import pytest
from simulation_utils import run_and_compare


@pytest.mark.integration
def test_coldplasma(tmp_path):
    run_and_compare('coldplasma_1d', tmp_path)
