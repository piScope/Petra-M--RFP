"""Full local-kinetic-plasma simulation regression."""
import pytest
from simulation_utils import run_and_compare


@pytest.mark.integration
def test_lkplasma(tmp_path):
    run_and_compare('lkplasma_1d', tmp_path)
