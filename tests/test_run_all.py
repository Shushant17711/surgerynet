import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TQEC_PYTHON = PROJECT_ROOT / ".venv-tqec" / "bin" / "python"

pytestmark = pytest.mark.skipif(not TQEC_PYTHON.exists(), reason=".venv-tqec not set up")


def test_run_all_executes_full_pipeline_and_writes_main_table(tmp_path):
    result = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-m", "experiments.run_all",
            "--distance", "3", "--k", "1", "--p", "0.02",
            "--shots", "150", "--epochs", "1",
            "--results-dir", str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    main_table = tmp_path / "main_table.md"
    assert main_table.exists()
    content = main_table.read_text()
    for experiment in ("E2", "E3", "E5", "E6", "E7", "E8", "E9"):
        assert experiment in content
    assert (tmp_path / "spacetime_heatmap.png").exists()
