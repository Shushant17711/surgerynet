import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "reproduce.sh"


def test_check_mode_passes_in_the_real_project():
    result = subprocess.run(["bash", str(SCRIPT), "--check"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "check OK" in result.stdout


def test_check_mode_reports_missing_scripts_without_running_anything(tmp_path):
    # Copy just the script into an empty fake project -- every required
    # file and both venvs are "missing" from its point of view.
    fake_root = tmp_path
    (fake_root / "scripts").mkdir()
    script_copy = fake_root / "scripts" / "reproduce.sh"
    script_copy.write_text(SCRIPT.read_text())
    script_copy.chmod(0o755)

    result = subprocess.run(["bash", str(script_copy), "--check"], cwd=fake_root, capture_output=True, text=True)
    assert result.returncode == 1
    assert "MISSING" in result.stdout
    assert "check OK" not in result.stdout
