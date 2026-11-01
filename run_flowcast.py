from pathlib import Path
import subprocess
import sys

from src.utils import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent
STAGES = (
    [sys.executable, "-m", "src.clean"],
    [sys.executable, "-m", "src.features"],
    [sys.executable, "-m", "src.ml_models"],
    [sys.executable, "-m", "src.dl_model"],
    [sys.executable, "-m", "src.benchmark"],
)


def main() -> None:
    seed_everything(42)
    for stage in STAGES:
        print(f"Running: {' '.join(stage)}", flush=True)
        subprocess.run(stage, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
