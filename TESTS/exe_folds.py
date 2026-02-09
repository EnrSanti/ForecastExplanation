import subprocess
from pathlib import Path

N_FOLDS = 5
FOLDS_DIR = Path(f"./{N_FOLDS}folds")

def run_fold(fold_id):
    fold_dir = FOLDS_DIR / f"fold{fold_id}"
    train_file = fold_dir / "train.txt"
    out_file = fold_dir / f"hyp{fold_id}.txt"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing {train_file}")

    # Read the FastLAS command
    cmd = train_file.read_text().strip()

    print(f"[Fold {fold_id}] running FastLAS...")
    print(cmd)

    # Execute command
    proc = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Save output
    out_file.write_text(proc.stdout)

    print(f"[Fold {fold_id}] saved → {out_file}")

def main():
    for i in range(1, N_FOLDS + 1):
        run_fold(i)

if __name__ == "__main__":
    main()
