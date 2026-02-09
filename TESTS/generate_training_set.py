import os
import re
from datetime import date, datetime

# ---------------- CONFIG ----------------
LAS_DIR = "."

FASTLAS_CMD = "../FastLAS_mod --nopl bg.las"
DEBUG_FLAG = "--debug --threads 12"

CITIES = [
    "barcis",
    "gemona_stolvizza",
    "gorizia",
    "lignano_grado",
    "pontebba_tarvisio",
    "pordenone",
    "sappada_forni_villa",
    "trieste",
    "udine_palmanova",
]
# ----------------------------------------

FILENAME_RE = re.compile(
    r"example_day_(\d+)_(\d+)_(\d+)_([a-z_]+)\.las"
)

# -------------------------------------------------
# Utilities
# -------------------------------------------------




def split_into_intervals(dates):
    intervals = []
    current = [dates[0]]

    for prev, cur in zip(dates, dates[1:]):
        if (cur - prev).days > 1:
            intervals.append(current)
            current = []
        current.append(cur)

    intervals.append(current)
    return intervals


def expand_dates_to_files(dates):
    """
    For each selected date, generate all 9 city filenames.
    """
    files = []
    for d in dates:
        for city in CITIES:
            fname = f"example_day_{d.year}_{d.month}_{d.day}_{city}.las"
            path = os.path.join(LAS_DIR, fname)
            if os.path.isfile(path):
                files.append(fname)
            else:
                print(f"[WARN] missing file: {fname}")
    return files

def parse_date(fname):
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    y, mth, d = map(int, m.groups()[:3])
    return date(y, mth, d)

def load_all_dates():
    dates = set()
    for f in os.listdir(LAS_DIR):
        d = parse_date(f)
        if d:
            dates.add(d)
    return sorted(dates)




# -------------------------------------------------
# TRAINING
# -------------------------------------------------

def assign_days_to_folds(intervals, k):
    folds = [[] for _ in range(k)]
    fold_id = 0

    for interval in intervals:
        for d in interval:
            folds[fold_id].append(d)
            fold_id = (fold_id + 1) % k

    return folds


def generate_cv_folds(k=5):
    dates = load_all_dates()
    intervals = split_into_intervals(dates)

    test_folds = assign_days_to_folds(intervals, k)

    os.makedirs("folds", exist_ok=True)

    for i, test_dates in enumerate(test_folds, start=1):
        train_dates = [d for d in dates if d not in test_dates]

        print(f"[Fold {i}] train={len(train_dates)} test={len(test_dates)}")

        train_files = expand_dates_to_files(train_dates)
        test_files  = expand_dates_to_files(test_dates)

        fold_dir = f"./folds/fold{i}"
        os.makedirs(fold_dir, exist_ok=True)

        # ---- TRAIN COMMAND ----
        train_cmd = " ".join([
            FASTLAS_CMD,
            *train_files,
            DEBUG_FLAG
        ])

        with open(f"{fold_dir}/train.txt", "w") as f:
            f.write(train_cmd + "\n")

        # ---- TEST FILE LIST ----
        with open(f"{fold_dir}/test.txt", "w") as f:
            for fn in test_files:
                f.write(fn + "\n")

        print(f"[saved] fold {i}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    generate_cv_folds(k=4)

if __name__ == "__main__":
    main()
