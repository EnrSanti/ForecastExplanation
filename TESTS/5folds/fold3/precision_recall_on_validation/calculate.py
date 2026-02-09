import os
import subprocess
import re

BG_FILE = "just_bg.las"
HYP_FILE = "hyp.las"
CONFUSION = "confusion.lp"

CTX_DIR = "CTX"
EPLUS_DIR = "E_PLUS"

os.makedirs(CTX_DIR, exist_ok=True)
os.makedirs(EPLUS_DIR, exist_ok=True)
TP_count = 0
FP_count = 0
FN_count = 0
TN_count = 0

files = sorted(f for f in os.listdir(".") if f.startswith("e") and f.endswith(".las"))

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clean_lines(block):
    lines = []
    for line in block.splitlines():
        line = line.strip()

        if not line or line.startswith("%"):
            continue
        if line.endswith(","):
            line = line[:-1] + "."
        elif not line.endswith("."):
            line += "."
        lines.append("gt_"+line)
    return lines

def parse_predictions(answer):
    skies = set()
    rains = set()

    for atom in answer.split():
        atom = atom.strip()
        if atom.startswith("forecasted_sky"):
            skies.add(atom)
        elif atom.startswith("forecasted_rain"):
            rains.add(atom)

    return skies, rains

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

TP = 0
FP = 0
FN = 0

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
import os
import shutil

SOURCE_DIR = "./../../../"
DEST_DIR = "./"
LIST_FILE = "test.txt"

os.makedirs(DEST_DIR, exist_ok=True)

with open(LIST_FILE) as f:
    content = f.read()
    files = [x for x in content.split() if x]

for fname in files:
    src = os.path.join(SOURCE_DIR, fname)
    dst = os.path.join(DEST_DIR, fname)

    if not os.path.exists(src):
        print(f"❌ Missing: {src}")
        continue

    shutil.copy(src, dst)
    print(f"✅ Copied: {fname}")


for i, fname in enumerate(files, 1):
    print(f"\n[{i}/{len(files)}] Processing {fname}")

    with open(fname) as f:
        content = f.read()

    m = re.search(
        r'#pos\([^\{]*\{(.*?)\},\s*\{.*?\},\s*\{(.*)\}\s*\)',
        content,
        re.DOTALL
    )

    if not m:
        print("  ❌ Cannot parse example")
        continue

    eplus_raw, context_raw = m.groups()
    eplus = clean_lines(eplus_raw)

    base = os.path.splitext(fname)[0]

    # --------------------------------------------------------
    # Write CTX
    # --------------------------------------------------------

    ctx_path = os.path.join(CTX_DIR, base + ".las")
    with open(ctx_path, "w") as f:
        f.write(context_raw.strip() + "\n")

    # --------------------------------------------------------
    # Write E_PLUS
    # --------------------------------------------------------

    eplus_path = os.path.join(EPLUS_DIR, base + ".las")
    with open(eplus_path, "w") as f:
        for fact in eplus:
            f.write(fact + "\n")

    # --------------------------------------------------------
    # Gold labels (strip final dot)
    # --------------------------------------------------------

    gold_sky = next(x.rstrip(".") for x in eplus if x.startswith("gt_forecasted_sky"))
    gold_rain = next(x.rstrip(".") for x in eplus if x.startswith("gt_forecasted_rain"))

    # --------------------------------------------------------
    # Run clingo
    # --------------------------------------------------------

    ep_path = os.path.join(EPLUS_DIR, base + ".las")
    cmd = ["clingo", ctx_path,ep_path, BG_FILE, HYP_FILE,CONFUSION]

    print("CMD:", " ".join(cmd))

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    atoms = []
    print(res.stdout)
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("answer:") or line.startswith("Answer:"):
            continue
        atoms.extend(line.split())

    # Count each type
    for a in atoms:
        if a.startswith("tp("):
            TP_count += 1
        elif a.startswith("fp("):
            FP_count += 1
        elif a.startswith("fn("):
            FN_count += 1
        elif a.startswith("tn("):
            TN_count += 1

# Compute metrics
precision = TP_count / (TP_count + FP_count) if (TP_count + FP_count) > 0 else 0.0
recall    = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0.0
accuracy  = (TP_count + TN_count) / (TP_count + TN_count + FP_count + FN_count)

print("\n===== FINAL METRICS =====")
print(f"TP: {TP_count}, FP: {FP_count}, FN: {FN_count}, TN: {TN_count}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")