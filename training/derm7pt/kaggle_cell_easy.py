# Paper 2: external replication (EASY) - CELL 2 of 2
# Run AFTER cell 1 (cell 1 writes /kaggle/working/train_full_easy.py)
import glob
import hashlib
import os
import subprocess
import sys
import time
import zipfile

try:
    LOCAL_TEST
except NameError:
    LOCAL_TEST = False
if LOCAL_TEST:
    WORK = os.environ["P2_OUT"]
    EASY = os.environ["P2_EASY_DIR"]
    SCRIPT = "train_full_easy.py"
else:
    WORK = "/kaggle/working"
    EASY = "/tmp/easy_images"
    SCRIPT = WORK + "/train_full_easy.py"
JOBS_PER_GPU = [
    "effb4:0,effb4:1,effb4:2",          # GPU 0
    "convnext_tiny:0,convnext_tiny:1",  # GPU 1
]

# 1) GPUs
if LOCAL_TEST:
    gpus = ["GPU 0 (test)", "GPU 1 (test)"]
else:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                         text=True).stdout
    gpus = [line for line in out.splitlines() if line.strip()]
print("\n".join(gpus) or "NO GPU FOUND")
assert gpus, "Turn on GPU: Settings -> Accelerator -> GPU T4 x2"
assert os.path.exists(SCRIPT), "Run cell 1 first"

# 1b) Check that cell 1 arrived undamaged. This catches copy/paste
#     damage in seconds, before any download or GPU time is spent.
EXPECTED_FP = "19a99cac1ce2"
chk = subprocess.run([sys.executable, "-m", "py_compile", SCRIPT],
                     capture_output=True, text=True)
with open(SCRIPT) as fh:
    code_text = fh.read()
squeezed = "".join(code_text.split()).encode()
fp = hashlib.md5(squeezed).hexdigest()[:12]
print("cell 1 check:", len(code_text.splitlines()), "lines | compiles:",
      chk.returncode == 0, "| fingerprint:", fp, "| expected:",
      EXPECTED_FP)
if chk.returncode != 0 or fp != EXPECTED_FP:
    print(chk.stderr[-1500:])
    raise RuntimeError("Cell 1 is damaged or is not the tested version."
                       " Import paper2_easy_job.ipynb instead of"
                       " pasting.")
print("cell 1 check: OK")


# 2) Download the 248 EASY images (ISIC collection 166)
def n_images():
    found = glob.glob(EASY + "/**/ISIC_*.jp*g", recursive=True)
    return len({os.path.basename(p) for p in found})


if not LOCAL_TEST and n_images() != 248:
    os.makedirs(EASY, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "isic-cli"], check=False)
    r = subprocess.run(["isic", "image", "download", "-l", "0",
                        "-c", "166", EASY],
                       capture_output=True, text=True, timeout=3600)
    print("isic-cli:", (r.stdout + r.stderr)[-500:])
    if n_images() != 248:
        print("isic-cli gave", n_images(), "images -> using the API")
        import requests
        import pandas as pd
        url = ("https://api.isic-archive.com/api/v2/images/search/"
               "?collections=166&limit=100")
        rows = []
        while url:
            page = requests.get(url, timeout=60).json()
            for res in page["results"]:
                iid = res["isic_id"]
                path = EASY + "/" + iid + ".jpg"
                if not os.path.exists(path):
                    img_url = res["files"]["full"]["url"]
                    data = requests.get(img_url, timeout=120).content
                    with open(path, "wb") as fh:
                        fh.write(data)
                clin = res.get("metadata", {}).get("clinical", {})
                rows.append({"isic_id": iid,
                             "diagnosis_1": clin.get("diagnosis_1")})
            url = page.get("next")
        pd.DataFrame(rows).to_csv(EASY + "/metadata_api.csv",
                                  index=False)
print("EASY images available:", n_images())
assert LOCAL_TEST or n_images() == 248, "EASY download incomplete"

# 3) Train + predict: one job list per GPU, in parallel
if len(gpus) >= 2:
    lists = JOBS_PER_GPU
else:
    lists = [",".join(JOBS_PER_GPU)]
jobs = []
for k, joblist in enumerate(lists):
    env = dict(os.environ)
    env["P2_JOBS"] = joblist
    env["P2_EASY_DIR"] = EASY
    env["P2_OUT"] = WORK
    if len(gpus) >= 2 and not LOCAL_TEST:
        env["CUDA_VISIBLE_DEVICES"] = str(k)
    logf = WORK + "/stdout_easy_gpu" + str(k) + ".txt"
    proc = subprocess.Popen([sys.executable, "-u", SCRIPT],
                            stdout=open(logf, "w"),
                            stderr=subprocess.STDOUT, env=env)
    jobs.append((k, proc, logf))
print("started", len(jobs), "job list(s) on", min(len(gpus), 2),
      "GPU(s)", flush=True)
KEYS = ("==", "epoch", "EASY", "wrote", "DONE", "Error",
        "Traceback", "skip")
seen = {k: 0 for k, _, _ in jobs}
while True:
    for k, proc, logf in jobs:
        lines = open(logf).read().splitlines()
        for line in lines[seen[k]:]:
            if any(s in line for s in KEYS):
                print("[gpu" + str(k) + "]", line, flush=True)
        seen[k] = len(lines)
    if all(proc.poll() is not None for _, proc, _ in jobs):
        break
    time.sleep(5 if LOCAL_TEST else 60)
for k, proc, _ in jobs:
    print("gpu" + str(k), "exit code:", proc.returncode,
          "(0 = success)")

# 4) Zip the small outputs (weights stay in the notebook output)
files = sorted(glob.glob(WORK + "/easy_preds_*.csv")
               + glob.glob(WORK + "/log_full_*.csv"))
with zipfile.ZipFile(WORK + "/p2_easy.zip", "w",
                     zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, os.path.basename(f))
print("\np2_easy.zip:", len(files), "files (expect 10)")
weights = glob.glob(WORK + "/weights/*.pt")
print("weights:", sorted(os.path.basename(p) for p in weights))
