import glob, re, subprocess, sys, os, time
SEEDS_PARALLEL = [0, 1]                       # one seed per GPU
FOLDS = [0, 1, 2, 3, 4]
try: INPUT_GLOB_ROOT
except NameError: INPUT_GLOB_ROOT, WORK, LOCAL_TEST = "/kaggle/input", "/kaggle/working", False
os.makedirs(WORK, exist_ok=True)

gpus = [l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.strip()] \
       if not LOCAL_TEST else ["GPU 0 (test)", "GPU 1 (test)"]
print("\n".join(gpus) or "NO GPU FOUND"); assert gpus, "Turn on GPU: Settings -> Accelerator -> GPU T4 x2"

src = glob.glob(f"{INPUT_GLOB_ROOT}/**/kaggle_dump_concepts.py", recursive=True)
assert src, "kaggle_dump_concepts.py not found in the Input panel"
base = open(src[0]).read()
assert 'BACKBONE' in base and 'convnext_tiny' in base, \
    "This is the OLD script. Upload the new kaggle_dump_concepts.py as a new version of your dataset first."

def patch(code, pairs):
    for pat, rep in pairs:
        code, n = re.subn(pat, rep, code, count=1); assert n == 1, "patch failed: " + pat
    return code

jobs = []
for k, seed in enumerate(SEEDS_PARALLEL):
    code = patch(base, [(r'FOLDS_TO_RUN\s*=\s*\[.*?\]', f'FOLDS_TO_RUN = {FOLDS}'),
                        (r'SEEDS\s*=\s*\[.*?\]',        f'SEEDS = [{seed}]'),
                        (r'DATA_FRACS\s*=\s*\[.*?\]',   'DATA_FRACS = [1.0]'),
                        (r'BACKBONE\s*=\s*"[a-z0-9_]+"', 'BACKBONE = "convnext_tiny"')])
    path = f"{WORK}/run_convnext_seed{seed}.py"; open(path, "w").write(code)
    env = dict(os.environ)
    if len(gpus) >= 2: env["CUDA_VISIBLE_DEVICES"] = str(k)      # seed k -> GPU k
    log = open(f"{WORK}/stdout_convnext_seed{seed}.txt", "w")
    jobs.append((seed, subprocess.Popen([sys.executable, "-u", path], stdout=log, stderr=subprocess.STDOUT, env=env), log))
    if len(gpus) < 2:                                              # one GPU: run sequentially
        jobs[-1][1].wait()
print(f"started {len(jobs)} job(s) on {min(len(gpus), 2)} GPU(s)", flush=True)

seen = {s: 0 for s, _, _ in jobs}
while True:
    for seed, p, _ in jobs:
        lines = open(f"{WORK}/stdout_convnext_seed{seed}.txt").read().splitlines()
        for l in lines[seen[seed]:]:
            if "'mel_auc_bb'" in l or l.startswith(("==", "wrote", "all 1011", "DONE", "device")) or "Error" in l \
               or "MISSING" in l or "Traceback" in l or ("'epoch': 1," in l):
                print(f"[seed {seed}] {l}", flush=True)
        seen[seed] = len(lines)
    if all(p.poll() is not None for _, p, _ in jobs): break
    time.sleep(5 if LOCAL_TEST else 60)
for seed, p, log in jobs:
    log.close(); print(f"seed {seed} exit code: {p.returncode} (0 = success)")
print("\noof files:", sorted(f for f in os.listdir(WORK) if f.startswith("oof_convnext")))
