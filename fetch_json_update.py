"""Replace outdated VLM json files with the corrected ones the dataset ships in json/.

The dataset's own update_trainingset_json.sh says the json files inside the trainingSet zips are outdated and
the current versions are in the top-level json/ folder. This script:
  1. downloads json/*.json from Hugging Face into <raw>/json   (~1 GB, on the external drive)
  2. compares each with the copy in holdoutSet/ or trainingSet/ and overwrites the ones that differ
  3. writes the changed wing names to <raw>/json_updated.txt (used by: python -m mfb.preprocess --redo-list ...)

    python fetch_json_update.py
"""
import argparse, hashlib, json, pathlib, shutil
import numpy as np
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="/Volumes/CRUCIAL_X6/double-delta/raw")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
raw = pathlib.Path(a.raw)

snapshot_download("yirens/double-delta-aero", repo_type="dataset", allow_patterns=["json/*.json"], local_dir=str(raw))
new_files = sorted((raw / "json").glob("*.json"))
print(f"{len(new_files)} corrected json files downloaded")

dest = {p.name: p for p in (raw / "trainingSet").glob("*/*/*.json")}
dest.update({p.name: p for p in (raw / "holdoutSet").glob("*/*.json")})
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()

changed, same, unmatched, diffs = [], 0, [], []
for f in new_files:
    d = dest.get(f.name)
    if d is None:
        unmatched.append(f.name); continue
    if md5(f) == md5(d):
        same += 1; continue
    try:   # how different is the VLM pressure? (max over panels and angles)
        o, n = json.loads(d.read_text()), json.loads(f.read_text())
        po = {r["angle_of_attack"]: np.array(r["vlm_pressure_coefficient"][0]) for r in o["runs"]}
        pn = {r["angle_of_attack"]: np.array(r["vlm_pressure_coefficient"][0]) for r in n["runs"]}
        diffs.append(max(float(np.abs(po[k] - pn[k]).max()) for k in pn if k in po))
    except Exception:
        diffs.append(float("nan"))
    changed.append(f.stem)
    if not a.dry_run:
        shutil.copyfile(f, d)

split = lambda names: (sum(n in {p.stem for p in (raw / 'holdoutSet').glob('*/*.json')} for n in names))
print(f"identical: {same}   changed: {len(changed)} (of which holdout: {split(changed)})   no match: {len(unmatched)}")
if diffs:
    d = np.array(diffs)
    print(f"max |change| in VLM pressure per changed wing: median {np.nanmedian(d):.3f}, largest {np.nanmax(d):.3f}")
(raw / "json_updated.txt").write_text("\n".join(changed) + "\n")
print("dry run, nothing replaced" if a.dry_run else f"replaced {len(changed)} files; list in {raw / 'json_updated.txt'}")
