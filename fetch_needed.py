"""Extract ONLY the files we need from the dataset's big remote zips (no full download).
Keeps: <wing>.json (VLM), surface_flow.vtu (RANS surface), forces_breakdown.dat, history.csv.
Skips: flow.vtu (volume, ~437 MB each). Resumable: files already on disk are skipped.

Examples:
  python fetch_needed.py --set holdout --max-wings 1     # quick test: 1 holdout wing (~70 MB)
  python fetch_needed.py --set holdout                   # all 16 test wings (~1.2 GB)
  python fetch_needed.py --set train --parts 15 16       # two training zips
  python fetch_needed.py --set train                     # all 18 training zips (~19 GB)
Output: <out>/holdoutSet/... and <out>/trainingSet/...  (default out = /Volumes/CRUCIAL_X6/double-delta/raw)
"""
import argparse, os, time
from remotezip import RemoteZip

BASE = "https://huggingface.co/datasets/yirens/double-delta-aero/resolve/main/"
KEEP = ("surface_flow.vtu", "forces_breakdown.dat", "history.csv", ".json")


def wanted(name):
    return name.endswith(KEEP) and not name.endswith("/flow.vtu") and not name.endswith("/")


def pull(zip_path, out_dir, max_wings=0):
    url = BASE + zip_path
    t0, got, skipped, nbytes = time.time(), 0, 0, 0
    with RemoteZip(url) as z:
        infos = [i for i in z.infolist() if wanted(i.filename)]
        if max_wings:
            wings = sorted({i.filename.split("/")[0] for i in infos})[:max_wings]
            infos = [i for i in infos if i.filename.split("/")[0] in wings]
        print(f"{zip_path}: {len(infos)} files to fetch ({sum(i.file_size for i in infos)/1e9:.2f} GB)")
        for k, i in enumerate(infos, 1):
            dest = os.path.join(out_dir, i.filename)
            if os.path.exists(dest) and os.path.getsize(dest) == i.file_size:
                skipped += 1
                continue
            z.extract(i, out_dir)
            got += 1; nbytes += i.file_size
            if k % 25 == 0 or k == len(infos):
                print(f"  {k}/{len(infos)}  {nbytes/1e9:.2f} GB  {time.time()-t0:.0f}s", flush=True)
    print(f"  done: {got} new, {skipped} already present, {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["holdout", "train"], required=True)
    ap.add_argument("--parts", type=int, nargs="*", help="training zip numbers (1-18); default all")
    ap.add_argument("--max-wings", type=int, default=0, help="only the first N wings per zip (testing)")
    ap.add_argument("--out", default="/Volumes/CRUCIAL_X6/double-delta/raw", help="where to save (default: the CRUCIAL_X6 drive)")
    a = ap.parse_args()
    if a.set == "holdout":
        pull("holdoutSet/holdoutSet-part1.zip", os.path.join(a.out, "holdoutSet"), a.max_wings)
    else:
        for p in (a.parts or range(1, 19)):
            pull(f"trainingSet/trainingSet-part{p}.zip", os.path.join(a.out, "trainingSet"), a.max_wings)


if __name__ == "__main__":
    main()
