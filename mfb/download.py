"""Step 1: list the dataset's files, then download only what the baseline needs.

    python -m mfb.download --list                 # prints the file tree summary, downloads nothing
    python -m mfb.download --out data/raw --include "*.json" "*surface*.vtk" "*.csv" --limit-geoms 3

Patterns are guesses until --list shows the real names: adjust --include after looking.
Skip volume VTK files (millions of cells each); they are most of the dataset's size.
"""
import argparse, collections, fnmatch
from huggingface_hub import list_repo_files, snapshot_download

REPO = "yirens/double-delta-aero"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--include", nargs="*", default=["*.json", "*surf*.vtk", "*.csv", "*.dat", "*.txt", "*.cfg"])
    ap.add_argument("--exclude", nargs="*", default=["*vol*", "*volume*", "*restart*"])
    ap.add_argument("--limit-geoms", type=int, default=0, help="download only the first N geometry folders (0 = all)")
    a = ap.parse_args()

    files = list_repo_files(REPO, repo_type="dataset")
    if a.list:
        exts = collections.Counter(f.rsplit(".", 1)[-1] if "." in f else "(none)" for f in files)
        tops = collections.Counter(f.split("/")[0] for f in files)
        print(f"{len(files)} files")
        print("by extension:", dict(exts.most_common()))
        print("top-level entries:", len(tops), "e.g.", list(tops)[:10])
        for f in files[:60]:
            print("  ", f)
        return

    keep = [f for f in files
            if any(fnmatch.fnmatch(f, p) for p in a.include)
            and not any(fnmatch.fnmatch(f, p) for p in a.exclude)]
    if a.limit_geoms:
        geoms = sorted({f.split("/")[0] for f in keep})[: a.limit_geoms]
        keep = [f for f in keep if f.split("/")[0] in geoms or "/" not in f]
    print(f"downloading {len(keep)} of {len(files)} files to {a.out}")
    snapshot_download(REPO, repo_type="dataset", local_dir=a.out, allow_patterns=keep)


if __name__ == "__main__":
    main()
