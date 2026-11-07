"""Show every file in the dataset outside the per-angle example folders, with sizes. Downloads nothing.
Run:  python list_files.py
"""
from huggingface_hub import HfApi
info = HfApi().dataset_info("yirens/double-delta-aero", files_metadata=True)
files = sorted(info.siblings, key=lambda f: f.rfilename)
print("== all zip files ==")
for f in files:
    if f.rfilename.endswith(".zip"):
        print(f"{(f.size or 0)/1e9:8.2f} GB  {f.rfilename}")
print("\n== everything that is not inside example/<angle>/ ==")
for f in files:
    n = f.rfilename
    if not (n.startswith("example/") and n.count("/") >= 2):
        print(f"{(f.size or 0)/1e6:10.1f} MB  {n}")
