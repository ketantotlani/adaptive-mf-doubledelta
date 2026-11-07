"""List the double-delta dataset's files by type and size. Downloads nothing.
Run:  python size_check.py
"""
from huggingface_hub import HfApi
from collections import defaultdict

info = HfApi().dataset_info("yirens/double-delta-aero", files_metadata=True)
by_type = defaultdict(lambda: [0, 0])
for f in info.siblings:
    n = f.rfilename.lower()
    kind = ("volume" if ("vol" in n or "flow" in n) and n.endswith((".vtk", ".vtu"))
            else "surface" if n.endswith((".vtk", ".vtu", ".vtp"))
            else n.rsplit(".", 1)[-1])
    by_type[kind][0] += 1
    by_type[kind][1] += f.size or 0

for k, (c, s) in sorted(by_type.items(), key=lambda x: -x[1][1]):
    print(f"{k:12s} {c:6d} files {s/1e9:8.1f} GB")
print("\nexample file names:")
for f in info.siblings[:25]:
    print("  ", f.rfilename)
