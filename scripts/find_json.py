"""List every file in the dataset repo whose path mentions json (and all top-level entries), with sizes."""
from huggingface_hub import HfApi
info = HfApi().dataset_info("yirens/double-delta-aero", files_metadata=True)
files = [(s.rfilename, s.size or 0) for s in info.siblings]
print(f"{len(files)} files in repo")
print("\n-- paths containing 'json':")
for f, sz in files:
    if "json" in f.lower():
        print(f"  {sz/1e6:9.1f} MB  {f}")
print("\n-- top-level entries:")
for f in sorted({f.split('/')[0] for f, _ in files}):
    print("  ", f)
