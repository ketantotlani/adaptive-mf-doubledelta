"""Read the table of contents of a remote zip WITHOUT downloading it, and optionally pull single files.
Needs:  pip install remotezip
Run:    python peek_zip.py                      -> lists files inside holdoutSet-part1.zip (sizes)
        python peek_zip.py --extract-one        -> also extracts ONE surface_flow.vtu + one json to data/raw/peek/
"""
import sys, collections
from remotezip import RemoteZip
URL = "https://huggingface.co/datasets/yirens/double-delta-aero/resolve/main/holdoutSet/holdoutSet-part1.zip"
with RemoteZip(URL) as z:
    infos = z.infolist()
    print(f"{len(infos)} entries in holdout zip")
    by_name = collections.defaultdict(lambda: [0, 0])
    for i in infos:
        base = i.filename.rsplit("/", 1)[-1] or "(dir)"
        if base.endswith(".json"): base = "<wing>.json"
        by_name[base][0] += 1; by_name[base][1] += i.file_size
    for k, (n, s) in sorted(by_name.items(), key=lambda x: -x[1][1]):
        print(f"{k:28s} {n:5d} files  {s/1e9:8.2f} GB total  {s/max(n,1)/1e6:9.1f} MB each")
    print("\nfirst entries:")
    for i in infos[:15]:
        print("  ", i.filename)
    if "--extract-one" in sys.argv:
        surf = next(i for i in infos if i.filename.endswith("surface_flow.vtu"))
        js = next(i for i in infos if i.filename.endswith(".json"))
        for i in (surf, js):
            z.extract(i, "data/raw/peek")
            print("extracted", i.filename, f"{i.file_size/1e6:.1f} MB")
