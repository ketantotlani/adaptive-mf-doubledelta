"""Step 2: describe what was downloaded so the preprocessor can be fitted to the real format.

    python -m mfb.inspect_data data/raw > inspect.txt

Prints: folder layout, the key structure of one VLM JSON (with array shapes), and the
point/cell arrays of one surface VTK. Share inspect.txt to finalise mfb/preprocess.py.
"""
import json, sys, pathlib, collections


def describe(obj, depth=0, maxdepth=4):
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:40]:
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__}{shape(v)}")
                if depth < maxdepth:
                    describe(v, depth + 1, maxdepth)
            else:
                print(f"{pad}{k}: {v!r}"[:160])
    elif isinstance(obj, list) and obj and isinstance(obj[0], (dict, list)) and depth < maxdepth:
        describe(obj[0], depth + 1, maxdepth)


def shape(v):
    s = []
    while isinstance(v, list) and v:
        s.append(len(v)); v = v[0]
    return f" shape={tuple(s)}" if s else ""


def main(root):
    root = pathlib.Path(root)
    files = [p for p in root.rglob("*") if p.is_file()]
    print(f"{len(files)} files under {root}")
    print("by extension:", dict(collections.Counter(p.suffix for p in files)))
    for p in sorted(files)[:40]:
        print("  ", p.relative_to(root), f"{p.stat().st_size/1e6:.2f} MB")
    js = sorted(p for p in files if p.suffix == ".json")
    if js:
        print(f"\n=== JSON: {js[0].relative_to(root)} ===")
        describe(json.loads(js[0].read_text()))
    vt = sorted(p for p in files if p.suffix == ".vtk")
    if vt:
        import pyvista as pv
        m = pv.read(vt[0])
        print(f"\n=== VTK: {vt[0].relative_to(root)} ===\n{m}")
        print("point arrays:", {k: m.point_data[k].shape for k in m.point_data})
        print("cell arrays:", {k: m.cell_data[k].shape for k in m.cell_data})
        print("bounds:", m.bounds)
    other = sorted(p for p in files if p.suffix in {".csv", ".dat", ".txt"})
    for p in other[:3]:
        print(f"\n=== {p.relative_to(root)} (first 8 lines) ===")
        print("\n".join(p.read_text(errors="ignore").splitlines()[:8]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
