"""Raw Shen & Alonso double-delta files -> processed format (see data.py).

    python -m mfb.preprocess --sets holdout                 # 16 test wings
    python -m mfb.preprocess --sets holdout train           # everything downloaded so far (re-run any time)

Reads from  /Volumes/CRUCIAL_X6/double-delta/raw   writes to  /Volumes/CRUCIAL_X6/double-delta/processed
Snapshots already processed are skipped, so it can be re-run as more training wings download.

What it does, per wing and angle of attack:
  * nodes = the 960 VLM lattice panels (30 chordwise x 32 spanwise), from <wing>.json
  * input feature = VLM pressure on each panel (a loading: lower minus upper, as SUAVE outputs it)
  * target = RANS (SU2) surface pressure from surface_flow.vtu, mapped onto the same panels:
      each surface triangle is assigned to the panel whose planform quad contains its centre,
      Cp is averaged per panel and per side (weighted by projected area), target dCp = Cp_lower - Cp_upper
  * both solvers' lift/moment are computed with the SAME reference area and moment centre (wing_reference*.csv)
  * checks: (1) integrating the raw RANS surface must reproduce forces_breakdown.dat pressure CL/CMy,
            (2) integrating the mapped panel dCp shows how much of the RANS lift the 960-panel target keeps.
    Both go to processed/validation.csv.
"""
import argparse, csv, json, pathlib, re
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from .data import knn_edges, write_index

FLAT_RANS_CPUH = 310835 / 2448          # paper: 310,835 CPU-hours for 2,448 snapshots (~127 per snapshot)
LREF = 16.0                             # SU2 REF_LENGTH used for CMy
RAW = "/Volumes/CRUCIAL_X6/double-delta/raw"
OUT = "/Volumes/CRUCIAL_X6/double-delta/processed"
DESIGN_KEYS = ["DROOP_ANGLE", "SW1", "SW2", "SR2", "BW2", "B"]


# ------------------------------------------------------------------ tables
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def tables(raw):
    g = pathlib.Path(raw) / "geometryDefinition"
    train = read_csv(g / "designs_256samples.csv"); hold = read_csv(g / "designs_holdout.csv")
    design = {r["design-call-sign"]: [float(r[k]) for k in DESIGN_KEYS] for r in train + hold}
    rank = {r["design-call-sign"]: i for i, r in enumerate(train)}        # nested Sobol order = row order
    ref = {r["design-call-sign"]: (float(r["reference_area"]), float(r["aero_center_x"]))
           for r in read_csv(g / "wing_reference.csv") + read_csv(g / "wing_reference_holdout.csv")}
    ausm_file = pathlib.Path(raw) / "AUSMruns.info.txt"
    ausm = ausm_file.read_text(errors="ignore") if ausm_file.exists() else ""
    return design, rank, ref, ausm


# ------------------------------------------------------------------ readers
def read_vlm(path):
    """Panel geometry (shared by all angles) + VLM loading per angle."""
    j = json.loads(pathlib.Path(path).read_text())
    vd = j["vlm_vd"]; g = lambda k: np.asarray(vd[k], dtype=np.float64)
    quads = np.stack([np.c_[g("XA1"), g("YA1")], np.c_[g("XB1"), g("YB1")],
                      np.c_[g("XB2"), g("YB2")], np.c_[g("XA2"), g("YA2")]], 1)     # (960,4,2), ordered around the panel
    x, y = quads[..., 0], quads[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1))   # planform area per panel
    pos = np.c_[g("XC"), g("YC"), g("ZC")]
    runs = {int(round(r["angle_of_attack"])): np.asarray(r["vlm_pressure_coefficient"][0], np.float64) for r in j["runs"]}
    return dict(quads=quads, area=area, pos=pos, thick=g("thickness"), runs=runs)


def read_surface(path):
    """SU2 surface: triangle centres, outward area vectors, Cp per triangle."""
    import pyvista as pv
    s = pv.read(str(path)).extract_surface(algorithm="dataset_surface")
    if not s.is_all_triangles:
        s = s.triangulate()
    tri = s.faces.reshape(-1, 4)[:, 1:]; P = np.asarray(s.points, np.float64)
    cp = np.asarray(s.point_data["Pressure_Coefficient"], np.float64)
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    nA = 0.5 * np.cross(b - a, c - a); cen = (a + b + c) / 3
    if (cen * nA).sum() < 0:          # enclosed volume must be positive for outward normals
        nA = -nA
    return cen, nA, cp[tri].mean(1)


def read_forces(path):
    """Pressure-only CL and CMy from forces_breakdown.dat (the part a pressure surrogate can explain)."""
    txt = pathlib.Path(path).read_text(errors="ignore")
    def get(k):
        m = re.search(rf"Total {k}:\s+([-0-9.eE+]+) \| Pressure \(\s*[-0-9]+%\):\s+([-0-9.eE+]+)", txt)
        return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
    (cl_tot, cl_p), (cm_tot, cm_p) = get("CL"), get("CMy")
    return dict(cl_total=cl_tot, cl_pressure=cl_p, cmy_total=cm_tot, cmy_pressure=cm_p)


# ------------------------------------------------------------------ physics
def coeffs_from_surface(cen, nA, cp, aoa, sref, xref):
    f = -cp[:, None] * nA                      # pressure force per triangle (in units of q_inf)
    F = f.sum(0); a = np.radians(aoa)
    cl = (F[2] * np.cos(a) - F[0] * np.sin(a)) / sref
    my = (cen[:, 2] * f[:, 0] - (cen[:, 0] - xref) * f[:, 2]).sum()
    return cl, my / (sref * LREF)


def coeffs_from_panels(dcp, area, pos, aoa, sref, xref):
    """Same formula for VLM, mapped RANS and model predictions (normal force on flat panels)."""
    fz = dcp * area; a = np.radians(aoa)
    return fz.sum() * np.cos(a) / sref, -(fz * (pos[:, 0] - xref)).sum() / (sref * LREF)


def assign_to_panels(cen, quads):
    """Index of the panel whose planform quad contains each triangle centre (nearest panel if none)."""
    qc = quads.mean(1); tree = cKDTree(qc)
    _, cand = tree.query(cen[:, :2], k=8)
    p = cen[:, None, :2]                                          # (M,1,2)
    Q = quads[cand]                                               # (M,8,4,2)
    e = np.roll(Q, -1, 2) - Q; r = p[:, :, None, :] - Q
    cross = e[..., 0] * r[..., 1] - e[..., 1] * r[..., 0]         # (M,8,4)
    inside = (cross >= -1e-9).all(2) | (cross <= 1e-9).all(2)
    first = np.where(inside.any(1), inside.argmax(1), 0)          # fallback: nearest centre
    return cand[np.arange(len(cen)), first], inside.any(1)


def map_to_panels(cen, nA, cp, quads):
    """Per-panel Cp on upper and lower side (projected-area weighted) and dCp = lower - upper."""
    idx, hit = assign_to_panels(cen, quads)
    w = np.abs(nA[:, 2]); up = nA[:, 2] > 0; n = len(quads)
    out = {}
    for side, m in (("up", up), ("lo", ~up)):
        sw = np.bincount(idx[m], w[m], n); scp = np.bincount(idx[m], w[m] * cp[m], n)
        out[side] = np.where(sw > 0, scp / np.maximum(sw, 1e-12), np.nan)
    for side in ("up", "lo"):                                     # fill any empty panel from its neighbours
        bad = np.isnan(out[side])
        if bad.any():
            good = np.where(~bad)[0]; t = cKDTree(quads.mean(1)[good])
            out[side][bad] = out[side][good[t.query(quads.mean(1)[bad])[1]]]
    return out["lo"] - out["up"], out["up"], out["lo"], float(hit.mean())


# ------------------------------------------------------------------ one wing
def process_wing(job):
    wing_dir, split, design, rank, sref, xref, ausm, out, force = job
    wing_dir = pathlib.Path(wing_dir); geom = wing_dir.name; out = pathlib.Path(out)
    jf = wing_dir / f"{geom}.json"
    if not jf.exists():
        return [], f"{geom}: no json yet"
    vlm = None; rows = []; notes = []
    for aoa_dir in sorted(wing_dir.glob("*degAOA")):
        aoa = int(round(float(aoa_dir.name.replace("degAOA", ""))))
        sid = f"{geom}_a{aoa:02d}"; npz = out / "snap" / f"{sid}.npz"
        vtu, fb = aoa_dir / "surface_flow.vtu", aoa_dir / "forces_breakdown.dat"
        if force or not npz.exists():
            if not (vtu.exists() and fb.exists()):
                notes.append(f"{sid}: files missing"); continue
            if vlm is None:
                vlm = read_vlm(jf)
            if aoa not in vlm["runs"]:
                notes.append(f"{sid}: no VLM run at this angle"); continue
            cen, nA, cp = read_surface(vtu)
            dcp, cp_up, cp_lo, hit = map_to_panels(cen, nA, cp, vlm["quads"])
            pos, area, dcp_vlm = vlm["pos"], vlm["area"], vlm["runs"][aoa]
            fr = read_forces(fb)
            cl_s, cm_s = coeffs_from_surface(cen, nA, cp, aoa, sref, xref)
            cl_m, cm_m = coeffs_from_panels(dcp, area, pos, aoa, sref, xref)
            cl_v, cm_v = coeffs_from_panels(dcp_vlm, area, pos, aoa, sref, xref)
            a_r = np.radians(aoa); n = len(pos)
            # no VLM "thickness": training and holdout json store it differently (train ~ -0.03, holdout ~ 0.5)
            x = np.concatenate([pos, area[:, None], np.repeat(np.asarray(design)[None], n, 0),
                                np.full((n, 1), np.sin(a_r)), np.full((n, 1), np.cos(a_r)), dcp_vlm[:, None]], 1)
            ei, ea = knn_edges(pos)
            meta = dict(cl_su2_pressure=fr["cl_pressure"], cmy_su2_pressure=fr["cmy_pressure"],
                        cl_su2_total=fr["cl_total"], cmy_su2_total=fr["cmy_total"],
                        cl_surface=cl_s, cmy_surface=cm_s, cl_mapped=cl_m, cmy_mapped=cm_m,
                        cl_vlm=cl_v, cmy_vlm=cm_v, frac_in_quad=hit)
            npz.parent.mkdir(parents=True, exist_ok=True)
            tmp = npz.with_suffix(".tmp.npz")
            np.savez(tmp, x=x.astype(np.float32), y=dcp.astype(np.float32),
                     y_up=cp_up.astype(np.float32), y_lo=cp_lo.astype(np.float32),
                     pos=pos.astype(np.float32), area=area.astype(np.float32), edge_index=ei, edge_attr=ea,
                     sref=np.float32(sref), xref=np.float32(xref), lref=np.float32(LREF),
                     meta=json.dumps(meta))
            tmp.replace(npz)
        else:
            meta = json.loads(str(np.load(npz)["meta"]))
        rows.append(dict(sid=sid, geom=geom, aoa_deg=aoa, split=split, sobol_rank=rank, cost_cpuh=round(FLAT_RANS_CPUH, 2),
                         ausm=int(geom in ausm), **{f"d{i+1}": design[i] for i in range(len(design))}, **meta))
    return rows, "; ".join(notes)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW); ap.add_argument("--out", default=OUT)
    ap.add_argument("--sets", nargs="+", choices=["holdout", "train"], default=["holdout", "train"])
    ap.add_argument("--max-wings", type=int, default=0, help="per set, for a quick test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--redo-list", default=None, help="file with wing names to reprocess even if done (e.g. raw/json_updated.txt)")
    a = ap.parse_args()
    raw, out = pathlib.Path(a.raw), pathlib.Path(a.out)
    design, rank, ref, ausm = tables(raw)
    redo = set(pathlib.Path(a.redo_list).read_text().split()) if a.redo_list else set()

    jobs = []
    for s in a.sets:
        folder = raw / ("holdoutSet" if s == "holdout" else "trainingSet")
        wings = sorted((p for p in folder.rglob("*") if p.is_dir() and p.name in design), key=lambda p: p.name)  # training wings sit in trainingSet-partN/
        if a.max_wings:
            wings = wings[: a.max_wings]
        for w in wings:
            sref, xref = ref[w.name]
            jobs.append((str(w), "test" if s == "holdout" else "train", design[w.name],
                         -1 if s == "holdout" else rank[w.name], sref, xref, ausm, str(out), w.name in redo))
    print(f"{len(jobs)} wings found ({', '.join(a.sets)}); expected {16 * ('holdout' in a.sets) + 256 * ('train' in a.sets)}")

    # keep rows already in the index for wings not in this run (e.g. run holdout, later train)
    rows = {}
    old = out / "index.csv"
    if old.exists():
        for r in read_csv(old):
            rows[r["sid"]] = r
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (rs, note) in enumerate(ex.map(process_wing, jobs), 1):
            for r in rs:
                rows[r["sid"]] = r
            if note:
                print("  note:", note)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} wings, {len(rows)} snapshots so far")
    rows = sorted(rows.values(), key=lambda r: r["sid"])
    if not rows:
        print("nothing processed"); return
    write_index(out, [{k: r[k] for k in ["sid", "geom", "aoa_deg", "split", "sobol_rank", "cost_cpuh", "ausm"]
                       + [f"d{i+1}" for i in range(6)]} for r in rows])
    vkeys = ["sid", "split", "cl_su2_pressure", "cl_surface", "cl_mapped", "cl_vlm",
             "cmy_su2_pressure", "cmy_surface", "cmy_mapped", "cmy_vlm", "frac_in_quad"]
    with open(out / "validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vkeys); w.writeheader(); w.writerows([{k: r[k] for k in vkeys} for r in rows])

    V = {k: np.array([float(r[k]) for r in rows]) for k in vkeys[2:]}
    print(f"\nwrote {len(rows)} snapshots to {out}  "
          f"(train {sum(r['split'] == 'train' for r in rows)}, test {sum(r['split'] == 'test' for r in rows)})")
    print("checks (mean |difference| over all snapshots):")
    print(f"  reading:  my surface integral vs SU2      CL {np.nanmean(abs(V['cl_surface'] - V['cl_su2_pressure'])):.4f}"
          f"   CMy {np.nanmean(abs(V['cmy_surface'] - V['cmy_su2_pressure'])):.4f}   (should be ~0)")
    print(f"  mapping:  960-panel target vs SU2         CL {np.nanmean(abs(V['cl_mapped'] - V['cl_su2_pressure'])):.4f}"
          f"   CMy {np.nanmean(abs(V['cmy_mapped'] - V['cmy_su2_pressure'])):.4f}")
    print(f"  fidelity: VLM vs RANS (same panels/refs)  CL {np.nanmean(abs(V['cl_vlm'] - V['cl_mapped'])):.4f}"
          f"   CMy {np.nanmean(abs(V['cmy_vlm'] - V['cmy_mapped'])):.4f}")
    print(f"  triangles inside a panel quad: {V['frac_in_quad'].mean():.1%}")


if __name__ == "__main__":
    main()
