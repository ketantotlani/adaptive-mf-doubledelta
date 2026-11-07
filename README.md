# Adaptive multi-fidelity surrogates: double-delta wing baseline

Baselines and an adaptive-sampling loop for learning RANS surface pressure from a cheap vortex-lattice (VLM)
solution, on the Shen & Alonso double-delta wing dataset.

- Dataset: 272 wings (256 Sobol design + 16 holdout) x 9 angles of attack (11-19 deg), Mach 0.3; VLM (SUAVE) and
  SU2 RANS (SA-R). https://huggingface.co/datasets/yirens/double-delta-aero (CC-BY-2.0)
- Paper: Shen & Alonso, https://arxiv.org/abs/2512.20941 (published GNN baseline: test MSE on Cp 0.052-0.060)

**Status (Nov 2026).** Test wings (16) downloaded and processed, with checks against SU2's force outputs
(`results/validation_holdout.csv`). Training wings next; B0/B1 not yet run on real data.

**AI-assisted code.** Parts of this code were written with AI assistance. All results are produced by running it
here and are checked against the solvers' own outputs (see Checks below).

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Uses an NVIDIA GPU or Apple-silicon GPU automatically if present.

## Pipeline
| Step | Command | Output |
| --- | --- | --- |
| 1. Download only the needed files (~20 GB of 612 GB) | `python fetch_needed.py --set holdout` then `--set train` | raw files on the data drive |
| 2. Preprocess | `python -m mfb.preprocess --sets holdout train` | `processed/index.csv`, one `.npz` per case, `validation.csv` |
| 3. B0: VLM alone | `python -m mfb.train --data <processed> --b0` | VLM error on test wings, spread of the gap |
| 4. B1: GNN, all data | `python -m mfb.train --data <processed> --mode all` | compare with the paper |
| 5. B2/B3: scaling | `--mode random` / `--mode sobol` with `--n-geoms 8 16 32 64 128 --seeds 0 1 2` | error vs RANS CPU-hours |
| 6. Adaptive loop | `python -m mfb.active --strategy uncertainty_diverse --budget-geoms 64 --seeds 0 1 2` | error per round vs CPU-hours |
| Timing | `python bench.py` | estimated run times on this machine |

Default data location is `/Volumes/CRUCIAL_X6/double-delta/{raw,processed}`; change with `--raw/--out/--data`.
Results are appended to `results/*.jsonl`.

## How the data is prepared (`mfb/preprocess.py`)
- Nodes: the 960 VLM lattice panels (30 chordwise x 32 spanwise), kNN graph (k = 8) over panel centres.
- Input: geometry, design variables, angle of attack and the VLM panel loading (lower minus upper Cp).
- Target: SU2 surface Cp mapped onto the same panels. Each surface triangle goes to the panel whose planform
  quad contains it; Cp is averaged per side (projected-area weighted); target = Cp_lower - Cp_upper.
  Upper and lower Cp are also stored.
- Lift and moment for both solvers use each wing's reference area and moment centre (`wing_reference*.csv`),
  reference length 16.

## Checks (16 test wings, 144 cases)
| Check | Result |
| --- | --- |
| Surface integral of SU2 Cp vs SU2 reported pressure lift / moment | 0.0000 / 0.0002 |
| 960-panel target vs SU2 lift | mean 0.012, max 0.042 |
| Surface triangles inside a VLM panel | 92% (rest assigned to nearest panel) |

Known issues: the CL/CM stored in the VLM json do not match an integration of their own panel pressures, so they
are not used. Cases rerun with AUSM (`AUSMruns.info.txt`) are flagged (`ausm` column in index.csv);
150201-4SN at 18 deg looks inconsistent. RANS cost is a flat ~127 CPU-h per case (no per-run times in the logs).

## Layout
`mfb/` pipeline: `preprocess.py` (raw -> graphs), `data.py` (format, graph), `model.py` (GATv2 GNN with K heads),
`core.py` (training, metrics, uncertainty), `train.py` (B0-B3), `active.py` (adaptive loop), `synthetic.py` (smoke test)
· `fetch_needed.py` download · `bench.py` timing · `scripts/` one-off helpers used while exploring the dataset
· `results/` small result files.
