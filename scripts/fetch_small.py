"""Download only the small files (README, example wing VLM json, configs, force summaries, histories).
Skips flow.vtu and all zip files. Run:  python fetch_small.py
"""
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="yirens/double-delta-aero", repo_type="dataset", local_dir="data/raw",
    allow_patterns=["README.md", "Util.py", "AUSMruns.info.txt", "*.sh",
                    "example/*.json", "example/*/config.cfg",
                    "example/*/forces_breakdown.dat", "example/*/history.csv"],
)
print("saved to", path)
