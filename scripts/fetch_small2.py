"""Download the remaining small files: design tables, wing reference values, READMEs, VLM JSON readme.
Run:  python fetch_small2.py
"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id="yirens/double-delta-aero", repo_type="dataset", local_dir="data/raw",
    allow_patterns=["geometryDefinition/*", "readme_vlm_json.md",
                    "holdoutSet/README.md", "holdoutSet/config.cfg",
                    "trainingSet/README.md", "trainingSet/config.cfg"])
print("done")
