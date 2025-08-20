from pathlib import Path

# (RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
#     Path(z)
#     for z in [
#         data["RESULTS_DIR"],
#         data["DATA_DIR"],
#         data["STATS_DIR"],
#         data["HPARAMS_DIR"],
#         data["KV_DIR"],
#     ]
# )

  # Result files
#   RESULTS_DIR: "results"

#   # Data files
#   DATA_DIR: "data"
#   STATS_DIR: "data/stats"
#   KV_DIR: "share/projects/rewriting-knowledge/kvs"

#   # Hyperparameters
#   HPARAMS_DIR: "hparams"

#   # Remote URLs
#   REMOTE_ROOT_URL: "https://memit.baulab.info"

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
STATS_DIR = Path("data/stats")
HPARAMS_DIR = Path("hparams")