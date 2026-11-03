# Storage ledger

Budget: 10 GB of additional storage for this run. Measured with `du`.

| When (IST) | Action | Size | Running total | Free disk |
|---|---|---:|---:|---:|
| 2026-11-02 21:15 | Starting point. Not counted: `~/signo_data` 14 GB, backup `~/signo_backup_2026-11-02` 777 MB, both made before this run | 0 | 0 | 121 GB |
| 2026-11-02 21:30 | Started `~/signo_data/asl_citizen/raw`: raw landmarks for 22,421 existing videos. No download. Benchmark: 49 KB per clip, so about 1.1 GB expected | about 1.1 GB planned | about 1.1 GB | 121 GB |
| 2026-11-02 22:40 | Downloaded 8,161 unused official test clips to `fresh_test_videos` (delete after processing) | 4.1 GB | 5.2 GB | about 116 GB |
| 2026-11-02 22:49 | Six partial feature sets in `fs/` (delete when gates end) | 0.78 GB | 6.0 GB | about 116 GB |
| 2026-11-03 00:10 | Raw landmark extraction stopped early at 8,862 clips | 0.44 GB, down from 1.1 GB planned | 5.3 GB | about 116 GB |
| 2026-11-03 02:30 | Raw landmarks for 8,161 unused test clips in `raw_fresh` | 376M | 5.7 GB before the deletion below | |
| 2026-11-03 02:30 | Deleted `fresh_test_videos` (after checking every clip's landmarks exist) and the partial `fs/` sets | -5.0 GB | about 1.0 GB (raw 0.45, raw_fresh 0.38, experiments about 0.2) | 118Gi |
| 2026-11-03 06:36 | End of run. Kept: raw landmarks, experiments, final models, browser model, notebook, result files | 1.35 GB in total | **Peak about 6.0 GB, within the 10 GB budget** | 115 GB |
