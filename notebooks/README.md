# Colab Notebooks

## `colab_publication_run.ipynb` — A/B/C publication run

The whole publication-grade ablation in one notebook. Runtime → Run All →
~3-4 hours on Colab Pro+ A100 (Research-tier) or ~7-8 hours on T4. Writes
results to Google Drive; you pull them locally and ask CRS to interpret.

### One-time setup (do this once before the first run)

On your local Mac, from the repo root:

```bash
# 1. Zip the dataset (~10 GB, slow first time)
tar -czf /tmp/camels_us.tar.gz datasets/camels_us

# 2. Zip the code (small, fast)
tar -czf /tmp/nh_code.tar.gz \
    --exclude='datasets' --exclude='runs' --exclude='.git' \
    --exclude='__pycache__' --exclude='*.pt' \
    .

# 3. Upload BOTH zips to Google Drive at:
#    My Drive/neural_hydrology_data/
#    (so the full path is My Drive/neural_hydrology_data/camels_us.tar.gz
#                       and My Drive/neural_hydrology_data/nh_code.tar.gz)
#
# Drag the files in the Drive web UI, or use rclone, or use the Drive
# desktop app's sync folder.
```

### Running the notebook

1. Upload `colab_publication_run.ipynb` to Google Drive.
2. Open it in Google Colab.
3. **Runtime → Change runtime type → GPU** (A100 if available on
   Pro+/Research, otherwise T4 is fine).
4. **Runtime → Run all**.
5. Walk away. Total wall-clock 3-8 hours depending on GPU.

The notebook is **idempotent** — every cell that runs an experiment skips
if the run already produced its checkpoint, so re-running after a session
disconnect just resumes from where it left off.

### Getting results back to the local Mac

Two options, pick the one that matches your setup.

**Option A — Google Drive desktop sync (cleanest):**

If you have Google Drive for desktop installed and synced on your Mac,
the result files at `My Drive/neural_hydrology_runs/...` automatically
appear in your `~/Google Drive/My Drive/neural_hydrology_runs/` folder.
Drag the `experiments/analysis_outputs/abc_publication/` folder from
there into your local repo at the same path, then in chat:

```
crs interpret abc results
```

**Option B — git push from Colab (universal):**

If you cloned via GitHub in Cell 2, uncomment the git-push block in the
final cell. It pushes only the small result files (CSVs + JSON, not
gigabyte checkpoints). Then on your local Mac:

```bash
cd /Users/om/Desktop/neural_hydrology
git pull
```

Then ask CRS to interpret.

### What to expect at the end

The notebook produces, in `experiments/analysis_outputs/abc_publication/`:

- `summary.json` — cross-seed per-condition median NSE (the headline
  numbers you'd want to see)
- `per_basin_per_seed.csv` — long-format table with one row per
  (condition, seed, basin) for downstream stats
- And on Google Drive at `My Drive/neural_hydrology_runs/`, the full
  trained checkpoints in case re-evaluation is needed

The notebook prints the **headline `C − A` and `C − B` medians** at the
end of Cell 11. That's the answer the publication run was set up to
produce.

### Compute estimate sanity check

| GPU | Per-run wall-clock | 15 runs total |
|---|---|---|
| Colab T4 (free / Pro) | ~30 min | ~7-8 hr |
| Colab V100 (Pro) | ~20 min | ~5 hr |
| Colab A100 (Pro+ / Research) | ~10-15 min | **~3-4 hr** |

You'll fit the whole publication run in **one** Colab Pro+ session
(24 hr limit). Pro (no plus) sessions are also enough for a full run.

### Known gotchas (already handled in the notebook)

- **numpy 2 vs torch 2.2 incompatibility**: notebook pins `numpy<2`.
- **Session disconnect**: skip-if-already-done logic in each loop cell.
- **GPU type variance on Pro (no plus)**: notebook checks GPU name + memory at start and proceeds regardless.
- **Drive mount permission**: standard Colab popup; allow it once.
