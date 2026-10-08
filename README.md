# RT-opto-model

A **CNN-GRU video classifier** that labels every frame of a grayscale behavior video with one of *K* behavior "clusters". It is designed to run frame-by-frame, carrying memory forward in time, so that a trained model can be used for **real-time (closed-loop) decisions**, such as triggering optogenetic stimulation when a behavior of interest is detected.

---

## Table of contents

1. [How it works (the 60-second version)](#how-it-works)
2. [What's in each file](#whats-in-each-file)
3. [Installation](#installation)
4. [Expected data format](#expected-data-format)
5. [Walkthrough: running the code](#walkthrough-running-the-code)
6. [Command-line reference](#command-line-reference)
7. [Running on new data: what to change](#running-on-new-data-what-to-change)
8. [Outputs](#outputs)

---

## How it works

```
 labels.pkl ─────────────┐                             ┌──► best_model.pt
 {session: cluster_ids}  │                             │
                         ▼                             │
 videos/…/Camera0/*.mp4 ─► decode once ─► uint8 memmap ─► TBPTT training ─► evaluation
   (e.g. 120 fps)          grayscale,       cache on      CNN → GRU → head   confusion matrix,
                           downscale,       disk          (frame-by-frame)   per-class acc,
                           keep every                                        latency benchmark
                           4th frame
```

1. **Decode once.** Each session's video is read sequentially, converted to grayscale, spatially downscaled, temporally subsampled (e.g. 120 → 30 fps), and written to a `uint8` memory-mapped `.npy` cache. Later runs reuse the cache, so startup is near-instant.
2. **Split by animal.** Sessions are split into train/validation at the *animal* level, so an animal never appears in both sets.
3. **Model.** Each frame goes through a small ResNet-style CNN (InstanceNorm, no batch statistics) to give a feature vector. A multi-layer GRU consumes the sequence of feature vectors, and a linear head outputs class logits **for every frame**.
4. **Train with TBPTT.** Sessions are processed in consecutive chunks (default 30 frames), and the GRU hidden state is carried across chunk boundaries with `.detach()`. This matches deployment, where the state persists for a whole session.
5. **Evaluate.** The best checkpoint (lowest validation loss) is run over the validation sessions as continuous streams. The code reports a classification report, confusion matrix, per-class accuracy, optional in-group vs. out-of-group (binarized) metrics, and a per-frame latency benchmark.

---

## What's in each file

| File | Role | What's inside |
|---|---|---|
| **`config.py`** | Central settings | A `Config` dataclass holding all hyperparameters and paths (video fps, downscale factor, CNN/GRU sizes, learning rate, early-stopping patience, and so on). **Edit defaults here.** |
| **`dataset.py`** | Data pipeline | Finds each session's video, decodes it to a cached uint8 memmap (in parallel), builds the chunk index, computes class weights, and splits train/val by animal. Also provides `preprocess_frame()` for single-frame inference. |
| **`model.py`** | Architecture | `ResBlock`, `CNNEncoder`, and `VideoClassifier` (CNN encoder → GRU → linear head). |
| **`train.py`** | Training loop | TBPTT training and validation, AdamW with a plateau LR scheduler, weighted cross-entropy with label smoothing, mixed precision, gradient accumulation, early stopping, best-checkpoint saving, `history.json` logging, and optional multi-GPU DDP. |
| **`evaluate.py`** | Post-training analysis | `plot_training_curves`, `full_evaluation` (report, confusion matrix, per-class accuracy), `binarized_evaluation` (in-group vs. out-of-group), and `benchmark_latency` (real-time streaming speed). |
| **`run.py`** | **Main entry point** | Command-line interface. It chains *train → plot curves → evaluate → latency benchmark*, or runs just part of that (`--eval_only`, `--plot_only`). |
| **`run.sh`** | Cluster launcher | A SLURM `sbatch` wrapper around `run.py` with three modes (`full`, `eval_only`, `plot_only`). It contains site-specific settings (partition, modules, conda env) that you will need to change. |
| **`live_monitor.py`** | Training dashboard | Run in a second terminal during training. It watches `history.json` and redraws loss and accuracy curves every few seconds. |

### Details worth knowing

- **`model.py`**
  - The CNN: 7×7 stride-2 conv → max-pool → one `ResBlock` per entry in `cnn_channels` → adaptive average pool to 2×2 → flatten → dropout. The feature size is `cnn_channels[-1] × 4`.
  - Input shape is `(B, T, 1, H, W)` and output logits are `(B, T, num_classes)`.
  - `InstanceNorm` is used instead of `BatchNorm` so behavior is identical at train and eval time, and robust to session-to-session brightness shifts.
- **`dataset.py`**
  - Frame cache files are named by an MD5 of *(video path, spatial scale, stride)* and live in `<output_dir>/frame_cache/`.
  - `SessionChunkDataset.get_session_chunk(sess, i)` returns the *i*-th non-overlapping chunk of a session. This is what TBPTT uses.
- **`train.py`**
  - `B = batch_size // grad_accum_steps` sessions are processed **in parallel as independent streams**, each with its own GRU state.
  - Loss uses inverse-frequency class weights and `label_smoothing=0.1`.
  - Every epoch it writes `history.json`, and it saves the checkpoint whenever validation loss improves.

---

## Installation

Required:

- Python 3.10+
- PyTorch 2.3+ (a CUDA GPU is strongly recommended)
- numpy
- opencv-python
- scikit-learn
- matplotlib
- tqdm
- Tk (only for `live_monitor.py`)

```bash
pip install torch numpy opencv-python scikit-learn matplotlib tqdm
```

---

## Expected data format

### Labels file (`labels.pkl`)

The labels file is a pickled Python **dictionary**, not a table, so there are no columns. Think of it as one row per session:

| Dictionary part | What it is |
|---|---|
| **Key** (one per session) | The session name, a string such as `CSDS-Day1-A_1-Defeat` |
| **Value** | A 1-D NumPy integer array. Entry *t* is the class ID of video frame *t*, so its length equals the number of frames in that session's video (at the native frame rate). |

Class IDs must be integers `0 … K−1` with no gaps and no `-1` or NaN values. The number of classes *K* is inferred as `max(label) + 1`.

### Session naming convention

```
ExperimentName-Day#-ID_#-Condition
```

| Field | Example | Notes |
|---|---|---|
| Experiment name | `CSDS` | Free text |
| Day | `Day1`, `Day5` | Free text |
| Animal ID | `A_1`, `A_5` | **Used to split train/val by animal.** Must be the third dash-separated field and must not itself contain a `-` (use `_`). |
| Condition | `Defeat`, `Control` | Free text |

The video folder for each session must have exactly this same name (see [Running on new data](#running-on-new-data-what-to-change) if your layout differs).

### Minimal example

```python
import pickle
import numpy as np

labels = {
    "CSDS-Day1-A_1-Defeat":  np.array([0, 0, 0, 3, 3, 1, 1, 0, ...], dtype=np.int64),
    "CSDS-Day5-A_1-Defeat":  np.array([0, 2, 2, 2, 0, 0, 0, 0, ...], dtype=np.int64),
    "CSDS-Day1-A_2-Control": np.array([0, 0, 0, 0, 0, 1, 0, 0, ...], dtype=np.int64),
}

with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)
```

---

## Walkthrough: running the code

### Step 0: Put your data in place

Have `labels.pkl` and the `data/` video tree ready (see above).

### Step 1: Smoke test (a few minutes)

Run a tiny job to confirm everything is wired up. It decodes the videos (the slow part, done once) and trains for 2 epochs:

```bash
python run.py --labels labels.pkl --video_root data/ --output_dir output_test/ --max_epochs 2
```

You should see, in order: per-session fps and stride, parallel video decoding, the train/val animal lists, the class count and class weights, the parameter count, and then epoch lines like:

```
Epoch   1/2 | train loss 1.2345 acc 0.6100 | val loss 1.1000 acc 0.6500 | lr 1.00e-03 | 95.2s
  Saved best model (val_loss=1.1000)
```

### Step 2: Full run (train → evaluate → benchmark)

```bash
python run.py --labels labels.pkl --video_root data/ --output_dir output/
```

Training runs for up to `max_epochs` (1000) with early stopping (`patience` = 100 epochs without a better validation loss). Afterwards the script automatically produces evaluation reports and the latency benchmark.

Common variations:

```bash
# Override hyperparameters
python run.py --labels labels.pkl --video_root data/ --lr 5e-4 --max_epochs 100 --patience 20

# Also report "in-group vs. out-of-group" metrics, e.g. clusters 0, 3 and 7 together
python run.py --labels labels.pkl --video_root data/ --binary_clusters 0,3,7
```

### Step 3 (optional): Watch training live

In a **second terminal** while training runs:

```bash
python live_monitor.py --output_dir output/ --interval 5
```

This opens a window with loss and accuracy curves that refresh every 5 s. It needs a display. If you train on a headless cluster, copy `history.json` to your laptop and run the monitor there, or use `--plot_only` (below).

### Step 4: Re-evaluate or re-plot without retraining

```bash
# Only redraw the training curves from an existing history.json
python run.py --plot_only --output_dir output/

# Skip training; evaluate + benchmark a saved model
python run.py --eval_only --labels labels.pkl --video_root data/ --output_dir output/
```

> **Important for `--eval_only`:** evaluation rebuilds the model from `config.py` and re-creates the validation split from `seed` and `val_fraction`. If you overrode `--gru_hidden`, `--dropout` or `--seed` during training, **pass the same values again**. Otherwise you get a `state_dict` shape mismatch or silently evaluate different sessions. (The saved checkpoint stores your training `config` under `ckpt["config"]` if you need to look up what was used.)

### Step 5 (cluster users): SLURM

`run.sh` wraps all of the above for a SLURM cluster:

```bash
sbatch run.sh                                           # full pipeline
sbatch --export=ALL,MODE=eval_only run.sh               # evaluate a saved model
sbatch --export=ALL,MODE=plot_only run.sh               # just redraw curves
LABELS=my_labels.pkl VIDEO_ROOT=/path/to/data BINARY_CLUSTERS=0,3,7 sbatch --export=ALL run.sh
```

**Edit these before first use**, because they are specific to the original cluster:

| Line | Default | Change to |
|---|---|---|
| `#SBATCH --partition=` | `witten,all` | your partition(s) |
| `#SBATCH --gpus / -c / --mem / --time` | 1 GPU, 32 CPUs, 128 GB, 72 h | what your job needs |
| `module load` | `anacondapy/2023.07-cuda` | your site's module (or remove) |
| `conda activate` | `general` | your environment name |
| `LABELS` default | `supervised_attack_classifications.pkl` | your labels file |
| `VIDEO_ROOT` default | `../../Behavior/Data/Defeat-Cohorts` | your video root |

Environment variables accepted by `run.sh`: `MODE`, `OUTPUT_DIR` (default `outputs/`), `MODEL_PATH`, `LABELS`, `VIDEO_ROOT`, `BINARY_CLUSTERS`, `SEED`, `GRU_HIDDEN`, `DROPOUT`, `BATCH_SIZE` (default 64), `NUM_WORKERS`. Logs go to `logs/out-<jobid>.txt` and `logs/error-<jobid>.txt`.

---

## Command-line reference

`python run.py [options]`

| Flag | Type / default | Meaning |
|---|---|---|
| `--labels` | path | Labels pickle (overrides `Config.labels_pkl`). |
| `--video_root` | path | Root folder searched for session directories. |
| `--output_dir` | `output/` | Where the model, plots, logs and frame cache go. |
| `--model_save_path` | `<output_dir>/best_model.pt` | Checkpoint path (to save, or to load with `--eval_only`). |
| `--lr` | `1e-3` | Learning rate. |
| `--batch_size` | `16` | Total sequences per optimizer step: `batch_size = parallel_streams × grad_accum_steps`. Must be ≥ `grad_accum_steps`. |
| `--grad_accum_steps` | `2` | Gradient-accumulation steps (the `--help` text says 4, but the config default is 2). |
| `--max_epochs` | `1000` | Maximum epochs. |
| `--patience` | `100` | Early-stopping patience (epochs). |
| `--gru_hidden` | `256` | GRU hidden size. |
| `--dropout` | `0.3` | Dropout in the CNN head and between GRU layers. |
| `--seed` | `42` | Seed for the train/val split and epoch shuffling. |
| `--num_workers` | `0` | DataLoader workers (not used by the current training loop). |
| `--no_amp` | off | Disable mixed-precision training. |
| `--binary_clusters` | e.g. `0,3,7` | Adds binarized in-group vs. out-of-group evaluation. |
| `--eval_only` | off | Skip training; evaluate and benchmark the saved model. |
| `--plot_only` | off | Only regenerate training curves from `history.json`. |

---

## Running on new data: what to change

### A. Things you **must** set (command line)

| What | Parameter |
|---|---|
| Your labels pickle | `--labels path/to/labels.pkl` |
| Your video root folder | `--video_root path/to/videos` |
| Where to save results | `--output_dir path/to/output` |

### B. Things you **should check** in `config.py`

These are **not** exposed on the command line, so edit them in `config.py`.

| Parameter | Default | Change it when… |
|---|---|---|
| `fps` | `120` | Your camera frame rate differs. It is only used to compute the real-time latency budget (`1000/fps` ms). |
| `target_fps` | `30` | You want a different processed frame rate. The stride is computed from each video's real fps. Lower is faster and gives more temporal context per chunk; higher is finer-grained. |
| `spatial_scale` | `0.35` | Your frames are larger or smaller, or fine detail matters. Output size is `int(H·scale) × int(W·scale)`. Raise it for small or detailed subjects; lower it to save memory and time. |
| `chunk_len` | `30` | You want longer or shorter training chunks (in processed frames). Sessions shorter than one chunk contribute nothing. |
| `val_fraction` | `0.10` | You have few animals. At least 1 animal is always held out; with N animals, `int(N × val_fraction)` are held out (min 1). |
| `cnn_channels` | `[16,32,64,128]` | You want a bigger or smaller encoder. |
| `gru_hidden`, `gru_layers` | `256`, `4` | You want a bigger or smaller temporal model (only `gru_hidden` has a CLI flag). Defaults are heavy; reduce them for small datasets or tight latency. |
| `weight_decay` | `1e-4` | Overfitting. |

Remember that **evaluation must use the same `cnn_channels`, `gru_hidden`, `gru_layers`, `dropout`, `spatial_scale`, `target_fps`, `seed` and `val_fraction` as training.**

### C. Things you must change in the **code** if your data differs

| Your situation | Where to edit |
|---|---|
| Session names are not `ExperimentName-Day#-ID_#-Condition` | `_animal()` inside `split_sessions()` in `dataset.py` (or rename sessions). |
| Videos aren't in `<session>/Camera0/*.mp4` | `find_session_video()` in `dataset.py`. The folder name `"Camera0"` and the excluded `"tracked_video.mp4"` are hard-coded there. |
| You want a different train/val strategy (e.g. fixed held-out sessions) | `split_sessions()` in `dataset.py`. |
| Videos have different resolutions | Resize/crop them beforehand. The pipeline requires a single resolution. |
| You want to label a different thing (e.g. fewer classes) | Re-map the arrays in `labels.pkl` to `0…K−1` before running. Nothing else changes. |

---

## Outputs

Everything lands in `--output_dir`:

| File | Produced by | Contents |
|---|---|---|
| `best_model.pt` | training | Best checkpoint (lowest val loss): weights, optimizer state, epoch, `num_classes`, and `config`. |
| `history.json` | training | Per-epoch train/val loss and accuracy, learning rate, and epoch time. |
| `training_curves.png` | `plot_training_curves` | Loss, accuracy, LR schedule, and epoch duration. |
| `classification_report.txt` | `full_evaluation` | Per-class precision, recall, and F1 on the validation set. |
| `confusion_matrix.png` | `full_evaluation` | Row-normalized (%) confusion matrix. |
| `per_class_accuracy.png` | `full_evaluation` | Bar chart of accuracy per cluster. |
| `binary_eval_clusters_<ids>.json/.txt` | `--binary_clusters` | TP/TN/FP/FN, accuracy, precision, recall, specificity, F1, AUROC. |
| `binary_confusion_clusters_<ids>.png` | `--binary_clusters` | 2×2 confusion matrix. |
| `latency_stats.json` | `benchmark_latency` | Mean, median, p95, p99, min, max, std per-frame latency, and max FPS. |
| `latency_histogram.png`, `latency_timeline.png` | `benchmark_latency` | Latency distribution and latency over time vs. the frame budget. |
| `frame_cache/*.npy` | `dataset.py` | Cached uint8 decoded videos (can be large; safe to delete, it will be rebuilt). |
