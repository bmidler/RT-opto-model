#!/bin/bash
#==============================================================================
# RT-opto CNN-GRU Video Classifier — SLURM submission script
#
# Usage:
#   sbatch run.sh                                  # full: train -> eval -> benchmark
#   sbatch --export=ALL,MODE=eval_only run.sh      # eval + benchmark on a saved model
#   sbatch --export=ALL,MODE=plot_only run.sh      # just redraw training curves
#
# Environment variables (all optional):
#   MODE             full | eval_only | plot_only       (default: full)
#   OUTPUT_DIR       run directory                      (default: outputs/)
#   MODEL_PATH       checkpoint to evaluate             (default: $OUTPUT_DIR/best_model.pt)
#   LABELS           labels pkl
#   VIDEO_ROOT       root dir containing sessions
#   BINARY_CLUSTERS  e.g. "0,3,7" — in-group vs. out-of-group evaluation
#   SEED, GRU_HIDDEN, DROPOUT   MUST match the training run — see note below
#   BATCH_SIZE, NUM_WORKERS     training only; evaluation ignores both
#
# NOTE: evaluate.py rebuilds VideoClassifier from Config (only num_classes is
# read from the checkpoint) and regenerates the validation split by calling
# split_sessions(). If you overrode --gru_hidden or --seed when training, pass
# the same values here, or you will hit a state_dict shape mismatch / silently
# evaluate on a different set of sessions.
#==============================================================================

#SBATCH --job-name=RT-opto-Classifier
#SBATCH --partition=witten,all
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH --mem=128GB
#SBATCH --time=72:00:00
#SBATCH --output=logs/out-%j.txt
#SBATCH --error=logs/error-%j.txt

mkdir -p logs

module load anacondapy/2023.07-cuda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate general

MODE=${MODE:-full}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/}
LABELS=${LABELS:-supervised_attack_classifications.pkl}
VIDEO_ROOT=${VIDEO_ROOT:-../../Behavior/Data/Defeat-Cohorts}
MODEL_PATH=${MODEL_PATH:-${OUTPUT_DIR%/}/best_model.pt}
HISTORY_PATH=${OUTPUT_DIR%/}/history.json

echo "============================================"
echo "Job $SLURM_JOB_ID  —  mode=$MODE"
echo "Host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Output dir: $OUTPUT_DIR"
date
echo "============================================"

# Print GPU info
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); \
           print(f'CUDA available: {torch.cuda.is_available()}'); \
           [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') \
            for i in range(torch.cuda.device_count())]"

#------------------------------------------------------------------------------
# Shared arguments
#------------------------------------------------------------------------------
COMMON_ARGS=(
    --labels     "$LABELS"
    --video_root "$VIDEO_ROOT"
    --output_dir "$OUTPUT_DIR"
)

# Reproducibility / architecture overrides — only forwarded when explicitly set,
# so that an unset variable falls through to the Config default.
[ -n "$SEED" ]       && COMMON_ARGS+=(--seed       "$SEED")
[ -n "$GRU_HIDDEN" ] && COMMON_ARGS+=(--gru_hidden "$GRU_HIDDEN")
[ -n "$DROPOUT" ]    && COMMON_ARGS+=(--dropout    "$DROPOUT")

EVAL_ARGS=()
[ -n "$BINARY_CLUSTERS" ] && EVAL_ARGS+=(--binary_clusters "$BINARY_CLUSTERS")

#------------------------------------------------------------------------------
# Dispatch
#------------------------------------------------------------------------------
case "$MODE" in

  plot_only)
      if [ ! -f "$HISTORY_PATH" ]; then
          echo "ERROR: no history.json at $HISTORY_PATH" >&2
          exit 1
      fi
      echo ""
      echo ">>> Regenerating training curves from $HISTORY_PATH ..."
      python3 run.py --plot_only --output_dir "$OUTPUT_DIR"
      ;;

  eval_only)
      if [ ! -f "$MODEL_PATH" ]; then
          echo "ERROR: no checkpoint found at $MODEL_PATH" >&2
          echo "       Set MODEL_PATH=... or OUTPUT_DIR=... to point at the trained model." >&2
          exit 1
      fi

      # run.py returns early from --plot_only and skips plotting entirely under
      # --eval_only, so redraw the curves in a separate pass if history exists.
      if [ -f "$HISTORY_PATH" ]; then
          echo ""
          echo ">>> Regenerating training curves from $HISTORY_PATH ..."
          python3 run.py --plot_only --output_dir "$OUTPUT_DIR"
      else
          echo ""
          echo ">>> No history.json in $OUTPUT_DIR — skipping training curves."
      fi

      echo ""
      echo ">>> Running Evaluation + Latency Benchmark on $MODEL_PATH ..."
      python3 run.py --eval_only \
          "${COMMON_ARGS[@]}" \
          --model_save_path "$MODEL_PATH" \
          "${EVAL_ARGS[@]}"
      ;;

  full)
      echo ""
      echo ">>> Running Full Pipeline (Train -> Evaluate -> Benchmark) ..."
      python3 run.py \
          "${COMMON_ARGS[@]}" \
          --model_save_path "$MODEL_PATH" \
          --batch_size  "${BATCH_SIZE:-64}" \
          --num_workers "${NUM_WORKERS:-4}" \
          "${EVAL_ARGS[@]}"
      ;;

  *)
      echo "ERROR: unknown MODE '$MODE' (expected: full | eval_only | plot_only)" >&2
      exit 1
      ;;
esac

STATUS=$?

echo ""
echo "Done (exit status $STATUS)."
date
exit $STATUS
