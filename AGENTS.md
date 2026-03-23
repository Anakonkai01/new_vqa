# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`. Use `src/train_h.py` and `src/evaluate_h.py` for Model H, `src/data/` for datasets, collate logic, and samplers, `src/models/` for model components, and `src/training/` for losses, curriculum, and SCST. Data assets are under `data/` (`annotations/`, `processed/`, `vg_features/`, `embeddings/`). Training outputs go to `checkpoints/`, `outputs/`, and `wandb/`. Utility and preprocessing scripts live in `src/scripts/`. Top-level shell launchers such as `train_model_h.sh` and `run_eval_h_comprehensive.sh` are the main entry points.

## Build, Test, and Development Commands
- `bash run_train_h_phase1.sh`: run Phase 1 only for Model H.
- `bash train_model_h.sh`: run the full Model H curriculum, Phase 1 to 4 plus eval.
- `bash run_eval_h_comprehensive.sh --phase phase4 --ckpt best --beam 3 --batch_size 256 --use_fasttext`: full evaluation on saved checkpoints.
- `PYTHONPATH=src python src/evaluate_h.py --checkpoint checkpoints/h/model_h_phase1_best.pth --datasets vqa_e vqa_x aokvqa --use_fasttext`: targeted evaluation.
- `python -m py_compile src/train_h.py src/evaluate_h.py`: quick syntax smoke test for edited Python files.
- `bash -n train_model_h.sh run_train_h_phase1.sh`: shell syntax check before running long jobs.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, `snake_case` for functions/variables, and `PascalCase` for classes. Keep modules focused; put data logic in `src/data/`, training logic in `src/training/`, and model changes in `src/models/`. Prefer ASCII text. Match existing script naming: numbered preprocessors in `src/scripts/` and phase launchers like `run_train_h_phase1.sh`.

## Testing Guidelines
This repo is script-tested rather than `pytest`-driven. Before committing, run `py_compile` on touched Python files, `bash -n` on edited shell scripts, and at least one small eval or train smoke test with reduced `--max_samples` or smaller batch sizes. For training changes, report the exact checkpoint, dataset subset, and metric deltas.

## Commit & Pull Request Guidelines
Recent history uses terse messages like `fix` and `change`; prefer clearer imperative commits such as `train_h: stabilize phase1 recovery`. Keep PRs scoped. Include: what changed, why, commands run, and any metric impact. Link relevant issues if present. For UI changes under `webapp/`, attach screenshots. Do not commit large data, checkpoints, or `wandb/` artifacts.

## Configuration & Data Notes
Set `PYTHONPATH=src` before direct Python entry points. Use `WANDB_MODE=offline` when network access is unavailable. Treat `data/`, `checkpoints/`, and `wandb/` as environment-specific; verify paths locally before launching multi-hour training jobs.
