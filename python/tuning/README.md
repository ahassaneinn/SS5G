# Selected hyperparameters

One file per model, written by `eua_pipeline.py tune` / `sota_models.py tune` (KerasTuner Bayesian optimisation, 15 trials, at most 15 epochs per trial, early stopping with patience 3, objective: validation loss).

- `best`: the selected values used for every reported result (`--use-tuned`);
- `trials`: every trial with its values and validation loss;
- `budget`, `tuning_time_s`, `code_version`.

Search space: learning rate 1e-4 to 3e-3 (log), L2 in {0, 1e-5, 1e-4, 1e-3}, layer width in {32, 64, 128}, dropout 0 to 0.5 (models with a dropout layer), model width in {32, 64} (attention-based models). The methods from the literature keep their published layer sizes and tune the learning rate, L2 and, where they have one, dropout.

To reuse these values without re-running the search, copy the files to `results/tuning/` (or run `run_all` with `USE_RELEASED_TUNING=1`).
