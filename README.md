# SS5G: Deep Learning-Based Edge Server Selection in 5G

Code for predicting which edge server serves a 5G user next and for evaluating the predicted assignments, with capacity-aware load balancing, in a trace-driven ns-3 simulation. All experiments use the public GAViST5G dataset.

| Version | Content |
|---|---|
| v1.0 | Code of the ICMEW 2025 study (offline comparison of DL models) |
| v2.0 | Rebuilt pipeline, baselines, and the ns-3 evaluation |
| v3.0 | This release: equal-budget hyperparameter tuning with the selected values, repeated simulation runs for every allocator, residual-aware load balancing, and measured server load and reassignment metrics |

## Repository layout

```
data/                  where to put the GAViST5G CSV files (not included)
python/                preprocessing, models, tuning, training, baselines, metrics, latency, reports
python/tuning/         selected hyperparameters and all 15 trials of every model
python/architectures/  layers, output shapes, parameters and hyperparameters of every model, written from the code
ns3/                   allocation and load-balancing rules, ns-3 simulator, run, analysis and table/figure scripts
```

## What it does

**Prediction (`python/`).** For every user and active second, the models predict the serving server of the *next* second (the server that carries the most bytes) from the previous 60 active seconds:

- seven DL models: DNN, LSTM, GRU, Transformer, and simplified encoder-only adaptations of Autoformer, TFT and FEDformer (`eua_pipeline.py`, `build_model`);
- four methods from the literature adapted to server prediction: DeepCog, Dong, DeepSlice and Chen (`sota_models.py`);
- four non-learning baselines: persistence, per-user majority, per-application majority and nearest server.

All models use the same inputs, a per-user chronological 70/15/15 split with a purge gap, the same training protocol and seeds, and the same tuning budget (Bayesian optimisation with KerasTuner, 15 trials). The selected values are in `python/tuning/`.

Metrics: top-1/top-5 accuracy, macro-/weighted-F1, accuracy on server changes, recall by class frequency, calibration, paired accuracy differences against persistence, GPU/CPU inference latency, unseen users and the international portion of the dataset.

**Network evaluation (`ns3/`).** The predicted assignments are replayed in an edge-server-level, trace-driven ns-3 simulation with identical traffic for every allocator. Every allocator runs with the same network seeds and the same random numbers in each run (for the learned models, run *r* uses training seed *r*−1; GRU also runs every training-seed × network-seed combination). Policies:

- direct assignment;
- capacity-aware load balancing (LB), LB with the model's top-5 candidates, and LB without prediction;
- residual-aware variants (LB-R, LB-only-R) that carry an estimate of each server's queued and draining traffic across epochs.

The simulator reports goodput, delivered traffic, loss, mean and tail delay, jitter and a video stall proxy, and measures the server utilisation variance, overloaded user-seconds, residual traffic, false admissions of the capacity check, and reassignment rates.

## Requirements

- **Python** 3.9 or newer: `pip install -r requirements.txt` (TensorFlow ≥ 2.10, KerasTuner). A GPU is recommended for tuning and training.
- **ns-3** 3.36 or newer on Linux, for the network evaluation (see `ns3/README.md`).

## Reproducing the results

1. Download GAViST5G and place the two CSV files in `data/` (see `data/README.md`).
2. Run the Python pipeline from the `python` folder: `run_all.bat` (Windows) or `./run_all.sh` (Linux/macOS).
   - To reuse the selected hyperparameters instead of re-running the search, set `USE_RELEASED_TUNING=1` first.
   - Output: `python/results/REPORT.md`, the architecture tables, and the ns-3 inputs in `ns3/ns3_inputs/`.
3. Copy the `ns3` folder to a Linux machine with ns-3 and run `./run_everything.sh` (see `ns3/README.md`).
4. Build the tables and figures of the ns-3 evaluation: `python3 make_paper_results.py` in `ns3/`.

Individual steps can also be run on their own: `python eua_pipeline.py <command> -h` with `prepare`, `smoke`, `baselines`, `tune`, `train`, `holdout`, `latency`, `describe` or `report` (the same commands exist in `sota_models.py`).

## Citation

If you use this code, please cite the study that introduced the models and the dataset:

```bibtex
@inproceedings{HassaneinICMEW2025,
  author    = {Hassanein, Ahmed and Hashemi, Mahmoud Reza and Shirmohammadi, Shervin},
  title     = {{AI}-Assisted Edge Server Selection in {5G} for Low-Latency Multimedia Applications},
  booktitle = {2025 IEEE International Conference on Multimedia and Expo Workshops (ICMEW)},
  year      = {2025},
  url       = {https://ieeexplore.ieee.org/document/11152098}
}

@article{HassaneinGAViST5G2025,
  author  = {Hassanein, Ahmed and Hashemi, Mahmoud Reza and Shirmohammadi, Shervin},
  title   = {Descriptor: Gaming and Video Streaming Traffic for {5G} Dataset ({GAViST5G})},
  journal = {IEEE Data Descriptions},
  volume  = {2},
  pages   = {151--162},
  year    = {2025},
  doi     = {10.1109/IEEEDATA.2025.3572868}
}
```
