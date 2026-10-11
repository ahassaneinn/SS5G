# SS5G: Deep Learning-Based Edge Server Selection in 5G

Code for predicting which edge server serves a 5G user next and for evaluating the predicted assignments, with capacity-aware load balancing, in a trace-driven ns-3 simulation. All experiments use the public GAViST5G dataset.

| Version | Content |
|---|---|
| v1.0 | Code of the ICMEW 2025 study (offline comparison of DL models) |
| v2.0 | This release: rebuilt pipeline, methods from the literature, and the ns-3 evaluation |

## Repository layout

```
data/      where to put the GAViST5G CSV files (not included)
python/    preprocessing, models, training, baselines, metrics, latency, reports
ns3/       allocation and load-balancing rule, ns-3 simulator, run and analysis scripts
```

## What it does

**Prediction (`python/`).** For every user and active second, the models predict the serving server of the *next* second (the server that carries the most bytes) from the previous 60 active seconds. The pipeline includes:

- seven DL models: DNN, LSTM, GRU, Transformer, Autoformer, TFT and FEDformer (`eua_pipeline.py`);
- four methods from the literature, adapted to the same task: DeepCog, Dong, DeepSlice and Chen (`sota_models.py`);
- four non-learning baselines: persistence, per-user majority, per-application majority and nearest server.

All models use the same inputs, a per-user chronological 70/15/15 split with a purge gap, and the same training protocol and seeds. Training-set statistics are used for normalization.

The pipeline reports the following metrics:

- top-1 and top-5 accuracy, macro- and weighted-F1;
- accuracy on server changes, recall by class frequency, and calibration;
- GPU and CPU inference latency;
- a test on unseen users and results on the international portion of the dataset.

**Network evaluation (`ns3/`).** The predicted assignments are replayed in an edge-server-level, trace-driven ns-3 simulation. Every allocator gets exactly the same traffic. The simulation compares four policies under several per-server capacities:

- direct assignment;
- capacity-aware load balancing (LB);
- LB with the model's top-5 candidates;
- LB without prediction.

It reports goodput, loss, mean and tail delay, jitter, a video stall proxy, and allocation metrics (override and switch rates, overload, load imbalance).

## Requirements

- **Python** 3.9 or newer: `pip install -r requirements.txt`. The models need TensorFlow ≥ 2.10. A GPU is recommended for training.
- **ns-3** 3.36 or newer on Linux, for the network evaluation (see `ns3/README.md`).

## Quick start

1. Download GAViST5G and place the two CSV files in `data/` (see `data/README.md`).
2. Run the Python pipeline from the `python` folder with **one** of the following:
   - Windows: `run_all.bat`
   - Linux/macOS: `./run_all.sh`
   - Jupyter: `run_all.ipynb`

   This writes `python/results/REPORT.md` and the ns-3 inputs in `ns3/ns3_inputs/`.
3. Copy the `ns3` folder to a Linux machine with ns-3 and follow `ns3/README.md`.

Individual steps can also be run on their own (`python eua_pipeline.py <command> -h`). The commands are `prepare`, `smoke`, `baselines`, `train`, `holdout`, `latency`, `tune` and `report`.

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

The citation of the journal article for v2.0 will be added on publication.
