#!/usr/bin/env python
"""
sota_models.py -- four methods from the literature (DeepCog, Dong, DeepSlice, Chen),
adapted to edge server selection and evaluated with exactly the same pipeline as the
seven models in eua_pipeline.py.

Keep this file next to eua_pipeline.py. It reuses everything from eua_pipeline.py --
data preparation, next-second label, per-user chronological 70/15/15 split with purge,
training-period scaler/vocabularies, inputs, training protocol, metrics, prediction files,
latency and report -- and only adds four model architectures. Results go into the same
results/ folder, so `report` puts all models in the same tables.

  python sota_models.py smoke   --work work --out results            # 2-3 min check
  python sota_models.py train   --work work --out results --seeds 0 1 2 3 4
  python sota_models.py latency --work work --out results            # GPU
  (set CUDA_VISIBLE_DEVICES=-1 and run latency again for CPU)
  python eua_pipeline.py report --out results                         # all models together

Adaptation of each method (the same for all four):
  * Inputs: the same 60-step window as the seven models (numeric traffic/server history,
    user location, application, and embeddings of past server and provider).
  * Output: softmax over the server vocabulary of the training period; label = serving
    server of the next active second.
  * Training: identical for every model -- Adam (lr 1e-3), sparse cross-entropy, batch 256,
    up to 40 epochs, early stopping on validation loss (patience 5, best weights restored),
    learning rate halved on plateau, 5 seeds.
Architectures (as in the original papers):
  DeepCog   (Bega et al., INFOCOM 2019): 3D-CNN encoder 3x3x3 (32) -> 6x6x6 (32) -> dropout 0.3
            -> 6x6x6 (16) -> dropout 0.3, MLP decoder 128-64-32 (ReLU). The per-step feature
            vector is linearly projected to a 3x4 grid so the window forms a 60x3x4 volume.
  Dong      (Dong et al., 2019): DNN, 4 hidden layers of 100 ReLU units on the flattened window.
  DeepSlice (Thantharate et al., UEMCON 2019): DLNN 32 (ReLU) - 32 (ReLU) - 12 (tanh) on the
            flattened window.
  Chen      (Chen et al., 2025): ResNet-style 1D CNN -- stem conv 64/7 stride 2 + BN + max-pool,
            residual stages 64-64, 128-128 (stride 2), 256-256 (stride 2), global average
            pooling, dense 256 (ReLU) + dropout 0.3.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eua_pipeline as E  # noqa: E402

SOTA = ["DeepCog", "Dong", "DeepSlice", "Chen"]
REFERENCE = {
    "DeepCog": "Bega et al., DeepCog, IEEE INFOCOM 2019",
    "Dong": "Dong et al., Deep learning for hybrid 5G services in MEC, 2019",
    "DeepSlice": "Thantharate et al., DeepSlice, IEEE UEMCON 2019",
    "Chen": "Chen et al., Joint client-server selection ... split federated learning, 2025",
}

_build_original = E.build_model


def build_model(name, W, F, use_srv, n_srv, n_org, n_app, n_classes, cfg=None):
    if name not in SOTA:
        return _build_original(name, W, F, use_srv, n_srv, n_org, n_app, n_classes, cfg)
    import tensorflow as tf
    L = tf.keras.layers
    cfg = {**E.DEFAULT_CFG, **(cfg or {})}

    # ---- identical input front-end to the seven models (eua_pipeline.build_model)
    num = L.Input((W, F), name="num")
    app = L.Input((W,), dtype="int32", name="app")
    ins = [num, app]
    parts = [num, L.Embedding(n_app, cfg["app_emb"], name="emb_app")(app)]
    if use_srv:
        srv = L.Input((W,), dtype="int32", name="srv")
        org = L.Input((W,), dtype="int32", name="org")
        ins += [srv, org]
        parts += [L.Embedding(n_srv, cfg["srv_emb"], name="emb_srv")(srv),
                  L.Embedding(n_org, cfg["org_emb"], name="emb_org")(org)]
    x = L.Concatenate(axis=-1)(parts)

    if name == "DeepCog":
        h = L.Dense(12, name="grid_proj")(x)                      # (W, 12) -> 3x4 grid per step
        h = L.Reshape((W, 3, 4, 1))(h)
        h = L.Conv3D(32, (3, 3, 3), padding="same", activation="relu")(h)
        h = L.Conv3D(32, (6, 6, 6), padding="same", activation="relu")(h)
        h = L.Dropout(0.3)(h)
        h = L.Conv3D(16, (6, 6, 6), padding="same", activation="relu")(h)
        h = L.Dropout(0.3)(h)
        h = L.Flatten()(h)
        h = L.Dense(128, activation="relu")(h)
        h = L.Dense(64, activation="relu")(h)
        h = L.Dense(32, activation="relu")(h)
    elif name == "Dong":
        h = L.Flatten()(x)
        for _ in range(4):
            h = L.Dense(100, activation="relu")(h)
    elif name == "DeepSlice":
        h = L.Flatten()(x)
        h = L.Dense(32, activation="relu")(h)
        h = L.Dense(32, activation="relu")(h)
        h = L.Dense(12, activation="tanh")(h)
    else:  # Chen: ResNet-style 1D CNN
        def res_block(z, filters, stride=1):
            short = z
            y = L.Conv1D(filters, 3, strides=stride, padding="same", use_bias=False)(z)
            y = L.BatchNormalization()(y)
            y = L.ReLU()(y)
            y = L.Conv1D(filters, 3, strides=1, padding="same", use_bias=False)(y)
            y = L.BatchNormalization()(y)
            if short.shape[-1] != filters or stride != 1:
                short = L.Conv1D(filters, 1, strides=stride, padding="same", use_bias=False)(short)
                short = L.BatchNormalization()(short)
            return L.ReLU()(L.Add()([y, short]))

        h = L.Conv1D(64, 7, strides=2, padding="same", use_bias=False)(x)
        h = L.BatchNormalization()(h)
        h = L.ReLU()(h)
        h = L.MaxPooling1D(pool_size=3, strides=2, padding="same")(h)
        h = res_block(h, 64)
        h = res_block(h, 64)
        h = res_block(h, 128, stride=2)
        h = res_block(h, 128)
        h = res_block(h, 256, stride=2)
        h = res_block(h, 256)
        h = L.GlobalAveragePooling1D()(h)
        h = L.Dense(256, activation="relu")(h)
        h = L.Dropout(0.3)(h)

    out = L.Dense(n_classes, activation="softmax", name="out")(h)
    model = tf.keras.Model(ins, out, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    names = ["num", "app", "srv", "org"] if use_srv else ["num", "app"]
    return model, names


_latency_original = E.cmd_latency


def cmd_latency(a):
    """Same measurement as eua_pipeline, but merges into latency_<device>.csv instead of
    overwriting the rows of the seven models."""
    tf = E.setup_tf(0)
    dev = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    path = os.path.join(a.out, f"latency_{dev}.csv")
    old = pd.read_csv(path) if os.path.exists(path) else None
    _latency_original(a)
    if old is not None and os.path.exists(path):
        new = pd.read_csv(path)
        pd.concat([old[~old.model.isin(new.model)], new], ignore_index=True).to_csv(path, index=False)
        E.log("merged with the existing rows in", path)


# route every sub-command of eua_pipeline through the four literature models
E.build_model = build_model
E.cmd_latency = cmd_latency
E.ALL_MODELS = list(SOTA)          # "--models all" / smoke / latency = the four literature methods

if __name__ == "__main__":
    E.main()
