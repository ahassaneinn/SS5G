# Model architectures (SS5G v3.0 (JCC revision 2))

TensorFlow 2.10.1, Python 3.9.22, window W=60, 14 numeric features, 712 output classes.

## DeepCog

Mechanism: 3D-CNN encoder over the window projected to a 3x4 grid per step, MLP decoder

Hyperparameters: units=64, dense=32, d_model=32, ff=128, heads=4, dropout=0.2, l2=0.001, lr=0.0011094158660866346, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5, sota_l2=1e-05, sota_dropout=0.2

Parameters: 1,853,184

| Layer | Type | Output shape | Params |
|---|---|---|---|
| app | InputLayer | [(None, 60)] | 0 |
| srv | InputLayer | [(None, 60)] | 0 |
| org | InputLayer | [(None, 60)] | 0 |
| num | InputLayer | [(None, 60, 14)] | 0 |
| emb_app | Embedding | (None, 60, 4) | 36 |
| emb_srv | Embedding | (None, 60, 16) | 11,392 |
| emb_org | Embedding | (None, 60, 4) | 48 |
| concatenate | Concatenate | (None, 60, 38) | 0 |
| grid_proj | Dense | (None, 60, 12) | 468 |
| reshape | Reshape | (None, 60, 3, 4, 1) | 0 |
| conv3d | Conv3D | (None, 60, 3, 4, 32) | 896 |
| conv3d_1 | Conv3D | (None, 60, 3, 4, 32) | 221,216 |
| dropout | Dropout | (None, 60, 3, 4, 32) | 0 |
| conv3d_2 | Conv3D | (None, 60, 3, 4, 16) | 110,608 |
| dropout_1 | Dropout | (None, 60, 3, 4, 16) | 0 |
| flatten | Flatten | (None, 11520) | 0 |
| dense | Dense | (None, 128) | 1,474,688 |
| dense_1 | Dense | (None, 64) | 8,256 |
| dense_2 | Dense | (None, 32) | 2,080 |
| out | Dense | (None, 712) | 23,496 |

## Dong

Mechanism: four fully connected layers of 100 ReLU units on the flattened window

Hyperparameters: units=64, dense=32, d_model=32, ff=128, heads=4, dropout=0.2, l2=0.001, lr=0.0003087902110917711, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5, sota_l2=0.0

Parameters: 341,788

| Layer | Type | Output shape | Params |
|---|---|---|---|
| app | InputLayer | [(None, 60)] | 0 |
| srv | InputLayer | [(None, 60)] | 0 |
| org | InputLayer | [(None, 60)] | 0 |
| num | InputLayer | [(None, 60, 14)] | 0 |
| emb_app | Embedding | (None, 60, 4) | 36 |
| emb_srv | Embedding | (None, 60, 16) | 11,392 |
| emb_org | Embedding | (None, 60, 4) | 48 |
| concatenate | Concatenate | (None, 60, 38) | 0 |
| flatten | Flatten | (None, 2280) | 0 |
| dense | Dense | (None, 100) | 228,100 |
| dense_1 | Dense | (None, 100) | 10,100 |
| dense_2 | Dense | (None, 100) | 10,100 |
| dense_3 | Dense | (None, 100) | 10,100 |
| out | Dense | (None, 712) | 71,912 |

## DeepSlice

Mechanism: fully connected 32-32-12 network on the flattened window

Hyperparameters: units=64, dense=32, d_model=32, ff=128, heads=4, dropout=0.2, l2=0.001, lr=0.003, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5, sota_l2=0.0001

Parameters: 95,176

| Layer | Type | Output shape | Params |
|---|---|---|---|
| app | InputLayer | [(None, 60)] | 0 |
| srv | InputLayer | [(None, 60)] | 0 |
| org | InputLayer | [(None, 60)] | 0 |
| num | InputLayer | [(None, 60, 14)] | 0 |
| emb_app | Embedding | (None, 60, 4) | 36 |
| emb_srv | Embedding | (None, 60, 16) | 11,392 |
| emb_org | Embedding | (None, 60, 4) | 48 |
| concatenate | Concatenate | (None, 60, 38) | 0 |
| flatten | Flatten | (None, 2280) | 0 |
| dense | Dense | (None, 32) | 72,992 |
| dense_1 | Dense | (None, 32) | 1,056 |
| dense_2 | Dense | (None, 12) | 396 |
| out | Dense | (None, 712) | 9,256 |

## Chen

Mechanism: ResNet-style 1D CNN over the window

Hyperparameters: units=64, dense=32, d_model=32, ff=128, heads=4, dropout=0.2, l2=0.001, lr=0.000592264562254169, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5, sota_l2=0.0, sota_dropout=0.1

Parameters: 1,236,508

| Layer | Type | Output shape | Params |
|---|---|---|---|
| app | InputLayer | [(None, 60)] | 0 |
| srv | InputLayer | [(None, 60)] | 0 |
| org | InputLayer | [(None, 60)] | 0 |
| num | InputLayer | [(None, 60, 14)] | 0 |
| emb_app | Embedding | (None, 60, 4) | 36 |
| emb_srv | Embedding | (None, 60, 16) | 11,392 |
| emb_org | Embedding | (None, 60, 4) | 48 |
| concatenate | Concatenate | (None, 60, 38) | 0 |
| conv1d | Conv1D | (None, 30, 64) | 17,024 |
| batch_normalization | BatchNormalization | (None, 30, 64) | 256 |
| re_lu | ReLU | (None, 30, 64) | 0 |
| max_pooling1d | MaxPooling1D | (None, 15, 64) | 0 |
| conv1d_1 | Conv1D | (None, 15, 64) | 12,288 |
| batch_normalization_1 | BatchNormalization | (None, 15, 64) | 256 |
| re_lu_1 | ReLU | (None, 15, 64) | 0 |
| conv1d_2 | Conv1D | (None, 15, 64) | 12,288 |
| batch_normalization_2 | BatchNormalization | (None, 15, 64) | 256 |
| add | Add | (None, 15, 64) | 0 |
| re_lu_2 | ReLU | (None, 15, 64) | 0 |
| conv1d_3 | Conv1D | (None, 15, 64) | 12,288 |
| batch_normalization_3 | BatchNormalization | (None, 15, 64) | 256 |
| re_lu_3 | ReLU | (None, 15, 64) | 0 |
| conv1d_4 | Conv1D | (None, 15, 64) | 12,288 |
| batch_normalization_4 | BatchNormalization | (None, 15, 64) | 256 |
| add_1 | Add | (None, 15, 64) | 0 |
| re_lu_4 | ReLU | (None, 15, 64) | 0 |
| conv1d_5 | Conv1D | (None, 8, 128) | 24,576 |
| batch_normalization_5 | BatchNormalization | (None, 8, 128) | 512 |
| re_lu_5 | ReLU | (None, 8, 128) | 0 |
| conv1d_6 | Conv1D | (None, 8, 128) | 49,152 |
| conv1d_7 | Conv1D | (None, 8, 128) | 8,192 |
| batch_normalization_6 | BatchNormalization | (None, 8, 128) | 512 |
| batch_normalization_7 | BatchNormalization | (None, 8, 128) | 512 |
| add_2 | Add | (None, 8, 128) | 0 |
| re_lu_6 | ReLU | (None, 8, 128) | 0 |
| conv1d_8 | Conv1D | (None, 8, 128) | 49,152 |
| batch_normalization_8 | BatchNormalization | (None, 8, 128) | 512 |
| re_lu_7 | ReLU | (None, 8, 128) | 0 |
| conv1d_9 | Conv1D | (None, 8, 128) | 49,152 |
| batch_normalization_9 | BatchNormalization | (None, 8, 128) | 512 |
| add_3 | Add | (None, 8, 128) | 0 |
| re_lu_8 | ReLU | (None, 8, 128) | 0 |
| conv1d_10 | Conv1D | (None, 4, 256) | 98,304 |
| batch_normalization_10 | BatchNormalization | (None, 4, 256) | 1,024 |
| re_lu_9 | ReLU | (None, 4, 256) | 0 |
| conv1d_11 | Conv1D | (None, 4, 256) | 196,608 |
| conv1d_12 | Conv1D | (None, 4, 256) | 32,768 |
| batch_normalization_11 | BatchNormalization | (None, 4, 256) | 1,024 |
| batch_normalization_12 | BatchNormalization | (None, 4, 256) | 1,024 |
| add_4 | Add | (None, 4, 256) | 0 |
| re_lu_10 | ReLU | (None, 4, 256) | 0 |
| conv1d_13 | Conv1D | (None, 4, 256) | 196,608 |
| batch_normalization_13 | BatchNormalization | (None, 4, 256) | 1,024 |
| re_lu_11 | ReLU | (None, 4, 256) | 0 |
| conv1d_14 | Conv1D | (None, 4, 256) | 196,608 |
| batch_normalization_14 | BatchNormalization | (None, 4, 256) | 1,024 |
| add_5 | Add | (None, 4, 256) | 0 |
| re_lu_12 | ReLU | (None, 4, 256) | 0 |
| global_average_pooling1d | GlobalAveragePooling1D | (None, 256) | 0 |
| dense | Dense | (None, 256) | 65,792 |
| dropout | Dropout | (None, 256) | 0 |
| out | Dense | (None, 712) | 182,984 |
