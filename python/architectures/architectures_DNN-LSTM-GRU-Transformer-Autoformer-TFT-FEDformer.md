# Model architectures (SS5G v3.0 (JCC revision 2))

TensorFlow 2.10.1, Python 3.9.22, window W=60, 14 numeric features, 712 output classes.

## DNN

Mechanism: three fully connected layers on the last time step

Hyperparameters: units=128, dense=32, d_model=32, ff=128, heads=4, dropout=0.2, l2=1e-05, lr=0.0025448623637421616, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5

Parameters: 50,300

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
| last_step | Lambda | (None, 38) | 0 |
| dense | Dense | (None, 128) | 4,992 |
| dense_1 | Dense | (None, 64) | 8,256 |
| dense_2 | Dense | (None, 32) | 2,080 |
| out | Dense | (None, 712) | 23,496 |

## LSTM

Mechanism: one LSTM layer over the window

Hyperparameters: units=128, dense=32, d_model=32, ff=128, heads=4, dropout=0.0, l2=0.0, lr=0.0013178214971561574, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5

Parameters: 124,604

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
| lstm | LSTM | (None, 128) | 85,504 |
| dropout | Dropout | (None, 128) | 0 |
| dense | Dense | (None, 32) | 4,128 |
| out | Dense | (None, 712) | 23,496 |

## GRU

Mechanism: one GRU layer over the window

Hyperparameters: units=32, dense=32, d_model=32, ff=128, heads=4, dropout=0.0, l2=0.0, lr=0.003, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5

Parameters: 42,940

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
| gru | GRU | (None, 32) | 6,912 |
| dropout | Dropout | (None, 32) | 0 |
| dense | Dense | (None, 32) | 1,056 |
| out | Dense | (None, 712) | 23,496 |

## Transformer

Mechanism: encoder: multi-head self-attention + feed-forward, sinusoidal positional encoding

Hyperparameters: units=64, dense=32, d_model=32, ff=128, heads=4, dropout=0.0, l2=1e-05, lr=0.001489940846672465, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5

Parameters: 65,820

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
| proj | Dense | (None, 60, 32) | 1,248 |
| positional_encoding | PositionalEncoding | (None, 60, 32) | 0 |
| enc0 | EncoderBlock | (None, 60, 32) | 12,704 |
| enc1 | EncoderBlock | (None, 60, 32) | 12,704 |
| global_average_pooling1d | GlobalAveragePooling1D | (None, 32) | 0 |
| dense_4 | Dense | (None, 64) | 2,112 |
| dropout_4 | Dropout | (None, 64) | 0 |
| dense_5 | Dense | (None, 32) | 2,080 |
| out | Dense | (None, 712) | 23,496 |

## Autoformer

Mechanism: encoder with series decomposition (moving average) into trend and seasonal parts; multi-head self-attention on the seasonal part

Hyperparameters: units=32, dense=32, d_model=32, ff=128, heads=4, dropout=0.0, l2=0.0, lr=0.00034393193073397867, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5

Parameters: 76,444

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
| proj | Dense | (None, 60, 32) | 1,248 |
| enc0 | EncoderBlock | (None, 60, 32) | 12,704 |
| enc1 | EncoderBlock | (None, 60, 32) | 12,704 |
| enc2 | EncoderBlock | (None, 60, 32) | 12,704 |
| global_average_pooling1d | GlobalAveragePooling1D | (None, 32) | 0 |
| dense_6 | Dense | (None, 32) | 1,056 |
| dropout_6 | Dropout | (None, 32) | 0 |
| dense_7 | Dense | (None, 32) | 1,056 |
| out | Dense | (None, 712) | 23,496 |

## TFT

Mechanism: encoder with sigmoid input gating before multi-head self-attention, sinusoidal positional encoding

Hyperparameters: units=32, dense=32, d_model=64, ff=128, heads=4, dropout=0.0, l2=1e-05, lr=0.00109257807060034, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5

Parameters: 153,500

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
| proj | Dense | (None, 60, 64) | 2,496 |
| positional_encoding | PositionalEncoding | (None, 60, 64) | 0 |
| enc0 | EncoderBlock | (None, 60, 64) | 37,632 |
| enc1 | EncoderBlock | (None, 60, 64) | 37,632 |
| enc2 | EncoderBlock | (None, 60, 64) | 37,632 |
| global_average_pooling1d | GlobalAveragePooling1D | (None, 64) | 0 |
| dense_9 | Dense | (None, 32) | 2,080 |
| dropout_6 | Dropout | (None, 32) | 0 |
| dense_10 | Dense | (None, 32) | 1,056 |
| out | Dense | (None, 712) | 23,496 |

## FEDformer

Mechanism: encoder with series decomposition (moving average); attention on the Fourier representation (FFT over time) of the seasonal part

Hyperparameters: units=128, dense=32, d_model=64, ff=128, heads=4, dropout=0.2, l2=1e-05, lr=0.0014599526426044272, srv_emb=16, org_emb=4, app_emb=4, ma_kernel=5

Parameters: 150,332

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
| proj | Dense | (None, 60, 64) | 2,496 |
| enc0 | EncoderBlock | (None, 60, 64) | 33,472 |
| enc1 | EncoderBlock | (None, 60, 64) | 33,472 |
| enc2 | EncoderBlock | (None, 60, 64) | 33,472 |
| global_average_pooling1d | GlobalAveragePooling1D | (None, 64) | 0 |
| dense_6 | Dense | (None, 128) | 8,320 |
| dropout_6 | Dropout | (None, 128) | 0 |
| dense_7 | Dense | (None, 32) | 4,128 |
| out | Dense | (None, 712) | 23,496 |
