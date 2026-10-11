# Data

The experiments use the GAViST5G dataset (doi:10.1109/IEEEDATA.2025.3572868). The dataset is not included in this repository.

Data pre-processing scripts are in the GAViST5G repository (https://github.com/ahassaneinn/GAViST5G).

Each record aggregates one second of traffic between a user device and a remote endpoint. It includes the timestamp, IP addresses, protocol, bytes, average RTT, geolocation, and the endpoint's provider (autonomous system).

If your copies have other names or locations, change `DATA` and `GLOBAL` at the top of `python/run_all.bat`, `python/run_all.sh` or `python/run_all.ipynb`.
