# Data

The experiments use the GAViST5G dataset (doi:10.1109/IEEEDATA.2025.3572868). The dataset is not included in this repository.

Download it and place these two files in this folder:

| File | Portion | Used for |
|---|---|---|
| `CombinedALLGeo.csv` | Canadian portion: 83 users, seven applications (YouTube, Netflix, Prime Video, Crave, League of Legends, Teamfight Tactics, Valorant) | main experiments and the ns-3 evaluation |
| `YouTubeCombinedGeo.csv` | International portion: 14 YouTube users | second-dataset experiment |

Each record aggregates one second of traffic between a user device and a remote endpoint. It includes the timestamp, IP addresses, protocol, bytes, average RTT, geolocation, and the endpoint's provider (autonomous system).

If your copies have other names or locations, change `DATA` and `GLOBAL` at the top of `python/run_all.bat`, `python/run_all.sh` or `python/run_all.ipynb`.
