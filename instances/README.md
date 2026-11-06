# Instances

Each instance is a single CSV file (`cbl_<id>.csv`) made of five blocks, one after the other:

1. **Size:** number of requests `k` and number of nodes `n`.
2. **Requests** (one row per request): origin node `oid`, destination node `did`, quantity `nk`, direct distance `d`, direct travel time `tt`, earliest pickup time `l` and latest delivery time `u` (minutes).
3. **Nodes:** latitude, longitude, member id, type (`O` origin, `D` destination, `H` hub) and node id.
4. **Distance matrix** between nodes.
5. **Travel-time matrix** between nodes (minutes).

> Instance files will be added here once it is confirmed which data can be shared publicly. Anonymized or synthetic instances will be provided if the original data cannot be published.
