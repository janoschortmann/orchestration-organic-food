# VTSP instances (cbl_1 ... cbl_100)

One CSV file per instance, `cbl_<id>.csv`, plus `instances.csv` (one row per instance:
size class, number of commodities, nodes, origins, destinations and hubs).

## File layout

1. `k,n,type`: number of commodities, number of nodes and size class
   (`s` = small, `m` = medium, `l` = large).
2. Commodity table: `oid, did, nk, d, tt, l, u` (origin node, destination node,
   quantity, direct distance and travel time origin -> destination, time window [l, u]).
3. Node table: `Member, Type, n_id` (Type: `O` origin, `D` destination, `H` hub).
4. Distance matrix (`id` row and column headers are node ids).
5. Travel-time matrix, in minutes.

## Notes

- Node and member ids are anonymous; a node keeps the same id in every instance.
  Locations have been removed.
- Distances and travel times are shortest-path values (metric closure), so both
  matrices satisfy the triangle inequality.
