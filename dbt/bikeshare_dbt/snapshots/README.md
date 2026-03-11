# Snapshots Placeholder

Use this directory for source-system history capture when upstream tables mutate in place.

Minimum enterprise checklist for any future snapshot:

- define the snapshot grain explicitly
- add tests for no overlapping validity ranges
- add tests for one current row per business key
- tag shared snapshots with `contract_critical`
- set `meta.owner`, `meta.domain`, `meta.sla`, and `meta.tier`
