# Seeds Placeholder

Use this directory for small, version-controlled reference data only.

Minimum enterprise checklist for any future seed:

- declare column types in `seeds.yml`
- add grain uniqueness tests on business keys
- add `accepted_values` tests for coded domains
- tag contract-facing seeds with `contract_critical`
- set `meta.owner`, `meta.domain`, `meta.sla`, and `meta.tier`
