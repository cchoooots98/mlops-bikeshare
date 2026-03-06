# DBT Quickstart

This repository includes a DBT project skeleton at:

- `dbt/bikeshare_dbt`

Use this profiles file:

- `dbt/profiles.yml`

Run commands from `dbt/bikeshare_dbt`:

```powershell
$env:DBT_PROFILES_DIR = ".."
dbt debug
dbt run --select staging
dbt test --select staging
```

