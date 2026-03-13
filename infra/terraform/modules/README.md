# Terraform Modules Layout

This directory is the target home for extracted Terraform modules.

The current project still uses the flat root module in `infra/terraform/`.
To avoid breaking existing state and references, the root module remains the source of truth for now.

New environment entrypoints live under:
- `infra/terraform/envs/dev`
- `infra/terraform/envs/prod`

Those entrypoints wrap the current root module without moving the original files.
