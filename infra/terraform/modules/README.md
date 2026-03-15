# Terraform Modules Layout

This directory contains the reusable Terraform modules for the single-account AWS layout.

Current entrypoints:
- `infra/terraform/bootstrap`
- `infra/terraform/live`

`bootstrap` provisions the remote-state backend primitives.
`live` provisions the long-lived platform stack by calling the modules in this directory.
