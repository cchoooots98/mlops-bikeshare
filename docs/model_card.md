# Model Card Notes

Generated model cards now belong inside each local model package under:

```text
artifacts/model_card.md
```

The generated card is derived from the training summary and kept next to:
- `package_manifest.json`
- `artifacts/eval_summary.json`
- the packaged MLflow model in `model/`

This keeps offline evaluation artifacts and deployable metadata in one package boundary.
