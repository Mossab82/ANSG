# ANSG Kaggle automation — smoke phase

This is a **manual-dispatch, no-training** connectivity test to verify GitHub Actions → Kaggle CLI → private Kaggle GPU run.

- Notebook owned by user: `mossabibrahim/notebook6367edd95f`; **not modified**.
- New smoke kernel: `mossabibrahim/ansg-api-smoke`; private, T4 requested, internet disabled.
- Python script writes `/kaggle/working/ansg_smoke_report.json`.
- The workflow is triggered ONLY manually via **Actions → Kaggle API smoke test → Run workflow**.
- Set a repository Actions secret `KAGGLE_API_TOKEN` with a **Kaggle API token string**, never a pasted `kaggle.json` blob. The local `kaggle auth login` browser session is not transferable to GitHub Actions.
- The initial workflow uploads the script and requests a run. Kaggle execution may finish **after** the GitHub workflow; artifact download/status are best-effort. The durable output remains available in Kaggle kernel versions.
- `kaggle kernels push` creates a new version and **starts** a Kaggle run. This uses Kaggle GPU quota; do not run repeatedly without need.
- Training scripts will be added only after smoke succeeds, with explicit dataset/checkpoint persistence and research provenance.
- The smoke output does not demonstrate any ANSG model training or scientific validity.

Official docs: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
Kaggle authentication: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md
