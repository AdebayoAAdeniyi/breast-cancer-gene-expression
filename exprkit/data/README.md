# Data

Nothing in this folder except this README is committed (see `.gitignore`).

| Folder | Holds |
|---|---|
| `data/raw/` | Downloaded files: GEO series matrices, platform tables, supplementary expression files |
| `data/user/` | Your own tables |
| `data/processed/<name>/` | Prepared datasets written by `exprkit prepare`: `expression.parquet`, `samples.csv`, `report.json` |

The breast cancer example lists its files and download commands in [examples/breast_cancer/README.md](../examples/breast_cancer/README.md). `examples/breast_cancer/download_geo.sh` records SHA256 checksums in `data/raw/SHA256SUMS.txt`, so results can be tied to exact files.
