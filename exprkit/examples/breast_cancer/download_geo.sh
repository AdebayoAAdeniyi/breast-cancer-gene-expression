#!/usr/bin/env bash
# Download the public GEO files used by the breast cancer example into data/raw.
# Run from anywhere; files land in <repository>/data/raw.
# Paths follow the GEO FTP layout (series/GSEnnn/GSExxxxx/{matrix,suppl}).
# Before relying on the results, open each GEO page and check that the file
# names and sample counts still match:
#   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22820
#   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058
#
# The GPL6480 platform table is not downloaded here: on the GPL6480 page use
# "Download full table" and save it as data/raw/GPL6480-9577.txt (or change
# id_mapping.path in examples/breast_cancer/prepare_gse22820.yaml).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p data/raw
BASE=https://ftp.ncbi.nlm.nih.gov/geo/series
get() { [ -s "data/raw/$(basename "$1")" ] || curl -fL --retry 3 -o "data/raw/$(basename "$1")" "$1"; }

get "$BASE/GSE22nnn/GSE22820/matrix/GSE22820_series_matrix.txt.gz"
get "$BASE/GSE96nnn/GSE96058/matrix/GSE96058-GPL11154_series_matrix.txt.gz"
get "$BASE/GSE96nnn/GSE96058/matrix/GSE96058-GPL18573_series_matrix.txt.gz"
get "$BASE/GSE96nnn/GSE96058/suppl/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz"

# Additional studies with both normal and tumour tissue (multi study example)
get "$BASE/GSE15nnn/GSE15852/matrix/GSE15852_series_matrix.txt.gz"
get "$BASE/GSE42nnn/GSE42568/matrix/GSE42568_series_matrix.txt.gz"
get "$BASE/GSE70nnn/GSE70947/matrix/GSE70947_series_matrix.txt.gz"
PL=https://ftp.ncbi.nlm.nih.gov/geo/platforms
[ -s data/raw/GPL96.annot.gz ]  || curl -fL --retry 3 -o data/raw/GPL96.annot.gz  "$PL/GPLnnn/GPL96/annot/GPL96.annot.gz"
[ -s data/raw/GPL570.annot.gz ] || curl -fL --retry 3 -o data/raw/GPL570.annot.gz "$PL/GPLnnn/GPL570/annot/GPL570.annot.gz"
# GPL13607 has no .annot file; extract the platform table from the family SOFT file (about 1.4 GB download)
if [ ! -s data/raw/GPL13607_table.txt ]; then
  curl -fL --retry 3 "$PL/GPL13nnn/GPL13607/soft/GPL13607_family.soft.gz" | gzip -dc \
    | sed -n '/^!platform_table_begin/,/^!platform_table_end/p' | grep -v '^!platform_table' > data/raw/GPL13607_table.txt
fi

# The example configs read the series matrices uncompressed (as GEO's web page serves them)
gunzip -f data/raw/*_series_matrix.txt.gz

# Record what was downloaded, so results can be tied to exact files.
sha256sum data/raw/* > data/raw/SHA256SUMS.txt
echo "Done. Checksums in data/raw/SHA256SUMS.txt"
