#!/usr/bin/env bash
# Promoter regions (2,000 bp upstream to 200 bp downstream of each gene's
# transcription start site) from a GENCODE GTF, in BED format.
#
# Not used by the expression models. Useful for mapping CpG methylation probes
# or other genomic intervals to genes through their promoters.
#
# Details that are easy to get wrong:
#   * splits fields on tabs (-F'\t'); the attribute column contains spaces, so
#     the default whitespace split left gene_id and gene_name empty
#   * strand aware: on the minus strand the TSS is the END coordinate ($5)
#   * converts GTF 1 based starts to BED 0 based half open intervals
#   * clamps region starts at 0
#
# Usage: bash scripts/reference/make_tss_regions.sh gencode.v47.annotation.gtf.gz > tss_regions.bed
set -euo pipefail
GTF="$1"
if [[ "$GTF" == *.gz ]]; then CAT="gzip -dc"; else CAT="cat"; fi
$CAT "$GTF" | awk -F'\t' -v UP=2000 -v DOWN=200 'BEGIN{OFS="\t"}
  $0 !~ /^#/ && $3 == "gene" {
    gid = ""; gname = ""
    if (match($9, /gene_id "[^"]+"/))   gid   = substr($9, RSTART + 9,  RLENGTH - 10)
    if (match($9, /gene_name "[^"]+"/)) gname = substr($9, RSTART + 11, RLENGTH - 12)
    if ($7 == "+") { tss = $4 - 1; s = tss - UP;   e = tss + DOWN + 1 }
    else           { tss = $5 - 1; s = tss - DOWN; e = tss + UP + 1 }
    if (s < 0) s = 0
    print $1, s, e, gid, gname, $7
  }'
