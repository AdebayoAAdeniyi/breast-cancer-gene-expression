# Scripts

| Script | Purpose |
|---|---|
| `make_tss_regions.sh` | Promoter regions (2,000 bp upstream to 200 bp downstream of each gene's transcription start site) from a GENCODE GTF, in BED format. Splits fields on tabs, uses the strand to find the start site, writes 0 based BED coordinates, and clamps starts at 0. Useful for mapping CpG methylation probes or other genomic intervals to genes. |

```bash
bash scripts/make_tss_regions.sh gencode.v47.annotation.gtf.gz > tss_regions.bed
```
