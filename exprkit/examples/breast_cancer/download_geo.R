# Alternative to download_geo.sh using Bioconductor GEOquery.
# Run from the repository root:  Rscript examples/breast_cancer/download_geo.R
# Requires: install.packages("BiocManager"); BiocManager::install("GEOquery")
suppressPackageStartupMessages(library(GEOquery))
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
options(timeout = 3600)

# Series matrices (exprkit reads them directly; they are written as .txt.gz)
for (gse in c("GSE22820", "GSE96058", "GSE15852", "GSE42568", "GSE70947")) {
  getGEO(gse, GSEMatrix = TRUE, destdir = "data/raw", getGPL = FALSE)
}
# GSE96058 expression values are a supplementary file (about 592 MB)
getGEOSuppFiles("GSE96058", baseDir = "data/raw", makeDirectory = FALSE,
                filter_regex = "gene_expression_.*transformed")

# Platform tables: probe ID -> gene symbol
save_platform <- function(gpl, file) {
  g <- getGEO(gpl, destdir = "data/raw")
  write.table(Table(g), file, sep = "\t", quote = FALSE, row.names = FALSE)
}
save_platform("GPL6480", "data/raw/GPL6480-9577.txt")       # columns ID, GENE_SYMBOL
save_platform("GPL13607", "data/raw/GPL13607_table.txt")    # columns ID, GeneName
# GPL96 and GPL570: the configs use GEO's curated .annot.gz files (column "Gene symbol");
# download them with download_geo.sh, or point id_mapping at a table saved as above
# with symbol_column: "Gene Symbol".

writeLines(capture.output(sessionInfo()), "data/raw/R_sessionInfo.txt")
# The configs read series matrices uncompressed: run  gunzip -k data/raw/*_series_matrix.txt.gz
