# Examples

| Example | Data | Shows |
|---|---|---|
| [breast_cancer](breast_cancer) | Public GEO studies: GSE22820, GSE96058, GSE15852, GSE42568, GSE70947 | Preparing GEO series, tumour vs normal within and across studies, survival with clinical baselines, gene selection, PAM50 subtypes, batch checks |
| [user_tables](user_tables) | Ready-made tables (samples by genes with label or survival columns) | Reading such tables, finding copied rows and imputed values, recovering each row's source study, and the confounding test |

Every config is plain YAML and can be copied as a starting point. Run commands from the repository root; paths in configs are relative to it. Raw data go in `data/raw/` and your own tables in `data/user/`; neither is committed.
