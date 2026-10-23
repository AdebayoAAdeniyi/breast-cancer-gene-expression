# Contributing

Contributions are welcome: bug reports, new readers, new models, examples on other diseases, and corrections to the documentation.

## Development setup

```bash
git clone https://github.com/AdebayoAAdeniyi/breast-cancer-gene-expression.git
cd breast-cancer-gene-expression/exprkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,deep]"
pytest
```

## Ground rules

1. **Every new method needs a test with a known answer.** Use `exprkit.simulate` to plant an effect, or compare against a reference implementation, as the C index and Cox loss tests do. A method that cannot be checked is not added.
2. **Nothing may see test samples.** Any step that learns from data (filtering, selection, scaling, imputation, tuning) belongs inside the model pipeline, so it is refitted in every training fold.
3. **Report, do not hide.** When a step drops, maps, imputes or changes anything, the count goes into the report or summary JSON.
4. **Cite methods.** Add the reference for any statistical method to its docstring and to the user guide.
5. **Never commit data or results.** `data/` and `results/` are ignored by git. Example configs point to public sources and say how to download them.

## Pull requests

- Keep each pull request to one change, with tests, and update `CHANGELOG.md`.
- Run `pytest` locally before opening the pull request; GitHub Actions runs it again.
- For a new example, include its configs, a README with the data sources and expected counts, and the headline numbers with intervals.

## Reporting problems

Open an issue with the command you ran, the config (with paths shortened), the full error message, and the output of `pip show exprkit scikit-learn scikit-survival pandas`.
