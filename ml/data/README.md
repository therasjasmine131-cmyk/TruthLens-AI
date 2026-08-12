# TruthLens AI - Dataset documentation

## ISOT Fake News Dataset (primary source)

- **Where:** https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf (and Kaggle mirrors)
- **Contents:** `True.csv` (~21.4k articles scraped from reuters.com) and `Fake.csv` (~23.5k articles from unreliable sources flagged by fact-checking organizations such as Politifact).
- **Columns:** `title`, `text`, `subject`, `date`
- **License / ethics:** Published for academic research; articles are copyright of their original publishers. This project uses it **only for educational, non-commercial research**. If you redistribute predictions or samples, always attribute the source.
- **Download:** Put `True.csv` and `Fake.csv` into `ml/data/raw/`. A documented helper is shown in `docs/ml-methodology.md`. The repository does **not** commit the raw CSVs because they total >110 MB.

## Bundled sample (fallback)

`sample.csv` is a stratified, random sample of the ISOT dataset (600 rows per
class) committed to the repository so the app trains out-of-the-box with no
external download. When the full raw CSVs are present they are used
automatically instead. Regenerate the sample with:

```bash
python -m ml.dataset   # not yet a script entrypoint; use the helper below
python -c "from ml.dataset import build_samples; build_samples(600)"
```

## Important caveats

- The dataset is **time-correlated** (2016-2017, US politics) and **domain-biased**
  (Reuters real news vs. partisan fake sources). Models trained on it do **not**
  generalize to all news; predictions are estimates, not proof.
- The loader performs no downloads silently — training always uses local files.
