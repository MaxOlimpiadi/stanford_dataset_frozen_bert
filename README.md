# Frozen BERT for IMDb Sentiment Classification

A research project investigating how training dataset size affects **sentiment classification** using **frozen BERT embeddings** and **logistic regression**.

**Method**

- **Encoder:** `bert-base-uncased`, used without fine-tuning.
- **Features:** 768-dimensional embeddings of the `[CLS]` token.
- **Classifier:** logistic regression with `max_iter=1000`.
- **Labels:** `0` = negative, `1` = positive.
- **Metrics:** Accuracy, Precision, Recall, and F1.

**Experiment Setup**

Training subsets contain **20, 40, 70, 100, 150, 200, 300, 500, 700, 900, and 1100** reviews, using seeds **7, 10, and 35**.

Subsets are balanced by class and nested within each seed. They are sampled from the official IMDb training partition.

The official test partition is divided into **10,000 validation reviews** and **10,000 test reviews** using stratified sampling. The remaining 5,000 reviews are unused. The generated validation set is not used by the current logistic regression pipeline.

**Project Structure**

```text
main.py              # Embedding extraction, splitting, training, and evaluation
merge_data.py        # Converts IMDb review files into the input CSV
requirements.txt     # Main dependencies with pinned versions
experiments_log.csv  # Experiment metrics
my_plot.png          # Learning curves
```

**Installation**

```bash
python -m pip install -r requirements.txt
```

BERT uses CUDA when available, otherwise CPU. Model files may be downloaded on the first run.

**Dataset Preparation**

Data and generated embeddings are not included in the repository.

1. Download the dataset from the [official Stanford IMDb page](https://ai.stanford.edu/~amaas/data/sentiment/).
2. Extract the archive into the project root so that `aclImdb/train/pos/`, `aclImdb/train/neg/`, `aclImdb/test/pos/`, and `aclImdb/test/neg/` exist.
3. Generate the input CSV:

   ```bash
   python merge_data.py
   ```

This creates `stanford_dataset_merged.csv` with the columns `text`, `label`, and `split`.

**Running**

For the first run, set these flags in `main.py`:

```python
CREATE_EMBEDDINGS = True
CREATE_SPLITS = True
```

Then run from the project root:

```bash
python main.py
```

The script generates embeddings and training subsets, then runs **33 logistic regression experiments**.

For subsequent runs, set both flags to `False` to reuse the generated splits.

**Results**

- `experiments_log.csv` — parameters and test metrics for each experiment.
- `my_plot.png` — mean metrics by training size, with ± one standard deviation.

**Note:** each execution deletes the previous experiment log before starting.

**Dataset Reference**

Maas et al. (2011), [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/).
