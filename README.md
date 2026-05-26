Step 3 — Write the README
Create README.md in the project root:
markdown# JMLR Thematic Alignment Analysis

A pipeline to quantitatively assess whether papers published in the
**Journal of Machine Learning Research (JMLR)** align with its stated
Aims & Scope, using semantic embeddings and cosine similarity.

## Project structure
jmlr-alignment/
├── src/
│   ├── base.py                  # Abstract base classes
│   ├── data/                    # Fetching and preprocessing
│   ├── models/                  # TF-IDF and SBERT embedders
│   ├── evaluation/              # Alignment scoring
│   └── visualization/           # Plotting utilities
├── notebooks/
│   └── 01_alignment_analysis.ipynb
├── tests/
│   └── test_pipeline.py
├── reports/                     # Output figures
├── data/                        # Local data (not tracked by git)
├── run_all.py                    # running whole the pipeline
├── Original Notebook              #notebook consisiting the whole pipeline (faltten version)
└── PDF Report                     

## Setup

```bash
git clone https://github.com/mohmmadhadi/jmlr-alignment.git
cd jmlr-alignment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

Run each step in order from the project root:

```bash
# 1. Fetch papers from arXiv and preprocess
python -m src.data.run_pipeline

# 2. Generate TF-IDF and SBERT embeddings
python -m src.models.run_embeddings

# 3. Compute alignment scores
python -m src.evaluation.run_evaluation

# 4. Open the analysis notebook
jupyter notebook notebooks/01_alignment_analysis.ipynb
```
OR
```bash
# 1. Run the whole pipeline at once
python -m run_all.py

# 2. Open the analysis notebook
jupyter notebook notebooks/01_alignment_analysis.ipynb
```


## Running tests

```bash
pytest tests/ -v
```

## Key findings

- **SBERT mean alignment score: 0.317** across 500 papers (2019–2026)
- **Thematic drift detected**: slope of −0.0007/year indicates somehow stability of JMLR's core ML focus over time
- **Outlier detection**: bottom 5% papers are clearly non-ML
  (physics, pure mathematics, materials science), likely arXiv
  cross-listing artifacts
- **Top papers** (score > 0.55) are canonical ML works on Bayesian
  inference, gradient estimation, and variational methods

## Methods

| Step | Tool |
|------|------|
| Data collection | arXiv API (`arxiv` Python library) |
| Baseline embedding | TF-IDF (scikit-learn, 5000 features) |
| Semantic embedding | Sentence-BERT `all-MiniLM-L6-v2` |
| Alignment metric | Cosine similarity |
| Drift analysis | Year-over-year mean score with linear trend |
| Projecting the embedding space |  UMAP Projection |
