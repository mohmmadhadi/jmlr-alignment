"""Master pipeline runner — executes all steps in order."""

import subprocess
import sys


steps = [
    ("Fetching papers",         "src.data.run_pipeline"),
    ("Generating embeddings",   "src.models.run_embeddings"),
    ("Evaluating alignment",    "src.evaluation.run_evaluation"),
    ("Fitting topic model",     "src.models.run_topics"),
    ("Projecting UMAP",         "src.models.run_umap"),
]

for label, module in steps:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    result = subprocess.run(
        [sys.executable, "-m", module],
        check=False,
    )
    if result.returncode != 0:
        print(f"\nPipeline stopped: {module} failed.")
        sys.exit(1)

print("\n" + "="*55)
print("  Pipeline complete. Open the notebook to view results.")
print("="*55)