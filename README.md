# SmartCart Customer Segmentation

This project prepares customer features, reduces the feature space with PCA, and assigns four groups using agglomerative clustering. It also generates feature and cluster visualizations.

## Run it

1. Install Python 3.10 or newer.
2. Install packages: `python -m pip install -r requirements.txt`.
3. Put `smartcart_customers.csv` in `data/`, or set `SMARTCART_CSV` to its location.
4. Run `python analyze_customers.py`.

The source dataset is not included because it contains customer-level records. The generated row-level customer export is excluded from Git as well. Aggregate summaries and charts are included under `results/`.

## Outputs

- `results/feature_pairplot.png`
- `results/correlation_heatmap.png`
- `results/pca_clusters_3d.png`
- `results/cluster_counts.png`
- `results/income_spending_clusters.png`
- `results/cluster_summary.csv`
- `results/analysis_overview.csv`

Age follows the supplied analysis assumption `2026 - Year_Birth`. The joining date reference is the latest customer date in the input file. Cluster numbers are arbitrary labels and do not imply ranking.
