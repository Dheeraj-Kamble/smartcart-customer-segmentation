"""Prepare SmartCart customer features, cluster customers, and save charts."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("SMARTCART_CSV", PROJECT_DIR / "data" / "smartcart_customers.csv"))
OUTPUT = PROJECT_DIR / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE)
initial_shape = df.shape
missing_before = df.isnull().sum()
df["Income"] = df["Income"].fillna(df["Income"].median())

df["Age"] = 2026 - df["Year_Birth"]
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
reference_date = df["Dt_Customer"].max()
df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days
spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df["Total_Spending"] = df[spending_cols].sum(axis=1)
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
})
df["Living_With"] = df["Marital_Status"].replace({
    "Married": "Partner", "Together": "Partner", "Single": "Alone",
    "Divorced": "Alone", "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
})

drop_cols = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"] + spending_cols
df_cleaned = df.drop(columns=drop_cols)

# Pairwise feature plots (matplotlib equivalent of seaborn's pairplot).
pair_cols = ["Income", "Recency", "Response", "Age", "Total_Spending", "Total_Children"]
pair = pd.plotting.scatter_matrix(df_cleaned[pair_cols], figsize=(12, 12), diagonal="hist", alpha=0.35, s=10)
for ax in pair.ravel():
    ax.tick_params(labelsize=7)
plt.suptitle("Customer Feature Pair Plots", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT / "feature_pairplot.png", dpi=160, bbox_inches="tight")
plt.close("all")

corr = df_cleaned.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(11, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=7)
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Numeric Feature Correlation Heatmap")
fig.tight_layout()
fig.savefig(OUTPUT / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
plt.close(fig)

cat_cols = ["Education", "Living_With"]
ohe = OneHotEncoder()
enc = ohe.fit_transform(df_cleaned[cat_cols])
enc_df = pd.DataFrame(enc.toarray(), columns=ohe.get_feature_names_out(cat_cols), index=df_cleaned.index)
df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)

X_scaled = StandardScaler().fit_transform(df_encoded)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
labels = AgglomerativeClustering(n_clusters=4).fit_predict(X_pca)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap="rainbow", s=14, alpha=0.75)
ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title="Agglomerative Customer Clusters in PCA Space")
fig.colorbar(scatter, ax=ax, label="Cluster", shrink=0.7)
fig.tight_layout()
fig.savefig(OUTPUT / "pca_clusters_3d.png", dpi=170, bbox_inches="tight")
plt.close(fig)

clustered = df_encoded.copy()
clustered["cluster"] = labels
palette = ["red", "blue", "gold", "green"]
counts = clustered["cluster"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(counts.index.astype(str), counts.values, color=palette)
ax.set(xlabel="Cluster", ylabel="Customer Count", title="Customers per Cluster")
fig.tight_layout()
fig.savefig(OUTPUT / "cluster_counts.png", dpi=160, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
for cluster_id, group in clustered.groupby("cluster"):
    ax.scatter(group["Total_Spending"], group["Income"], color=palette[cluster_id], label=f"Cluster {cluster_id}", alpha=0.65, s=18)
ax.set(xlabel="Total Spending", ylabel="Income", title="Income vs. Total Spending by Cluster")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "income_spending_clusters.png", dpi=160, bbox_inches="tight")
plt.close(fig)

summary = clustered.groupby("cluster").mean(numeric_only=True)
summary.insert(0, "Customer_Count", counts)
summary.to_csv(OUTPUT / "cluster_summary.csv", index_label="cluster")

overview = pd.DataFrame({
    "metric": ["input_rows", "input_columns", "missing_income_before", "missing_values_after_income_fill", "pca_variance_pc1", "pca_variance_pc2", "pca_variance_pc3", "reference_date", "clusters"],
    "value": [initial_shape[0], initial_shape[1], int(missing_before.get("Income", 0)), int(df_cleaned.isnull().sum().sum()), *pca.explained_variance_ratio_.tolist(), reference_date.date().isoformat(), 4]
})
overview.to_csv(OUTPUT / "analysis_overview.csv", index=False)

print(f"Input shape: {initial_shape}")
print(f"Missing Income before fill: {int(missing_before.get('Income', 0))}")
print(f"Cluster sizes:\n{counts.to_string()}")
print(f"PCA variance explained: {pca.explained_variance_ratio_.round(4).tolist()}")
print(f"Results saved in: {OUTPUT}")
