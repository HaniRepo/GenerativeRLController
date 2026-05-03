# genai_hackathon/plot_hardstress_satisfaction_from_result.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = "genai_hackathon/hardstress_result_figures"
CSV_PATH = "genai_hackathon/hardstress_satisfaction_figure_out/result.csv"   # put your CSV file in the project root, or change this path

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV_PATH)

methods = ["PPO", "PPO+CONF", "PPO+GENAI"]
x_pos = {m: i for i, m in enumerate(methods)}

plt.figure(figsize=(7.0, 3.8))

for method in methods:
    sub = df[df["method"] == method].copy()
    for _, row in sub.iterrows():
        x = x_pos[method]
        jitter = (int(row["seed"]) - 42 - 4.5) * 0.025

        marker = "o" if int(row["sat"]) == 1 else "x"
        size = 80 if int(row["sat"]) == 1 else 90

        plt.scatter(
            x + jitter,
            row["rho"],
            s=size,
            marker=marker,
            linewidths=2,
        )

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1.4,
    label=r"STL boundary $\rho=0$",
)

plt.xticks(range(len(methods)), methods)
plt.ylabel(r"STL robustness $\rho$")
plt.xlabel("Method")
plt.title("Hard-Stress Per-Seed STL Satisfaction")
plt.legend(fontsize=8)
plt.tight_layout()

fig_path = os.path.join(OUT, "fig_hardstress_robustness_scatter.png")
plt.savefig(fig_path, dpi=300)
plt.close()

# Summary table
summary = (
    df.groupby("method")
    .agg(
        sat_percent=("sat", lambda x: 100 * np.mean(x)),
        mean_rho=("rho", "mean"),
        mean_rmse=("rmse", "mean"),
        changed=("changed", "mean"),
    )
    .reindex(methods)
)

summary_path = os.path.join(OUT, "hardstress_summary.csv")
summary.to_csv(summary_path)

print("\nSummary:")
print(summary)

print("\nSaved:")
print(fig_path)
print(summary_path)