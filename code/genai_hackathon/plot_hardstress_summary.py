import os
import pandas as pd
import matplotlib.pyplot as plt

IN = "genai_hackathon/candidate_ablation_hardstress_suite_out/candidate_ablation_results.csv"
OUT = "genai_hackathon/final_paper_figures"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(IN)

# Keep only the strongest comparison
df = df[df["method"].isin(["PPO", "PPO+CONF", "GENAI-9"])].copy()

# Rename for paper
df["method"] = df["method"].replace({
    "PPO": "PPO",
    "PPO+CONF": "PPO+CONF",
    "GENAI-9": "PPO+GENAI"
})

# -----------------------------
# Figure 1: main comparison
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

metrics = [
    ("sat_percent", "STL satisfaction (%)", True),
    ("rho", "Mean STL robustness", True),
    ("rmse", "RMSE", False),
]

for ax, (col, ylabel, higher_better) in zip(axes, metrics):
    bars = ax.bar(df["method"], df[col])

    for bar, val in zip(bars, df[col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}" if col != "sat_percent" else f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel(ylabel)
    ax.set_xticklabels(df["method"], rotation=20)
    ax.axhline(0, linestyle="--", linewidth=1)

fig.suptitle("Hard-Stress Evaluation: Safety and Tracking Performance")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_hardstress_main_comparison.png"), dpi=300)
plt.close()


# -----------------------------
# Figure 2: candidate ablation
# -----------------------------
df2 = pd.read_csv(IN)
df2 = df2[df2["method"].str.contains("GENAI")].copy()
df2["candidates"] = df2["method"].str.extract(r"GENAI-(\d+)").astype(int)

df2 = df2.sort_values("candidates")

fig, ax1 = plt.subplots(figsize=(6.5, 3.5))

ax1.plot(df2["candidates"], df2["rho"], marker="o", linewidth=2, label="Robustness")
ax1.set_xlabel("Number of generated candidates")
ax1.set_ylabel("Mean STL robustness")

ax2 = ax1.twinx()
ax2.plot(df2["candidates"], df2["rmse"], marker="s", linewidth=2, linestyle="--", label="RMSE")
ax2.set_ylabel("RMSE")

fig.suptitle("Candidate Ablation: Effect of Generated Actions")
fig.tight_layout()
plt.savefig(os.path.join(OUT, "fig_candidate_ablation.png"), dpi=300)
plt.close()

print("Saved:")
print(os.path.join(OUT, "fig_hardstress_main_comparison.png"))
print(os.path.join(OUT, "fig_candidate_ablation.png"))