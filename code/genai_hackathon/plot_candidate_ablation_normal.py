import os
import pandas as pd
import matplotlib.pyplot as plt

IN = "genai_hackathon/candidate_ablation_suite_out/candidate_ablation_results.csv"
OUT = "genai_hackathon/final_paper_figures"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(IN)

# -------- Figure 1: main method comparison --------
main = df[df["method"].isin(["PPO", "PPO+CONF", "GENAI-5"])].copy()
main["method"] = main["method"].replace({"GENAI-5": "PPO+GENAI"})

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

plots = [
    ("sat_percent", "STL satisfaction (%)", "{:.0f}%"),
    ("rho", "Mean STL robustness", "{:.4f}"),
    ("rmse", "RMSE", "{:.4f}"),
]

for ax, (col, ylabel, fmt) in zip(axes, plots):
    bars = ax.bar(main["method"], main[col])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)

    if col == "rho":
        ax.axhline(0, linestyle="--", linewidth=1)

    for b, v in zip(bars, main[col]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=8,
        )

fig.suptitle("Overspeed-Tracking Stress: Safety and Performance")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_overspeed_tracking_main_comparison.png"), dpi=300)
plt.close()


# -------- Figure 2: candidate ablation --------
gen = df[df["method"].str.contains("GENAI")].copy()
gen["candidates"] = gen["method"].str.extract(r"GENAI-(\d+)").astype(int)
gen = gen.sort_values("candidates")

fig, ax1 = plt.subplots(figsize=(6.5, 3.5))

ax1.plot(
    gen["candidates"],
    gen["rho"],
    marker="o",
    linewidth=2,
    label="Mean STL robustness",
)
ax1.set_xlabel("Number of generated candidates")
ax1.set_ylabel("Mean STL robustness")

ax2 = ax1.twinx()
ax2.plot(
    gen["candidates"],
    gen["rmse"],
    marker="s",
    linewidth=2,
    linestyle="--",
    label="RMSE",
)
ax2.set_ylabel("RMSE")

# mark best RMSE
best = gen.loc[gen["rmse"].idxmin()]
ax2.scatter([best["candidates"]], [best["rmse"]], s=70, marker="*", zorder=5)
ax2.text(
    best["candidates"],
    best["rmse"],
    " best RMSE",
    fontsize=8,
    va="bottom",
)

fig.suptitle("Candidate Ablation: Effect of Generated Actions")
fig.tight_layout()
plt.savefig(os.path.join(OUT, "fig_candidate_ablation_normal.png"), dpi=300)
plt.close()

print("Saved:")
print(os.path.join(OUT, "fig_overspeed_tracking_main_comparison.png"))
print(os.path.join(OUT, "fig_candidate_ablation_normal.png"))