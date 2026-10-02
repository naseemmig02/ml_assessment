import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("train-test.csv")
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["rate_per_mile"] = df["posted_rate"] / df["distance"]

os.makedirs("eda_charts", exist_ok=True)

# 1) Rate distribution by equipment
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Freight Rate Prediction - EDA Overview", fontsize=14, fontweight="bold")
colors = {"Dry Van":"#2196F3","Reefer":"#4CAF50","Flatbed":"#FF9800"}
for ax, eq in zip(axes, ["Dry Van", "Reefer", "Flatbed"]):
    sub = df[df["equipment"]==eq]["posted_rate"]
    ax.hist(sub, bins=60, color=colors[eq], alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_title(f"{eq} (n={len(sub):,})", fontsize=11)
    ax.set_xlabel("Posted Rate ($)")
    ax.set_ylabel("Frequency")
    med = sub.median()
    ax.axvline(med, color="red", linestyle="--", label=f"Median: ${med:,.0f}")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("eda_charts/01_rate_dist_by_equipment.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 1 saved")

# 2) Monthly trend
monthly = df.groupby(df["date"].dt.to_period("M"))["posted_rate"].agg(["mean","std"]).reset_index()
monthly["date_dt"] = monthly["date"].dt.to_timestamp()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly["date_dt"], monthly["mean"], marker="o", color="#1565C0", linewidth=2.2, markersize=5, label="Mean rate")
ax.fill_between(monthly["date_dt"], monthly["mean"]-monthly["std"], monthly["mean"]+monthly["std"], alpha=0.15, color="#1565C0", label="+-1 Std dev")
ax.set_title("Monthly Average Freight Rate Trend (Jan-Oct 2025)", fontsize=13, fontweight="bold")
ax.set_ylabel("Posted Rate ($)")
ax.set_xlabel("Month")
ax.spines[["top","right"]].set_visible(False)
ax.legend()
plt.tight_layout()
plt.savefig("eda_charts/02_monthly_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 2 saved")

# 3) Rate vs Distance scatter
fig, ax = plt.subplots(figsize=(10, 6))
for eq, grp in df.groupby("equipment"):
    samp = grp.sample(min(3000, len(grp)), random_state=42)
    ax.scatter(samp["distance"], samp["posted_rate"], alpha=0.3, s=6, c=colors[eq], label=eq)
ax.set_xlabel("Distance (miles)")
ax.set_ylabel("Posted Rate ($)")
ax.set_title("Rate vs Distance by Equipment Type", fontsize=13, fontweight="bold")
ax.legend()
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("eda_charts/03_rate_vs_distance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 3 saved")

# 4) Rate per mile boxplot
fig, ax = plt.subplots(figsize=(9, 5))
data_box = [df[df["equipment"]==eq]["rate_per_mile"].dropna().values for eq in ["Dry Van","Reefer","Flatbed"]]
bp = ax.boxplot(data_box, labels=["Dry Van","Reefer","Flatbed"], patch_artist=True, notch=True, sym="")
box_colors = ["#2196F3","#4CAF50","#FF9800"]
for patch, col in zip(bp["boxes"], box_colors):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax.set_title("Rate per Mile by Equipment Type", fontsize=13, fontweight="bold")
ax.set_ylabel("Rate per Mile ($/mile)")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("eda_charts/04_rpm_by_equipment.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 4 saved")

# 5) Market index over time
mi_monthly = df.groupby(df["date"].dt.to_period("M"))["market_index"].mean().reset_index()
mi_monthly["date_dt"] = mi_monthly["date"].dt.to_timestamp()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(mi_monthly["date_dt"], mi_monthly["market_index"], marker="o", color="#7B1FA2", linewidth=2.2, markersize=5)
ax.set_title("Monthly Average Market Index (Jan-Oct 2025)", fontsize=13, fontweight="bold")
ax.set_ylabel("Market Index")
ax.set_xlabel("Month")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("eda_charts/05_market_index_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 5 saved")

# 6) Correlation heatmap
corr_cols = ["distance","weight","market_index","quote_signal","posted_rate"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(7,5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="right")
ax.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j,i,f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=10, color="black")
plt.colorbar(im, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_charts/06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 6 saved")

# 7) Missing value analysis
missing = pd.DataFrame({
    "Column": ["weight","market_index","quote_signal","posted_rate"],
    "Missing": [300, 374, 0, 0],
    "Pct": [300/48000*100, 374/48000*100, 0, 0]
})
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(missing["Column"], missing["Pct"], color=["#EF5350","#EF5350","#66BB6A","#66BB6A"])
ax.set_xlabel("Missing (%)")
ax.set_title("Missing Value Analysis (Training Set)", fontsize=13, fontweight="bold")
for i, (v, n) in enumerate(zip(missing["Pct"], missing["Missing"])):
    ax.text(v+0.01, i, f"{n} rows ({v:.2f}%)", va="center", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("eda_charts/07_missing_values.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 7 saved")

# 8) December predictions chart
dec = pd.read_csv("december-chart-inputs.csv")
dec["date"] = pd.to_datetime(dec["date"])
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dec["date"], dec["predicted_rate"], marker="o", color="#064A56", linewidth=2.6, markersize=4)
floor = dec["predicted_rate"].min()
ax.fill_between(dec["date"], dec["predicted_rate"], floor-20, color="#064A56", alpha=0.08)
ax.set_title("December 2025: Predicted Rates (Lexington to Fort Wayne)", fontsize=13, fontweight="bold")
ax.set_ylabel("Predicted Rate ($)")
ax.set_xlabel("Date")
ax.tick_params(axis="x", rotation=35)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("eda_charts/08_december_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 8 saved")

print("\nAll EDA charts saved to eda_charts/")
