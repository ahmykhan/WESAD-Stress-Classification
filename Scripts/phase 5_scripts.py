# ==============================================================================
# PHASE 5: OUTPUT, ERROR ANALYSIS & THEORETICAL INSIGHTS
# File Name: F26-02.py
# Group ID: F26-02 | Track A: Biosignal Stress Classification (WESAD)
# Description: Reproducible end-to-end verification, feature importance theory,
#              outlier anatomy (S2 vs S7), and 60s online calibration benchmark.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# 1. Reproducibility & Setup
SEED = 42
np.random.seed(SEED)
OUT_DIR = "phase5_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = "WESAD_ML_Ready_Features.csv"  # put the CSV next to this script
if not os.path.exists(CSV_PATH):
    CSV_PATH = "/content/drive/MyDrive/WESAD_ML_Ready_Features.csv"

df = pd.read_csv(CSV_PATH)
features = [c for c in df.columns if c not in ("Subject", "State")]
subjects = sorted(df["Subject"].unique(), key=lambda s: int(s[1:]))
y_all = (df["State"] == "Stressed").astype(int).values

print(f"✅ Loaded {len(df)} windows across {len(subjects)} subjects.")

# ------------------------------------------------------------------------------
# 2. THEORETICAL FEATURE IMPORTANCE (GINI IMPORTANCE ACROSS ALL 15 FOLDS)
# ------------------------------------------------------------------------------
importances = np.zeros(len(features))
uncal_results = []
online_cal_results = []

for s in subjects:
    test_mask = (df["Subject"] == s).values
    train_df = df.loc[~test_mask]
    test_df = df.loc[test_mask].reset_index(drop=True)

    X_tr, y_tr = train_df[features].values, (train_df["State"] == "Stressed").astype(int).values
    X_te, y_te = test_df[features].values, (test_df["State"] == "Stressed").astype(int).values

    # A. Standard Population-Scaled LOSO
    scaler = StandardScaler().fit(X_tr)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=4, random_state=SEED, n_jobs=-1)
    rf.fit(scaler.transform(X_tr), y_tr)
    importances += rf.feature_importances_ / len(subjects)

    probs_uncal = rf.predict_proba(scaler.transform(X_te))[:, 1]
    preds_uncal = (probs_uncal >= 0.5).astype(int)
    uncal_results.append({
        "Subject": s,
        "Acc_Uncal": accuracy_score(y_te, preds_uncal),
        "F1_Uncal": f1_score(y_te, preds_uncal, zero_division=0),
        "AUC_Uncal": roc_auc_score(y_te, probs_uncal)
    })

    # B. Realistic Online 60-Second Calibration (first 12 Relaxed windows = 60 s of rest)
    # FIX: training subjects are normalised with the SAME procedure as the test subject
    # (subtract each person's own 60 s resting mean). The original trained on full-session
    # z-scores but tested on 60 s statistics, so the model saw a different feature scale.
    # Mean-centering only: the std of just 12 windows is too unstable to divide by.
    N_CAL = 12
    def calib_mean(sub_df):
        return sub_df[sub_df["State"] == "Relaxed"][features].iloc[:N_CAL].values.mean(axis=0)

    X_tr_norm = np.vstack([
        g[features].values - calib_mean(g) for _, g in train_df.groupby("Subject")
    ])
    y_tr_norm = np.concatenate([
        (g["State"] == "Stressed").astype(int).values for _, g in train_df.groupby("Subject")
    ])
    rf_cal = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=4, random_state=SEED, n_jobs=-1)
    rf_cal.fit(X_tr_norm, y_tr_norm)

    X_te_cal = X_te - calib_mean(test_df)
    probs_cal = rf_cal.predict_proba(X_te_cal)[:, 1]
    preds_cal = (probs_cal >= 0.5).astype(int)
    # Score only windows NOT used for calibration (the calibration windows are the first N_CAL rows,
    # because each subject's Relaxed block comes first in the CSV).
    keep = np.ones(len(y_te), dtype=bool); keep[:N_CAL] = False
    online_cal_results.append({
        "Subject": s,
        "Acc_Cal_60s": accuracy_score(y_te[keep], preds_cal[keep]),
        "F1_Cal_60s": f1_score(y_te[keep], preds_cal[keep], zero_division=0),
        "AUC_Cal_60s": roc_auc_score(y_te[keep], probs_cal[keep])
    })

res_df = pd.merge(pd.DataFrame(uncal_results), pd.DataFrame(online_cal_results), on="Subject")
res_df.to_csv(f"{OUT_DIR}/phase5_reproducibility_summary.csv", index=False)

print("\n📊 --- PHASE 5 REPRODUCIBLE BENCHMARK SUMMARY (15-FOLD LOSO) ---")
print(f"Uncalibrated LOSO Mean Accuracy      : {res_df['Acc_Uncal'].mean()*100:.2f}% (±{res_df['Acc_Uncal'].std()*100:.2f}%)")
print(f"60s Calibrated (mean-centred) Mean Acc: {res_df['Acc_Cal_60s'].mean()*100:.2f}% (±{res_df['Acc_Cal_60s'].std()*100:.2f}%)")
print(f"Subject S7 Accuracy (Uncal -> 60s Cal): {res_df.loc[res_df['Subject']=='S7', 'Acc_Uncal'].values[0]*100:.2f}% -> {res_df.loc[res_df['Subject']=='S7', 'Acc_Cal_60s'].values[0]*100:.2f}%")

# ------------------------------------------------------------------------------
# 3. FIGURE 1: INTER-SUBJECT TONIC EDA SHIFT (VISUALIZING WHY S7 & S2 FAIL)
# ------------------------------------------------------------------------------
plt.figure(figsize=(13, 5))
sns.boxplot(data=df, x="Subject", y="EDA_Mean", hue="State", order=subjects, palette={"Relaxed": "#4C72B0", "Stressed": "#DD8452"})
plt.title("Inter-Subject Physiological Baseline Heterogeneity (Tonic EDA Distribution)", fontsize=13, fontweight="bold")
plt.ylabel("Mean Electrodermal Activity (µS)")
plt.xlabel("Subject ID (Held-Out LOSO Target)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig1_InterSubject_EDA_Shift.png", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 4. FIGURE 2: BIOMARKER FEATURE IMPORTANCE (AUTONOMIC NERVOUS SYSTEM THEORY)
# ------------------------------------------------------------------------------
imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)
colors = ["#2b5c8f" if f.startswith("EDA") else "#c44e52" for f in imp_df["Feature"]]

plt.figure(figsize=(9, 4.5))
plt.barh(imp_df["Feature"], imp_df["Importance"], color=colors)
plt.title("Mean Gini Feature Importance Across 15 LOSO Folds (Blue = EDA, Red = ECG)", fontsize=12, fontweight="bold")
plt.xlabel("Relative Gini Importance")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig2_Feature_Importance_Theory.png", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 5. FIGURE 3: ERROR MITIGATION VIA 60-SECOND BASELINE CALIBRATION
# ------------------------------------------------------------------------------
x = np.arange(len(subjects))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - width/2, res_df["Acc_Uncal"] * 100, width, label="Uncalibrated Population LOSO", color="#8c969e")
ax.bar(x + width/2, res_df["Acc_Cal_60s"] * 100, width, label="60-Second Resting Calibration LOSO", color="#2ca02c")
ax.axhline(50, color="red", linestyle=":", linewidth=1, label="Random Chance (50%)")

ax.set_ylabel("Classification Accuracy (%)")
ax.set_title("Per-Subject Generalization: Uncalibrated vs. 60-Second Online Calibration", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylim(0, 108)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig3_Calibration_Recovery.png", dpi=300)
plt.close()

print(f"\n🎉 All Phase 5 outputs and figures saved to ./{OUT_DIR}/")