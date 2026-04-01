"""Predictive pipeline v3: binary + 3-class, LogReg + RF + XGBoost, LOPO-CV."""

import time, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier

DATA = "/Users/wu.w4/Desktop/User-log/batch_output_k5/all_windows_with_clusters.csv"
FEATURES = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time",
    "puzzle_active", "error_count", "time_since_action",
]

# ── Load & clean ──────────────────────────────────────────────────────
df = pd.read_csv(DATA)
X_raw = df[FEATURES].values
X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
players = df["player_id"].values
unique_players = sorted(df["player_id"].unique())

# ── Label schemes ─────────────────────────────────────────────────────
#   Binary:  0=not-confused (C0,C2,C3)  1=confused (C1,C4)
#   3-class: 0=productive (C2)  1=confused (C1,C4)  2=transition (C0,C3)
label_schemes = {
    "binary": {
        "y": (df["cluster_id"].isin([1, 4])).astype(int).values,
        "names": ["Not Confused", "Confused"],
    },
    "3-class": {
        "y": df["cluster_id"].map({
            0: 2,   # C0 → transition
            1: 1,   # C1 → confused
            2: 0,   # C2 → productive
            3: 2,   # C3 → transition
            4: 1,   # C4 → confused
        }).astype(int).values,
        "names": ["Productive", "Confused", "Transition"],
    },
}

# ── Models ────────────────────────────────────────────────────────────
def make_models(n_classes):
    xgb_objective = "binary:logistic" if n_classes == 2 else "multi:softmax"
    xgb_params = dict(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    if n_classes > 2:
        xgb_params["objective"] = xgb_objective
        xgb_params["num_class"] = n_classes

    return {
        "LogReg": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42),
        "RF": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42,
            n_jobs=-1),
        "XGB": XGBClassifier(**xgb_params),
    }

# ── Pretty-print confusion matrix ────────────────────────────────────
def print_cm(cm, names):
    max_name = max(len(n) for n in names)
    header = " " * (max_name + 2) + "  ".join(f"{n:>8s}" for n in names)
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:8d}" for v in row)
        print(f"  {names[i]:>{max_name}s}  {row_str}")

# ── Run experiments ───────────────────────────────────────────────────
rf_importances = {scheme: [] for scheme in label_schemes}
xgb_importances = {scheme: [] for scheme in label_schemes}
summary_rows = []

t0 = time.time()

for scheme_name, scheme in label_schemes.items():
    y = scheme["y"]
    names = scheme["names"]
    n_classes = len(names)

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: {scheme_name}  ({n_classes} classes)")
    dist = {names[c]: int((y == c).sum()) for c in range(n_classes)}
    print(f"  Distribution: {dist}")
    print(f"{'='*65}")

    for model_name in ["LogReg", "RF", "XGB"]:
        all_true, all_pred = [], []

        for i, held_out in enumerate(unique_players):
            mask = players == held_out
            X_tr, y_tr = X_raw[~mask], y[~mask]
            X_te, y_te = X_raw[mask], y[mask]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = make_models(n_classes)[model_name]
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)

            fold_f1 = f1_score(y_te, preds, average="macro")
            print(f"    {model_name:6s}  Fold {i+1:2d}/11  "
                  f"held_out=User-{held_out:2d}  "
                  f"n={mask.sum():4d}  F1={fold_f1:.3f}")

            all_true.extend(y_te)
            all_pred.extend(preds)

            if model_name == "RF":
                rf_importances[scheme_name].append(model.feature_importances_)
            elif model_name == "XGB":
                xgb_importances[scheme_name].append(model.feature_importances_)

        acc = accuracy_score(all_true, all_pred)
        f1m = f1_score(all_true, all_pred, average="macro")
        summary_rows.append((scheme_name, model_name, acc, f1m))
        print(f"\n  >>> {model_name} | {scheme_name} | "
              f"Accuracy={acc:.3f}  F1-macro={f1m:.3f}")
        print(classification_report(
            all_true, all_pred, target_names=names, digits=3))

        cm = confusion_matrix(all_true, all_pred)
        print_cm(cm, names)
        print()

# ── Feature importance (RF) ──────────────────────────────────────────
print(f"\n{'='*65}")
print("  RF FEATURE IMPORTANCE  (mean ± std across 11 LOPO folds)")
print(f"{'='*65}")
for scheme_name, imps in rf_importances.items():
    if not imps:
        continue
    imps = np.array(imps)
    means = imps.mean(axis=0)
    stds  = imps.std(axis=0)
    order = np.argsort(means)[::-1]
    print(f"\n  [{scheme_name}]")
    for rank, idx in enumerate(order, 1):
        print(f"    {rank}. {FEATURES[idx]:<20s}  "
              f"{means[idx]:.4f} ± {stds[idx]:.4f}")

# ── Feature importance (XGBoost) ─────────────────────────────────────
print(f"\n{'='*65}")
print("  XGB FEATURE IMPORTANCE  (mean ± std across 11 LOPO folds)")
print(f"{'='*65}")
for scheme_name, imps in xgb_importances.items():
    if not imps:
        continue
    imps = np.array(imps)
    means = imps.mean(axis=0)
    stds  = imps.std(axis=0)
    order = np.argsort(means)[::-1]
    print(f"\n  [{scheme_name}]")
    for rank, idx in enumerate(order, 1):
        print(f"    {rank}. {FEATURES[idx]:<20s}  "
              f"{means[idx]:.4f} ± {stds[idx]:.4f}")

# ── Final comparison table ────────────────────────────────────────────
print(f"\n{'='*65}")
print("  MODEL COMPARISON SUMMARY")
print(f"{'='*65}")
print(f"  {'Scheme':<10s}  {'Model':<8s}  {'Accuracy':>8s}  {'F1-macro':>8s}")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
for scheme, model, acc, f1m in summary_rows:
    print(f"  {scheme:<10s}  {model:<8s}  {acc:>8.3f}  {f1m:>8.3f}")

elapsed = time.time() - t0
print(f"\n  Total time: {elapsed:.1f}s")
print(f"{'='*65}")
