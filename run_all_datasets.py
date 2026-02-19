"""
=======================================================================
Quantile Regression using Random Forest Proximities — Multi-Dataset
=======================================================================
Basado en: Li et al. (2024) "Quantile Regression using Random Forest
Proximities". arXiv:2408.02355v1 [stat.ML].

Ejecuta la metodologia completa del paper sobre 3 datasets:
  1. Online Retail II  (Quantity)
  2. Energy Efficiency  (Heating Load — Y1)
  3. Online News Popularity  (shares, log-transformed)

Uso:
    python run_all_datasets.py
=======================================================================
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# =====================================================================
# Rutas
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Estilo global de graficos — storytelling
plt.rcParams.update({
    "figure.figsize": (14, 7),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLORS = {"QRF": "#2196F3", "RF-GAP": "#FF9800", "OOB": "#4CAF50", "ORIGINAL": "#9C27B0"}
METHODS = ["QRF", "RF-GAP", "OOB", "ORIGINAL"]
QUANTILES = [0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995]


# =====================================================================
# SECCION 1: PROXIMIDADES (Sec. 2.2 del paper)
# =====================================================================

def get_leaf_indices(rf, X):
    """Indices de nodos terminales por observacion y arbol."""
    return rf.apply(X)


def compute_proximity_original(leaf_train, leaf_test):
    """
    Original Proximity (Eq. 5):
    Prox_Original(i,j) = (1/T) sum_t I(j in v_i(t))
    """
    n_te, n_tr, T = leaf_test.shape[0], leaf_train.shape[0], leaf_train.shape[1]
    prox = np.zeros((n_te, n_tr))
    for t in range(T):
        prox += (leaf_test[:, t:t+1] == leaf_train[:, t].reshape(1, -1))
    prox /= T
    rs = prox.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return prox / rs


def compute_proximity_oob(rf, X_train, leaf_train, leaf_test):
    """
    OOB Proximity (Eq. 6):
    Solo pares donde la observacion de training es out-of-bag.
    """
    n_te, n_tr, T = leaf_test.shape[0], X_train.shape[0], leaf_train.shape[1]
    oob_masks = []
    for t in range(T):
        inbag = set(rf.estimators_samples_[t])
        oob_masks.append(np.array([i not in inbag for i in range(n_tr)]))

    prox = np.zeros((n_te, n_tr))
    oob_count = np.zeros((n_te, n_tr))
    for t in range(T):
        same = (leaf_test[:, t:t+1] == leaf_train[:, t].reshape(1, -1))
        oob_exp = oob_masks[t].reshape(1, -1)
        prox += same * oob_exp
        oob_count += oob_exp
    oob_count[oob_count == 0] = 1
    prox = prox / oob_count
    rs = prox.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return prox / rs


def compute_proximity_rfgap(rf, X_train, leaf_train, leaf_test):
    """
    RF-GAP Proximity (Eq. 4):
    Prox_GAP(i,j) = (1/|S_i|) sum_{t in S_i} c_j(t)*I(j in J_i(t))/|M_i(t)|
    Geometry- and Accuracy-Preserving proximity.
    """
    n_te, n_tr, T = leaf_test.shape[0], leaf_train.shape[0], leaf_train.shape[1]
    prox = np.zeros((n_te, n_tr))

    for t in range(T):
        inbag = rf.estimators_samples_[t]
        c_j = np.zeros(n_tr)
        uniq, cnts = np.unique(inbag, return_counts=True)
        c_j[uniq] = cnts

        tl_test = leaf_test[:, t]
        tl_train = leaf_train[:, t]

        leaf_M = {}
        for leaf in np.unique(tl_train):
            leaf_M[leaf] = c_j[tl_train == leaf].sum()

        for i in range(n_te):
            lf = tl_test[i]
            mask = (tl_train == lf)
            M = leaf_M.get(lf, 0)
            if M > 0:
                prox[i] += (c_j * mask) / M

    prox /= T
    rs = prox.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return prox / rs


# =====================================================================
# SECCION 2: PREDICCION CUANTILICA (Sec. 2.1, 2.3)
# =====================================================================

def quantile_predict_prox(prox, y_train, quantiles):
    """Eq. 7: F(y|X=x) = sum_i Prox(j,i) * I_{Y_i <= y}"""
    n = prox.shape[0]
    preds = {q: np.zeros(n) for q in quantiles}
    sidx = np.argsort(y_train)
    ys = y_train[sidx]

    for i in range(n):
        w = prox[i, sidx]
        cw = np.cumsum(w)
        tw = cw[-1]
        if tw > 0:
            cw /= tw
            for q in quantiles:
                idx = min(np.searchsorted(cw, q), len(ys) - 1)
                preds[q][i] = ys[idx]
        else:
            for q in quantiles:
                preds[q][i] = np.quantile(y_train, q)
    return preds


def qrf_predict(rf, X_test, leaf_train, y_train, quantiles):
    """QRF estandar (Meinshausen, 2006 — Ref. [26])."""
    leaf_te = rf.apply(X_test)
    n, T = X_test.shape[0], leaf_te.shape[1]
    preds = {q: np.zeros(n) for q in quantiles}

    for i in range(n):
        w = np.zeros(len(y_train))
        for t in range(T):
            same = (leaf_train[:, t] == leaf_te[i, t])
            ns = same.sum()
            if ns > 0:
                w[same] += 1.0 / ns
        w /= T

        sidx = np.argsort(y_train)
        ys, ws = y_train[sidx], w[sidx]
        cw = np.cumsum(ws)
        tw = cw[-1]
        if tw > 0:
            cw /= tw
            for q in quantiles:
                idx = min(np.searchsorted(cw, q), len(ys) - 1)
                preds[q][i] = ys[idx]
        else:
            for q in quantiles:
                preds[q][i] = np.quantile(y_train, q)
    return preds


# =====================================================================
# SECCION 3: METRICAS (Sec. 2.4)
# =====================================================================

def quantile_loss(y, q_pred, alpha):
    """Pinball Loss (Eq. 8)."""
    r = y - q_pred
    return np.mean(np.where(r > 0, alpha * np.abs(r), (1 - alpha) * np.abs(r)))


def mse(y, yhat):
    """Mean Squared Error (Eq. 9)."""
    return np.mean((y - yhat) ** 2)


def mape(y, yhat):
    """Mean Absolute Percentage Error (Eq. 10)."""
    m = y != 0
    if m.sum() == 0:
        return np.nan
    return np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100


# =====================================================================
# SECCION 4: PIPELINE DE EVALUACION
# =====================================================================

def run_pipeline(X, y, feature_names, dataset_name, plot_dir,
                 n_folds=5, sample_size=15000):
    """
    Pipeline completo del paper aplicado a un dataset.

    1. Grid Search + 5-fold CV (Sec. 4.1)
    2. Evaluacion de 4 metodos con k-fold CV (Sec. 4.2)
    3. Analisis de criterio de split (Sec. 4.3)
    4. Generacion de todas las visualizaciones
    """
    print(f"\n{'#' * 70}")
    print(f"  DATASET: {dataset_name}")
    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"{'#' * 70}")

    out = Path(plot_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Muestreo si es necesario (proximidad es O(n^2))
    if len(y) > sample_size:
        idx = np.random.choice(len(y), sample_size, replace=False)
        X, y = X[idx], y[idx]
        print(f"  Muestreado a {sample_size:,} registros")

    # ---- 4.1 Grid Search ----
    print("\n  [1/4] Grid Search + 5-Fold CV (Sec. 4.1)...")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 4, 8],
        "max_features": ["sqrt"],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        param_grid, cv=n_folds, scoring="neg_mean_squared_error",
        n_jobs=-1, verbose=0)
    gs.fit(X, y)
    bp = gs.best_params_
    print(f"    Mejores params: {bp}")
    print(f"    Mejor MSE (CV): {-gs.best_score_:.4f}")

    # ---- 4.2 Evaluacion 5-Fold CV ----
    print("\n  [2/4] Evaluacion 5-Fold CV — 4 metodos (Sec. 4.2)...")
    res_ql = {m: {q: [] for q in QUANTILES} for m in METHODS}
    res_mse = {m: [] for m in METHODS}
    res_mape = {m: [] for m in METHODS}
    last_preds, last_yt = {}, None

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        rf = RandomForestRegressor(**bp, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(Xtr, ytr)
        lt, le = get_leaf_indices(rf, Xtr), get_leaf_indices(rf, Xte)

        preds_all = {
            "QRF": qrf_predict(rf, Xte, lt, ytr, QUANTILES),
            "ORIGINAL": quantile_predict_prox(compute_proximity_original(lt, le), ytr, QUANTILES),
            "OOB": quantile_predict_prox(compute_proximity_oob(rf, Xtr, lt, le), ytr, QUANTILES),
            "RF-GAP": quantile_predict_prox(compute_proximity_rfgap(rf, Xtr, lt, le), ytr, QUANTILES),
        }

        for m in METHODS:
            for q in QUANTILES:
                res_ql[m][q].append(quantile_loss(yte, preds_all[m][q], q))
            med = preds_all[m][0.5]
            res_mse[m].append(mse(yte, med))
            res_mape[m].append(mape(yte, med))

        if fold == n_folds:
            last_preds, last_yt = preds_all, yte
            last_Xtr, last_Xte, last_ytr, last_rf = Xtr, Xte, ytr, rf

        print(f"    Fold {fold}: QL(0.5) — " +
              ", ".join(f"{m}={np.mean(res_ql[m][0.5]):.4f}" for m in METHODS))

    # ---- 4.3 Criterio de split ----
    print("\n  [3/4] Impacto del criterio de split (Sec. 4.3)...")
    split_res = {}
    tr0, te0 = next(iter(kf.split(X)))
    for crit in ["squared_error", "absolute_error"]:
        p = bp.copy()
        p["criterion"] = crit
        rf_c = RandomForestRegressor(**p, random_state=RANDOM_SEED, n_jobs=-1)
        rf_c.fit(X[tr0], y[tr0])
        lt_c = get_leaf_indices(rf_c, X[tr0])
        le_c = get_leaf_indices(rf_c, X[te0])
        split_res[crit] = {}
        for m_name, pfunc in [("QRF", None), ("RF-GAP", compute_proximity_rfgap),
                               ("OOB", compute_proximity_oob), ("ORIGINAL", compute_proximity_original)]:
            if m_name == "QRF":
                pr = qrf_predict(rf_c, X[te0], lt_c, y[tr0], QUANTILES)
            elif m_name == "ORIGINAL":
                pr = quantile_predict_prox(pfunc(lt_c, le_c), y[tr0], QUANTILES)
            else:
                pr = quantile_predict_prox(pfunc(rf_c, X[tr0], lt_c, le_c), y[tr0], QUANTILES)
            losses = {q: quantile_loss(y[te0], pr[q], q) for q in QUANTILES}
            losses["MSE"] = mse(y[te0], pr[0.5])
            losses["MAPE"] = mape(y[te0], pr[0.5])
            split_res[crit][m_name] = losses
    print("    Completado.")

    # Modelo final con todos los datos
    rf_final = RandomForestRegressor(**bp, random_state=RANDOM_SEED, n_jobs=-1)
    rf_final.fit(X, y)

    # ---- Tabla de resultados ----
    df_res = _build_table(res_ql, res_mse, res_mape)
    print(f"\n  {'=' * 80}")
    print(f"  RESULTADOS: {dataset_name}")
    print(f"  {'=' * 80}")
    print(df_res.to_string(index=False))
    df_res.to_csv(out / "results_table.csv", index=False)

    # ---- Tabla de split criteria ----
    df_split = _build_split_table(split_res)
    df_split.to_csv(out / "split_criteria_table.csv", index=False)

    # ---- 4. Graficos ----
    print(f"\n  [4/4] Generando graficos storytelling en {out}/...")
    _plot_all(res_ql, res_mse, res_mape, last_preds, last_yt,
              rf_final, feature_names, dataset_name, out,
              last_Xtr, last_Xte, last_ytr, last_rf, split_res)

    print(f"\n  Dataset {dataset_name} completado.\n")
    return {"results": df_res, "split": df_split, "params": bp}


def _build_table(res_ql, res_mse, res_mape):
    rows = []
    for m in METHODS:
        r = {"Method": m}
        for q in QUANTILES:
            r[f"QL(a={q})"] = np.mean(res_ql[m][q])
        r["MSE"] = np.mean(res_mse[m])
        r["MAPE(%)"] = np.mean(res_mape[m])
        rows.append(r)
    return pd.DataFrame(rows)


def _build_split_table(split_res):
    rows = []
    for crit, methods_data in split_res.items():
        for m, losses in methods_data.items():
            r = {"Criterion": crit, "Method": m}
            for q in QUANTILES:
                r[f"QL(a={q})"] = losses[q]
            r["MSE"] = losses["MSE"]
            r["MAPE(%)"] = losses["MAPE"]
            rows.append(r)
    return pd.DataFrame(rows)


# =====================================================================
# SECCION 5: GRAFICOS STORYTELLING
# =====================================================================

def _plot_all(res_ql, res_mse, res_mape, preds, yt,
              rf_final, fnames, ds_name, out,
              Xtr_last, Xte_last, ytr_last, rf_last, split_res):
    """Genera los 9 graficos del paper con estilo storytelling."""

    c_list = [COLORS[m] for m in METHODS]

    # --- 01. Quantile Loss comparativo (Table 2 visual) ---
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(QUANTILES))
    w = 0.18
    for i, m in enumerate(METHODS):
        vals = [np.mean(res_ql[m][q]) for q in QUANTILES]
        bars = ax.bar(x + i * w, vals, w, label=m, color=c_list[i], edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([f"α={q}" for q in QUANTILES], fontsize=11)
    ax.set_ylabel("Quantile Loss (Eq. 8)", fontsize=13)
    ax.set_title(f"Quantile Loss por Metodo y Cuantil — {ds_name}\n"
                 f"Menor es mejor | 5-Fold CV | Paper: Li et al. (2024), Eq. 8",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    _add_watermark(ax)
    fig.tight_layout()
    fig.savefig(out / "01_quantile_loss_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # --- 02. MSE y MAPE (Eq. 9 y 10) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, (metric, label, vals) in enumerate([
        ("MSE", "Mean Squared Error (Eq. 9)", [np.mean(res_mse[m]) for m in METHODS]),
        ("MAPE", "MAPE % (Eq. 10)", [np.mean(res_mape[m]) for m in METHODS])]):
        ax = axes[idx]
        bars = ax.bar(METHODS, vals, color=c_list, edgecolor="white", linewidth=0.5)
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.2f}" if metric == "MSE" else f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"{label}\nMejor: {METHODS[best_idx]} (borde rojo)", fontsize=13)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.suptitle(f"Estimacion Puntual (Mediana Condicional) — {ds_name}\n"
                 f"Paper: Li et al. (2024), Sec. 2.4", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out / "02_mse_mape_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # --- 03. Prediction Intervals RF-GAP (Fig. 1 del paper) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    med = preds["RF-GAP"][0.5]
    lo, hi = preds["RF-GAP"][0.025], preds["RF-GAP"][0.975]

    ax = axes[0]
    sidx = np.argsort(med)
    ax.scatter(med[sidx], yt[sidx], alpha=0.5, s=12, color="#FF9800", label="Observado", zorder=3)
    for idx_v in sidx[::max(1, len(sidx)//200)]:
        ax.vlines(med[idx_v], lo[idx_v], hi[idx_v], alpha=0.15, color="#2196F3", linewidth=1)
    lims = [min(yt.min(), med.min()), np.percentile(np.concatenate([yt, med]), 96)]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1.5, label="Prediccion perfecta")
    ax.set_xlabel("Fitted Values (Mediana Condicional)")
    ax.set_ylabel("Valores Observados")
    ax.set_title("(a) Fitted vs Observed + Intervalos 95%\nRF-GAP Proximity (Eq. 4)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax = axes[1]
    iw = hi - lo
    sw = np.argsort(iw)
    co = yt - med
    cl, cu = lo - med, hi - med
    xo = np.arange(len(sw))
    ax.fill_between(xo, cl[sw], cu[sw], alpha=0.35, color="#2196F3", label="Intervalo 95%")
    ax.scatter(xo, co[sw], alpha=0.5, s=6, color="#FF9800", label="Observado (centrado)", zorder=3)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    cov = np.mean((yt >= lo) & (yt <= hi)) * 100
    out_up = np.mean(yt > hi) * 100
    out_lo = np.mean(yt < lo) * 100
    ax.text(0.98, 0.97, f"↑ {out_up:.1f}%", transform=ax.transAxes, ha="right", va="top",
            fontsize=13, fontweight="bold", color="red")
    ax.text(0.02, 0.03, f"↓ {out_lo:.1f}%", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold", color="red")
    ax.set_xlabel("Muestras ordenadas por ancho de intervalo")
    ax.set_ylabel("Valores observados e intervalos")
    ax.set_title(f"(b) Intervalos ordenados por ancho | Cobertura: {cov:.1f}%\n"
                 f"\"<5% outside\" — Sec. 4.2 del paper", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")

    fig.suptitle(f"Intervalos de Prediccion al 95% — RF-GAP — {ds_name}\n"
                 f"Replicando Figura 1 del paper", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out / "03_prediction_intervals_rfgap.png", bbox_inches="tight")
    plt.close(fig)

    # --- 04. Width of Prediction Intervals (Fig. 3 del paper) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    for m in METHODS:
        w_arr = preds[m][0.975] - preds[m][0.025]
        ax.plot(np.arange(len(w_arr)), np.sort(w_arr), label=m.lower(),
                color=COLORS[m], linewidth=2, alpha=0.85)
    ax.set_xlabel("Muestras ordenadas", fontsize=13)
    ax.set_ylabel("Ancho del intervalo de prediccion", fontsize=13)
    ax.set_title(f"Ancho de Intervalos de Prediccion (95%) — {ds_name}\n"
                 f"RF-GAP produce los intervalos mas estrechos (Sec. 4.2) | Replicando Fig. 3",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    _add_watermark(ax)
    fig.tight_layout()
    fig.savefig(out / "04_width_prediction_intervals.png", bbox_inches="tight")
    plt.close(fig)

    # --- 05. Intervalos por metodo (Fig. 2 del paper) ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.ravel()
    for idx, m in enumerate(METHODS):
        ax = axes[idx]
        med_m = preds[m][0.5]
        lo_m, hi_m = preds[m][0.025], preds[m][0.975]
        iw_m = hi_m - lo_m
        sw_m = np.argsort(iw_m)
        co_m = yt - med_m
        cl_m, cu_m = lo_m - med_m, hi_m - med_m
        xo_m = np.arange(len(sw_m))

        ax.fill_between(xo_m, cl_m[sw_m], cu_m[sw_m], alpha=0.35, color="#2196F3")
        ax.scatter(xo_m, co_m[sw_m], alpha=0.45, s=5, color="#FF9800", zorder=3)
        ax.axhline(0, color="black", linestyle="--", alpha=0.3)

        cov_m = np.mean((yt >= lo_m) & (yt <= hi_m)) * 100
        out_u = np.mean(yt > hi_m) * 100
        out_l = np.mean(yt < lo_m) * 100
        ax.text(0.98, 0.97, f"↑ {out_u:.1f}%", transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold", color="red")
        ax.text(0.02, 0.03, f"↓ {out_l:.1f}%", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=12, fontweight="bold", color="red")
        ax.set_title(f"{m} — Cobertura: {cov_m:.1f}%", fontsize=13, fontweight="bold")
        ax.set_xlabel("Muestras ordenadas")
        ax.set_ylabel("Obs. e intervalos")

    fig.suptitle(f"Intervalos de Prediccion (95%) — Comparacion de 4 Metodos — {ds_name}\n"
                 f"Replicando Figura 2 del paper | \"<5% outside these intervals\" (Sec. 4.2)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "05_prediction_intervals_all_methods.png", bbox_inches="tight")
    plt.close(fig)

    # --- 06. Feature Importance ---
    imp = rf_final.feature_importances_
    si = np.argsort(imp)[::-1]
    n_feat = min(20, len(fnames))
    fig, ax = plt.subplots(figsize=(12, max(6, n_feat * 0.4)))
    bars = ax.barh(range(n_feat), imp[si[:n_feat]], color="#2196F3", edgecolor="white")
    bars[0].set_color("#FF9800")
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([fnames[i] for i in si[:n_feat]], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia (MDI — Mean Decrease Impurity)", fontsize=12)
    ax.set_title(f"Top {n_feat} Features — {ds_name}\n"
                 f"Determina la estructura de proximidades (Sec. 2.3)\n"
                 f"Feature #1 en naranja", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out / "06_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    # --- 07. Cobertura / Calibracion ---
    cov_levels = [(0.005, 0.995, 99.0), (0.025, 0.975, 95.0), (0.05, 0.95, 90.0)]
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, m in enumerate(METHODS):
        noms, acts = [], []
        for lq, uq, nom in cov_levels:
            act = np.mean((yt >= preds[m][lq]) & (yt <= preds[m][uq])) * 100
            noms.append(nom)
            acts.append(act)
        ax.plot(noms, acts, "o-", color=c_list[i], label=m, markersize=12, linewidth=2.5)
    ax.plot([85, 100], [85, 100], "k--", alpha=0.5, linewidth=1.5, label="Calibracion perfecta")
    ax.set_xlabel("Cobertura Nominal (%)", fontsize=13)
    ax.set_ylabel("Cobertura Real (%)", fontsize=13)
    ax.set_title(f"Calibracion de Intervalos de Prediccion — {ds_name}\n"
                 f"Puntos sobre la diagonal = sobre-cobertura (conservador)\n"
                 f"Paper Sec. 4.2: \"less than 5% of data falls outside\"",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out / "07_coverage_calibration.png", bbox_inches="tight")
    plt.close(fig)

    # --- 08. Distribucion Condicional (Eq. 7 — CDF) ---
    fine_q = np.arange(0.01, 1.0, 0.01).tolist()
    lt_l = get_leaf_indices(rf_last, Xtr_last)

    np.random.seed(RANDOM_SEED)
    n_sel = min(4, len(yt))
    si_obs = np.random.choice(len(yt), size=n_sel, replace=False)
    Xte_sel = Xte_last[si_obs]
    le_sel = get_leaf_indices(rf_last, Xte_sel)
    prox_fine = compute_proximity_rfgap(rf_last, Xtr_last, lt_l, le_sel)
    preds_fine = quantile_predict_prox(prox_fine, ytr_last, fine_q)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i in range(n_sel):
        ax = axes[i]
        cdf_vals = [preds_fine[q][i] for q in fine_q]
        true_v = yt[si_obs[i]]
        ax.plot(cdf_vals, fine_q, color="#2196F3", linewidth=2.5, label="CDF Condicional")
        ax.axvline(true_v, color="red", linestyle="--", linewidth=2, label=f"Real = {true_v:.1f}")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        q03 = preds_fine[0.03][i] if 0.03 in preds_fine else cdf_vals[2]
        q97 = preds_fine[0.97][i] if 0.97 in preds_fine else cdf_vals[-3]
        ax.axvspan(q03, q97, alpha=0.15, color="green", label="IC 95%")
        ax.set_xlabel("Valor objetivo", fontsize=11)
        ax.set_ylabel("P(Y \u2264 y | X=x)", fontsize=11)
        ax.set_title(f"Obs. {si_obs[i]} \u2014 Valor real: {true_v:.1f}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"Distribucion Condicional Completa F(y|X=x) — {ds_name}\n"
                 f"RF-GAP Proximity (Eq. 7) | \"estimate the entire conditional distribution\" (Abstract)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "08_conditional_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # --- 09. Split Criteria Comparison (Table 3 / Sec. 4.3) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ci, crit in enumerate(["squared_error", "absolute_error"]):
        ax = axes[ci]
        data_c = split_res[crit]
        x_c = np.arange(len(QUANTILES))
        for mi, m in enumerate(METHODS):
            vals = [data_c[m][q] for q in QUANTILES]
            ax.plot(x_c, vals, "o-", color=c_list[mi], label=m, markersize=7, linewidth=1.5)
        ax.set_xticks(x_c)
        ax.set_xticklabels([f"α={q}" for q in QUANTILES], fontsize=9)
        ax.set_ylabel("Quantile Loss")
        ax.set_title(f"Criterio: {crit.upper()}\nMSE best: "
                     f"{min(data_c.items(), key=lambda x: x[1]['MSE'])[0]}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"Impacto del Criterio de Split — {ds_name}\n"
                 f"Paper Sec. 4.3 / Table 3: \"Three distinct criteria were tested\" ",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out / "09_split_criteria_comparison.png", bbox_inches="tight")
    plt.close(fig)

    print(f"    9 graficos guardados en {out}/")


def _add_watermark(ax):
    ax.text(0.99, 0.01, "Li et al. (2024) arXiv:2408.02355",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray", alpha=0.6, style="italic")


# =====================================================================
# SECCION 6: PREPARACION DE DATOS POR DATASET
# =====================================================================

def prepare_online_retail():
    """Prepara Online Retail II — target: Quantity."""
    print("\n  Cargando Online Retail II...")
    path = DATA_DIR / "online_retail_II.xlsx"
    df1 = pd.read_excel(path, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(path, sheet_name="Year 2010-2011")
    df = pd.concat([df1, df2], ignore_index=True)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    print(f"    Raw: {len(df):,} registros")

    # Fase 1: Filtrado basico
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[~df["Invoice"].str.startswith("C")].copy()
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    df = df.dropna(subset=["Customer_ID"]).copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[df["StockCode"].astype(str).str.match(r"^\d+[A-Za-z]*$")].copy()

    # Fase 2: Outliers IQR
    for col in ["Quantity", "Price"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[df[col] <= q3 + 3 * iqr].copy()

    print(f"    Limpio: {len(df):,} registros")

    # Feature Engineering
    df["TotalAmount"] = df["Quantity"] * df["Price"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Quarter"] = df["InvoiceDate"].dt.quarter
    df["IsHighSeason"] = df["Month"].isin([11, 12]).astype(int)
    df["LogPrice"] = np.log1p(df["Price"])
    df["PriceCategory"] = pd.qcut(df["Price"], q=5, labels=[0,1,2,3,4], duplicates="drop").astype(int)
    df["IsUK"] = (df["Country"] == "United Kingdom").astype(int)
    le = LabelEncoder()
    df["CountryEncoded"] = le.fit_transform(df["Country"])
    df["CountryFreq"] = df["Country"].map(df["Country"].value_counts(normalize=True))

    for col, grp, agg in [("ProductFreq", "StockCode", "count"),
                           ("ProductAvgPrice", "StockCode", "mean"),
                           ("ProductAvgQty", "StockCode", "mean"),
                           ("ProductStdQty", "StockCode", "std")]:
        src = "Price" if "Price" in col else "Quantity"
        if agg == "count":
            df[col] = df["StockCode"].map(df["StockCode"].value_counts())
        elif agg == "std":
            df[col] = df["StockCode"].map(df.groupby("StockCode")["Quantity"].std().fillna(0))
        else:
            df[col] = df["StockCode"].map(df.groupby("StockCode")[src].mean())

    df["CustomerTxnCount"] = df["Customer_ID"].map(df.groupby("Customer_ID")["Invoice"].nunique())
    df["CustomerAvgSpend"] = df["Customer_ID"].map(df.groupby("Customer_ID")["TotalAmount"].mean())
    df["CustomerAvgQty"] = df["Customer_ID"].map(df.groupby("Customer_ID")["Quantity"].mean())
    df["InvoiceItemCount"] = df["Invoice"].map(df.groupby("Invoice")["StockCode"].nunique())
    df["InvoiceTotal"] = df["Invoice"].map(df.groupby("Invoice")["TotalAmount"].sum())

    feat_cols = [
        "Year", "Month", "DayOfWeek", "Hour", "IsWeekend", "Quarter", "IsHighSeason",
        "Price", "LogPrice", "PriceCategory",
        "CountryFreq", "IsUK", "CountryEncoded",
        "ProductFreq", "ProductAvgPrice", "ProductAvgQty", "ProductStdQty",
        "CustomerTxnCount", "CustomerAvgSpend", "CustomerAvgQty",
        "InvoiceItemCount", "InvoiceTotal",
    ]

    df_model = df[feat_cols + ["Quantity"]].dropna()
    X = df_model[feat_cols].values.astype(float)
    y = df_model["Quantity"].values.astype(float)
    return X, y, feat_cols


def prepare_energy_efficiency():
    """Prepara Energy Efficiency — target: Y1 (Heating Load)."""
    print("\n  Cargando Energy Efficiency desde OpenML...")
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="energy_efficiency", version=1, as_frame=True)
    df = data.frame
    print(f"    Shape: {df.shape}")

    y = df["Y1"].astype(float).values
    X_df = df.drop(columns=["Y1"])
    if "Y2" in X_df.columns:
        X_df = X_df.drop(columns=["Y2"])
    feat_cols = list(X_df.columns)
    X = X_df.values.astype(float)
    print(f"    Features: {feat_cols}")
    return X, y, feat_cols


def prepare_online_news():
    """Prepara Online News Popularity — target: log(1+shares)."""
    print("\n  Cargando Online News Popularity...")
    path = DATA_DIR / "OnlineNewsPopularity.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"    Raw: {df.shape}")

    # Eliminar columnas no predictivas
    df = df.drop(columns=["url", "timedelta"], errors="ignore")

    # Target: log-transform (Sec. 3.2 del paper)
    df = df[df["shares"] > 0].copy()
    y_log = np.log1p(df["shares"].values)

    # Feature engineering
    df["content_density"] = df["n_unique_tokens"] / df["n_tokens_content"].replace(0, np.nan)
    df["img_per_100w"] = (df["num_imgs"] / df["n_tokens_content"].replace(0, np.nan)) * 100
    df["video_per_100w"] = (df["num_videos"] / df["n_tokens_content"].replace(0, np.nan)) * 100
    df["total_media"] = df["num_imgs"] + df["num_videos"]
    df["self_link_ratio"] = df["num_self_hrefs"] / df["num_hrefs"].replace(0, np.nan)
    df["title_content_ratio"] = df["n_tokens_title"] / df["n_tokens_content"].replace(0, np.nan)
    df["kw_spread_max"] = df["kw_max_max"] - df["kw_min_max"]
    df["kw_avg_vs_min"] = df["kw_avg_avg"] / (df["kw_min_avg"] + 1)
    df["log_kw_max_max"] = np.log1p(df["kw_max_max"])
    df["log_kw_avg_avg"] = np.log1p(df["kw_avg_avg"])

    for c in ["self_reference_min_shares", "self_reference_max_shares", "self_reference_avg_sharess"]:
        if c in df.columns:
            df[f"log_{c.split('_')[-1]}"] = np.log1p(df[c])

    df["polarity_range"] = df["max_positive_polarity"] - df["min_negative_polarity"]
    df["sentiment_magnitude"] = df["global_sentiment_polarity"].abs()

    feat_cols = [c for c in df.columns if c != "shares"]
    df_clean = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df_clean.values.astype(float)
    y = y_log

    print(f"    Features: {len(feat_cols)}, Samples: {len(y)}")
    print(f"    Target (log1p shares): mean={y.mean():.2f}, std={y.std():.2f}")
    return X, y, feat_cols


# =====================================================================
# MAIN
# =====================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  Quantile Regression using Random Forest Proximities")
    print("  Li et al. (2024), arXiv:2408.02355")
    print("  Multi-Dataset Runner")
    print("=" * 70)

    all_results = {}

    # Dataset 1: Online Retail
    X1, y1, f1 = prepare_online_retail()
    all_results["Online Retail II"] = run_pipeline(
        X1, y1, f1, "Online Retail II (Quantity)",
        PLOTS_DIR / "online_retail", sample_size=15000)

    # Dataset 2: Energy Efficiency
    X2, y2, f2 = prepare_energy_efficiency()
    all_results["Energy Efficiency"] = run_pipeline(
        X2, y2, f2, "Energy Efficiency (Heating Load)",
        PLOTS_DIR / "energy_efficiency", sample_size=15000)

    # Dataset 3: Online News
    X3, y3, f3 = prepare_online_news()
    all_results["Online News"] = run_pipeline(
        X3, y3, f3, "Online News Popularity (log shares)",
        PLOTS_DIR / "online_news", sample_size=15000)

    # Resumen final
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  RESUMEN FINAL")
    print("=" * 70)
    for name, res in all_results.items():
        df_r = res["results"]
        best_mse = df_r.loc[df_r["MSE"].idxmin(), "Method"]
        best_mape = df_r.loc[df_r["MAPE(%)"].idxmin(), "Method"]
        print(f"\n  {name}:")
        print(f"    Mejor MSE:  {best_mse} ({df_r['MSE'].min():.4f})")
        print(f"    Mejor MAPE: {best_mape} ({df_r['MAPE(%)'].min():.2f}%)")
        print(f"    Params: {res['params']}")

    print(f"\n  Tiempo total: {elapsed/60:.1f} minutos")
    print(f"  Graficos en: {PLOTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
