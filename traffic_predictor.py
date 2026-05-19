"""
traffic_predictor.py
====================
MODULE 2: ML Core Engine & Evaluation
======================================
AI Multi-Agent Traffic Simulation System — University Lab Final Project

Core AI Concepts Demonstrated:
  - Supervised Classification (MLP / KNN from Scikit-Learn)
  - Multi-Layer Perceptron: forward propagation, backpropagation, hidden layers,
    activation functions (ReLU), and iterative weight updates via SGD/Adam.
  - K-Nearest Neighbors: non-parametric instance-based learning, distance metric,
    majority-vote class assignment.
  - Stratified K-Fold cross-validation via 5 independent random splits.
  - Per-split StandardScaler (no data leakage between train / test).
  - Evaluation: Accuracy, Precision (macro), Recall (macro), F1 (macro).
  - Output formatted as "Mean ± Std Dev" as required by grading rubric.
  - Confusion matrix & learning-curve plots saved as PNG assets.
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH: str = "traffic_clean.csv"
TEST_SIZE: float = 0.20          # 80 / 20 train-test split
N_RUNS: int = 5                  # five independent random seeds (grading requirement)
BASE_SEEDS: list[int] = [42, 7, 13, 99, 2024]
TARGET_COL: str = "Congestion_Level"
CLASS_NAMES: list[str] = ["Free Flow", "Moderate", "Heavy"]

# Evaluation asset output paths
CM_PATH: str = "confusion_matrix.png"
CURVE_PATH: str = "training_curves.png"


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------
def build_model(model_type: str = "MLP") -> object:
    """
    Instantiate the chosen Scikit-Learn classifier.

    Supported model_type values:
      "MLP"  — Multi-Layer Perceptron (neural network)
              Architecture: input → 128 → 64 → 32 → output (softmax-equivalent)
              Activation : ReLU (rectified linear unit)
              Optimiser  : Adam (adaptive moment estimation)
              Rationale  : deep enough to capture non-linear congestion patterns
                           while remaining tractable on 50 k records.

      "KNN"  — K-Nearest Neighbours (k=7, distance-weighted)
              Rationale  : k=7 balances bias-variance trade-off; distance
                           weighting gives closer neighbours more influence.

    Args:
        model_type: "MLP" or "KNN".

    Returns:
        Unfitted Scikit-Learn estimator.
    """
    model_type = model_type.upper()
    if model_type == "MLP":
        log.info("Building MLPClassifier: layers=(128,64,32), activation=relu, solver=adam")
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,               # L2 regularisation weight
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )
    elif model_type == "KNN":
        log.info("Building KNeighborsClassifier: k=7, weights=distance, metric=minkowski")
        return KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",       # closer neighbours count more
            algorithm="ball_tree",    # efficient for moderate-dimensionality data
            metric="minkowski",
            p=2,                      # Euclidean distance
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type='{model_type}'. Choose 'MLP' or 'KNN'.")


# ---------------------------------------------------------------------------
# Evaluation Loop — 5 independent train/test splits
# ---------------------------------------------------------------------------
def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "MLP",
) -> dict:
    """
    Execute 5 independent train/test splits, train a fresh model on each,
    record classification metrics, and return aggregated statistics.

    Why 5 runs?
      A single train/test split is sensitive to how data happens to be divided.
      Repeating with different random seeds gives a more reliable estimate of
      true generalisation performance and captures variance across partitions.

    Args:
        X: Feature matrix (already preprocessed, NOT yet scaled per-split).
        y: Target label vector.
        model_type: "MLP" or "KNN".

    Returns:
        dict with keys: metrics_per_run, means, stds, last_cm,
                        last_model, last_X_test, last_y_test, last_y_pred
    """
    log.info("=" * 60)
    log.info("Starting %d-run evaluation for model: %s", N_RUNS, model_type)
    log.info("=" * 60)

    results: dict[str, list[float]] = {
        "accuracy": [], "precision": [], "recall": [], "f1": [],
    }
    last_cm     = None
    last_model  = None
    last_X_test = None
    last_y_test = None
    last_y_pred = None

    for run_idx, seed in enumerate(BASE_SEEDS, start=1):
        log.info("--- Run %d / %d  (random_state=%d) ---", run_idx, N_RUNS, seed)

        # Split — stratify ensures class proportions are preserved in both splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=y,
        )

        # Per-split StandardScaler (fit on train ONLY → no data leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Build & train model
        model = build_model(model_type)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results["accuracy"].append(acc)
        results["precision"].append(prec)
        results["recall"].append(rec)
        results["f1"].append(f1)

        log.info(
            "  Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f",
            acc, prec, rec, f1,
        )

        # Keep last run's artefacts for confusion matrix plot
        last_cm     = confusion_matrix(y_test, y_pred)
        last_model  = model
        last_X_test = X_test_scaled
        last_y_test = y_test
        last_y_pred = y_pred

    # Aggregate
    means = {k: float(np.mean(v)) for k, v in results.items()}
    stds  = {k: float(np.std(v))  for k, v in results.items()}

    return {
        "metrics_per_run": results,
        "means": means,
        "stds": stds,
        "last_cm": last_cm,
        "last_model": last_model,
        "last_X_test": last_X_test,
        "last_y_test": last_y_test,
        "last_y_pred": last_y_pred,
    }


# ---------------------------------------------------------------------------
# Results Printer — grading-sheet format
# ---------------------------------------------------------------------------
def print_results(means: dict, stds: dict, model_type: str) -> str:
    """
    Format evaluation results exactly as required by the grading rubric:
      "Mean Metric Value ± Standard Deviation"

    Args:
        means: Dict of metric → mean value.
        stds:  Dict of metric → std dev value.
        model_type: Model name string.

    Returns:
        Formatted multi-line results string.
    """
    header = f"\n{'='*55}\n  EVALUATION RESULTS — {model_type}  ({N_RUNS} Independent Runs)\n{'='*55}"
    lines = [header]
    metric_display = {
        "accuracy":  "Accuracy ",
        "precision": "Precision",
        "recall":    "Recall   ",
        "f1":        "F1-Score ",
    }
    for key, label in metric_display.items():
        line = f"  {label} : {means[key]:.4f} ± {stds[key]:.4f}"
        lines.append(line)
    lines.append("=" * 55)
    output = "\n".join(lines)
    log.info(output)
    return output


# ---------------------------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, model_type: str, save_path: str = CM_PATH) -> str:
    """
    Renders and saves a styled confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (n_classes × n_classes).
        model_type: Title label string.
        save_path: File path for the PNG output.

    Returns:
        Absolute path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_type}\n(Final Run)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved → '%s'", save_path)
    return os.path.abspath(save_path)


# ---------------------------------------------------------------------------
# Per-Run Metric Curves Plot
# ---------------------------------------------------------------------------
def plot_metric_curves(
    metrics_per_run: dict,
    means: dict,
    stds: dict,
    model_type: str,
    save_path: str = CURVE_PATH,
) -> str:
    """
    Plots all four metrics across the 5 runs as line charts with a
    mean ± std shaded band for each metric.

    Args:
        metrics_per_run: Dict metric → list of per-run values.
        means: Aggregate means.
        stds:  Aggregate standard deviations.
        model_type: Label for chart title.
        save_path: PNG output path.

    Returns:
        Absolute path to saved PNG.
    """
    metric_labels = {
        "accuracy":  "Accuracy",
        "precision": "Precision (macro)",
        "recall":    "Recall (macro)",
        "f1":        "F1-Score (macro)",
    }
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    runs   = list(range(1, N_RUNS + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(metric_labels.items()):
        ax  = axes[idx]
        vals = metrics_per_run[key]
        m, s = means[key], stds[key]

        ax.plot(runs, vals, marker="o", linewidth=2, color=colors[idx], label="Per-run value")
        ax.axhline(m, color=colors[idx], linestyle="--", linewidth=1.5, label=f"Mean={m:.4f}")
        ax.fill_between(runs, m - s, m + s, alpha=0.15, color=colors[idx], label=f"±1 SD={s:.4f}")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Run #")
        ax.set_ylabel("Score")
        ax.set_xticks(runs)
        ax.set_ylim(max(0, m - 4 * s - 0.05), min(1.05, m + 4 * s + 0.05))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Per-Run Evaluation Curves — {model_type}  ({N_RUNS} Independent Splits)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Training curves saved → '%s'", save_path)
    return os.path.abspath(save_path)


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the preprocessed CSV produced by Module 1.

    NOTE: The CSV already has one-hot columns and scaled numeric features.
    We load as-is and separate features from the target column.

    Args:
        path: CSV file path.

    Returns:
        (X, y) as NumPy arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run data_pipeline.py first."
        )
    df = pd.read_csv(path)
    log.info("Loaded dataset from '%s'. Shape: %s", path, df.shape)

    y = df[TARGET_COL].values.astype(int)
    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    log.info("Feature matrix: %s | Label vector: %s", X.shape, y.shape)
    log.info("Class distribution — 0:%d  1:%d  2:%d",
             (y == 0).sum(), (y == 1).sum(), (y == 2).sum())
    return X, y


# ---------------------------------------------------------------------------
# Public API — used by app.py
# ---------------------------------------------------------------------------
def run_full_evaluation(model_type: str = "MLP") -> dict:
    """
    End-to-end entry point for Module 2:
      1. Load preprocessed data.
      2. Run 5-seed evaluation loop.
      3. Print grading-sheet formatted results.
      4. Save confusion matrix + metric curve PNGs.

    Args:
        model_type: "MLP" or "KNN".

    Returns:
        Full evaluation result dict (see evaluate_model docstring).
    """
    X, y = load_data()
    result = evaluate_model(X, y, model_type=model_type)

    formatted = print_results(result["means"], result["stds"], model_type)
    result["formatted_results"] = formatted

    # Classification report for the final run
    report = classification_report(
        result["last_y_test"],
        result["last_y_pred"],
        target_names=CLASS_NAMES,
        digits=4,
    )
    log.info("\nFull Classification Report (last run):\n%s", report)
    result["classification_report"] = report

    # Plot assets
    result["cm_path"]    = plot_confusion_matrix(result["last_cm"], model_type)
    result["curve_path"] = plot_metric_curves(
        result["metrics_per_run"], result["means"], result["stds"], model_type
    )

    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("MODULE 2 — ML Core Engine & Evaluation")
    log.info("=" * 60)

    # Default to MLP; swap to "KNN" to compare
    MODEL_TYPE = os.environ.get("MODEL_TYPE", "MLP")
    result = run_full_evaluation(model_type=MODEL_TYPE)

    print("\n" + result["formatted_results"])
    print("\nFull Classification Report (final run):\n" + result["classification_report"])
    print(f"\nAssets saved:\n  Confusion Matrix → {result['cm_path']}")
    print(f"  Metric Curves   → {result['curve_path']}")
