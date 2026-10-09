"""Semeion handwritten digits: self-written kNN, LOO and real Weka comparison.

Run: python lab1/exp_1.py --weka
Only NumPy and Matplotlib are needed for the Python experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent


def load_data(path):
    """Validate 256 binary pixels + 10 one-hot labels; never use labels as pixels."""
    raw = np.loadtxt(path, ndmin=2)
    if raw.shape[1] != 266 or len(raw) == 0:
        raise ValueError(f"{path}: expected a nonempty 266-column matrix")
    if not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
        raise ValueError(f"{path}: all pixels and labels must be finite 0/1 values")
    if not np.all(raw[:, 256:].sum(axis=1) == 1):
        raise ValueError(f"{path}: labels must be one-hot")
    return raw[:, :256], np.argmax(raw[:, 256:], axis=1)


def validate_xy(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] == 0 or len(X) == 0:
        raise ValueError("X must be a nonempty two-dimensional matrix")
    if y.shape != (len(X),) or not np.isfinite(X).all():
        raise ValueError("X and y must have matching lengths and finite features")
    if not np.isin(y, np.arange(10)).all():
        raise ValueError("labels must be integers in 0..9")
    return X, y.astype(int)


def check_k(k, n):
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
        raise ValueError(f"k must be an integer in 1..{n}")


def squared_distances(query, train):
    """Euclidean distance squared. sqrt is monotone, so the order is identical.

    Float64 prevents overflow from uint8 dot products. This reuses pairwise
    distances without changing the training set in any LOO fold.
    """
    query = np.asarray(query, dtype=np.float64)
    train = np.asarray(train, dtype=np.float64)
    if (query.ndim != 2 or train.ndim != 2 or query.shape[1] != train.shape[1]
            or not np.isfinite(query).all() or not np.isfinite(train).all()):
        raise ValueError("query and train must have matching finite feature columns")
    qnorm = np.sum(query * query, axis=1)[:, None]
    tnorm = np.sum(train * train, axis=1)[None, :]
    return np.maximum(qnorm + tnorm - 2 * query @ train.T, 0)


def predict_from_distances(distances, y, k, *, tie_break="smallest", seed=42,
                           include_boundary=False):
    """Uniform majority vote over exactly k neighbours by default.

    Stable sorting breaks equal distances by training row index. Voting ties
    choose the smallest digit, or seeded random selection. include_boundary
    includes every neighbour tied at the kth distance (for Weka diagnostics).
    """
    distances = np.asarray(distances, dtype=np.float64)
    y = np.asarray(y)
    if distances.ndim != 2 or y.shape != (distances.shape[1],):
        raise ValueError("distance columns must match training labels")
    if np.isnan(distances).any() or (distances < 0).any():
        raise ValueError("distances must be nonnegative and not NaN")
    if not np.isin(y, np.arange(10)).all():
        raise ValueError("labels must be integers in 0..9")
    check_k(k, distances.shape[1])
    if tie_break not in ("smallest", "random"):
        raise ValueError("tie_break must be smallest or random")
    order = np.argsort(distances, axis=1, kind="stable")
    chosen_distances = np.take_along_axis(distances, order[:, :k], axis=1)
    if not np.isfinite(chosen_distances).all():
        raise ValueError("each query must have at least k eligible neighbours")
    predictions = np.empty(len(distances), dtype=int)
    rng = np.random.default_rng(seed)
    for i, indices in enumerate(order):
        if include_boundary:
            boundary = chosen_distances[i, -1]
            indices = indices[distances[i, indices] <= boundary]
        else:
            indices = indices[:k]
        votes = np.bincount(y[indices].astype(int), minlength=10)
        winners = np.flatnonzero(votes == votes.max())
        predictions[i] = winners[0] if tie_break == "smallest" else rng.choice(winners)
    return predictions


def knn_predict(X_train, y_train, X_test, k=3, **kwargs):
    X_train, y_train = validate_xy(X_train, y_train)
    X_test = np.asarray(X_test, dtype=np.float64)
    distances = squared_distances(X_test, X_train)
    return predict_from_distances(distances, y_train, k, **kwargs)


def loo_predictions(X, y, k=3, **kwargs):
    X, y = validate_xy(X, y)
    check_k(k, len(X) - 1)
    distances = squared_distances(X, X)
    # Exclude only the current ROW, not other rows with identical pixels.
    np.fill_diagonal(distances, np.inf)
    return predict_from_distances(distances, y, k, **kwargs)


def loo_eval(X, y, k):
    """Interface requested by the assignment: return LOO accuracy as a float."""
    return float(np.mean(loo_predictions(X, y, k) == np.asarray(y)))


def confusion_matrix(y, prediction):
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (y, prediction), 1)
    return matrix


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_arff(path, X, y):
    """One nominal class attribute replaces the ten one-hot columns."""
    with Path(path).open("w", encoding="ascii", newline="\n") as handle:
        handle.write("@relation semeion\n\n")
        for i in range(X.shape[1]):
            handle.write(f"@attribute pixel_{i + 1} numeric\n")
        handle.write("@attribute digit {0,1,2,3,4,5,6,7,8,9}\n\n@data\n")
        for pixels, label in zip(X, y):
            handle.write(",".join(str(int(v)) for v in pixels) + f",{label}\n")


def find_java(explicit=None):
    if explicit:
        return str(Path(explicit).resolve())
    if shutil.which("java"):
        return shutil.which("java")
    if os.environ.get("JAVA_HOME"):
        candidate = Path(os.environ["JAVA_HOME"]) / "bin" / "java.exe"
        if candidate.is_file():
            return str(candidate)
    # MATLAB includes a Java runtime on this machine; use it without modifying it.
    candidates = sorted(Path("C:/Program Files/MATLAB").glob(
        "*/sys/java/jre/win64/jre/bin/java.exe"))
    if candidates:
        return str(candidates[-1])
    raise RuntimeError("Java not found; pass --java PATH or configure JAVA_HOME")


def run_weka(java, jar, arff, n, ks, output):
    """Execute actual Weka IBk, n folds = LOO (NOT IBk's -X k selection)."""
    jar = Path(jar).resolve()
    if not jar.is_file():
        raise RuntimeError("Weka jar missing; run python lab1/prepare_weka.py first")
    classpath = os.pathsep.join(str(p) for p in sorted(jar.parent.glob("*.jar")))
    (output / "weka_home").mkdir(parents=True, exist_ok=True)
    results = {}
    for k in ks:
        command = [java, "-Djava.awt.headless=true", "-Dweka.packageManager.offline=true",
                   f"-Duser.home={output / 'weka_home'}", "-cp", classpath,
                   "weka.classifiers.lazy.IBk", "-t", str(arff), "-c", "last",
                   "-x", str(n), "-s", "42", "-v", "-K", str(k)]
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, errors="replace",
                                timeout=600, check=False)
        elapsed = time.perf_counter() - started
        log = output / f"weka_k{k}.txt"
        log.write_text("COMMAND: " + subprocess.list2cmdline(command) + "\n\nSTDOUT:\n"
                       + result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
        matches = re.findall(r"Correctly Classified Instances\s+(\d+)\s+([\d.]+)\s*%",
                             result.stdout)
        total = re.findall(r"Total Number of Instances\s+(\d+)", result.stdout)
        if result.returncode or len(matches) != 1 or not total or int(total[-1]) != n:
            raise RuntimeError(f"Weka failed or unexpected evaluation output; inspect {log}")
        correct = int(matches[0][0])
        results[k] = {"correct": correct, "accuracy": correct / n, "seconds": elapsed}
        print(f"Weka IBk k={k}: {correct}/{n} = {correct / n:.6f}", flush=True)
    return results


def plot_images(X, y, indices, path, predictions=None, title="Digit samples"):
    indices = np.asarray(indices, dtype=int)[:12]
    if len(indices) == 0:
        return
    cols = min(6, len(indices))
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.7 * rows + .5), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, i in zip(axes.flat, indices):
        axis.imshow(X[i].reshape(16, 16), cmap="gray", vmin=0, vmax=1,
                    interpolation="nearest")
        label = f"row {i}: label {y[i]}"
        if predictions is not None:
            label += f" -> {predictions[i]}"
        axis.set_title(label, fontsize=9, pad=8)
    fig.suptitle(title, y=.98)
    # Reserve vertical space explicitly: axis('off') can make tight_layout
    # underestimate titles and place the next row's labels behind images.
    fig.subplots_adjust(left=.025, right=.975, bottom=.035, top=.84 if rows == 1 else .88,
                        wspace=.2, hspace=.42)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plots(output, X, y, ks, all_predictions, train_scores, holdout_scores, chosen_k,
               test_X, test_y, test_predictions, weka):
    plot_images(X, y, np.random.default_rng(42).choice(len(y), 6, replace=False),
                output / "digit_samples.png")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(ks, [np.mean(all_predictions[k] == y) * 100 for k in ks], "o-", label="All-data LOO")
    ax.plot(ks, [train_scores[k] * 100 for k in ks], "s--", label="Train-only LOO (k selection)")
    ax.plot(ks, [holdout_scores[k] * 100 for k in ks], "^:", label="Fixed test (exploratory)")
    ax.set(xlabel="k", ylabel="Accuracy (%)", xticks=ks, title="kNN accuracy vs. k")
    ax.grid(alpha=.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_k.png", dpi=180)
    plt.close(fig)
    for name, labels, predictions in [
        ("loo_k3", y, all_predictions.get(3, all_predictions[ks[0]])),
        (f"holdout_k{chosen_k}", test_y, test_predictions[chosen_k])]:
        matrix = confusion_matrix(labels, predictions)
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xlabel="Predicted digit", ylabel="True digit", xticks=range(10),
               yticks=range(10), title=name.replace("_", " "))
        fig.colorbar(image, ax=ax, shrink=.8)
        fig.tight_layout()
        fig.savefig(output / f"confusion_{name}.png", dpi=180)
        plt.close(fig)
        np.savetxt(output / f"confusion_{name}.csv", matrix, delimiter=",", fmt="%d")
    diagnostic_k = 3 if 3 in all_predictions else ks[0]
    pred = all_predictions[diagnostic_k]
    plot_images(X, y, np.flatnonzero(pred != y), output / "loo_errors.png", pred,
                f"LOO errors, k={diagnostic_k}")
    plot_images(test_X, test_y, np.flatnonzero(test_predictions[chosen_k] != test_y),
                output / "holdout_errors.png", test_predictions[chosen_k],
                f"Fixed test errors, train-selected k={chosen_k}")
    if weka:
        compare_ks = sorted(weka)
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        positions = np.arange(len(compare_ks))
        self_bars = ax.bar(positions - .18, [100 * np.mean(all_predictions[k] == y) for k in compare_ks],
                           .36, label="Self-written kNN")
        weka_bars = ax.bar(positions + .18, [100 * weka[k]["accuracy"] for k in compare_ks],
                           .36, label="Weka IBk")
        for bars in (self_bars, weka_bars):
            ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
        ax.set(xticks=positions, xticklabels=compare_ks, xlabel="k", ylabel="LOO accuracy (%)",
               title="Actual Weka comparison", ylim=(0, 100))
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output / "weka_comparison.png", dpi=180)
        plt.close(fig)


def discover_data():
    pairs = [p for p in BASE.rglob("semeion_train.txt")
             if "__MACOSX" not in p.parts and (p.parent / "semeion_test.txt").is_file()]
    if len(pairs) != 1:
        raise ValueError("Expected exactly one train/test pair; specify --train and --test")
    return pairs[0], pairs[0].parent / "semeion_test.txt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "results")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11, 13, 15])
    parser.add_argument("--tie-break", choices=["smallest", "random"], default="smallest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weka", action="store_true", help="Run real Weka IBk for k=1,3,5")
    parser.add_argument("--java", help="Java executable path")
    parser.add_argument("--weka-jar", type=Path, default=BASE / "tools/weka/weka-stable-3.8.6.jar")
    args = parser.parse_args(argv)
    if bool(args.train) != bool(args.test):
        parser.error("--train and --test must be supplied together")
    train_path, test_path = (args.train, args.test) if args.train else discover_data()
    train_X, train_y = load_data(train_path)
    test_X, test_y = load_data(test_path)
    X = np.vstack([train_X, test_X])
    y = np.concatenate([train_y, test_y])
    ks = sorted(set(args.ks) | {1, 3, 5})
    for k in ks:
        check_k(k, len(train_y) - 1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {"tie_break": args.tie_break, "seed": args.seed}
    print(f"Train: {train_X.shape}; test: {test_X.shape}; full LOO: {X.shape}", flush=True)
    distances = squared_distances(X, X)
    np.fill_diagonal(distances, np.inf)
    train_distances = distances[:len(train_y), :len(train_y)]
    test_distances = distances[len(train_y):, :len(train_y)]
    all_predictions, train_scores, test_scores, test_predictions = {}, {}, {}, {}
    timings, boundary_scores = {}, {}
    for k in ks:
        started = time.perf_counter()
        prediction = predict_from_distances(distances, y, k, **settings)
        timings[k] = time.perf_counter() - started
        all_predictions[k] = prediction
        train_prediction = predict_from_distances(train_distances, train_y, k, **settings)
        train_scores[k] = float(np.mean(train_prediction == train_y))
        test_prediction = predict_from_distances(test_distances, train_y, k, **settings)
        test_predictions[k] = test_prediction
        test_scores[k] = float(np.mean(test_prediction == test_y))
        if k in (1, 3, 5):
            # Diagnostic only; the basic algorithm still selects exactly k neighbours.
            boundary_prediction = predict_from_distances(distances, y, k, include_boundary=True)
            boundary_scores[k] = float(np.mean(boundary_prediction == y))
        print(f"k={k:2d} LOO={np.mean(prediction == y):.6f} "
              f"train LOO={train_scores[k]:.6f} fixed test={test_scores[k]:.6f}", flush=True)
    # Choose k from training-only LOO. Test labels are not used for selection.
    chosen_k = max(ks, key=lambda k: (train_scores[k], -k))
    print(f"Training-only LOO selects k={chosen_k}; test accuracy={test_scores[chosen_k]:.6f}", flush=True)
    for name, pixels, labels in [("semeion_full", X, y), ("semeion_train", train_X, train_y),
                                 ("semeion_test", test_X, test_y)]:
        write_arff(output / f"{name}.arff", pixels, labels)
    weka = run_weka(find_java(args.java), args.weka_jar, output / "semeion_full.arff",
                    len(y), [1, 3, 5], output) if args.weka else {}
    rows = [{"k": k, "loo_correct": int(np.sum(all_predictions[k] == y)), "loo_total": len(y),
             "loo_accuracy": float(np.mean(all_predictions[k] == y)),
             "train_loo_accuracy": train_scores[k], "fixed_test_accuracy": test_scores[k],
             "prediction_seconds_excluding_distance_matrix": timings[k]} for k in ks]
    write_csv(output / "metrics.csv", rows, list(rows[0]))
    comparison = [{"k": k, "self_loo_accuracy": float(np.mean(all_predictions[k] == y)),
                   "weka_loo_accuracy": weka[k]["accuracy"] if weka else "",
                   "absolute_difference_percentage_points":
                       100 * abs(float(np.mean(all_predictions[k] == y)) - weka[k]["accuracy"]) if weka else "",
                   "include_all_kth_distance_ties_accuracy": boundary_scores[k],
                   "weka_status": "completed" if weka else "not_run"} for k in (1, 3, 5)]
    write_csv(output / "weka_comparison.csv", comparison, list(comparison[0]))
    prediction_rows = [{"row_index_zero_based": i, "source": "train" if i < len(train_y) else "test",
                        "source_row_zero_based": i if i < len(train_y) else i - len(train_y),
                        "label": int(y[i]), **{f"loo_k{k}": int(all_predictions[k][i]) for k in ks}}
                       for i in range(len(y))]
    write_csv(output / "loo_predictions.csv", prediction_rows, list(prediction_rows[0]))
    test_rows = [{"test_row_zero_based": i, "label": int(test_y[i]),
                 **{f"k{k}": int(test_predictions[k][i]) for k in ks}} for i in range(len(test_y))]
    write_csv(output / "holdout_predictions.csv", test_rows, list(test_rows[0]))
    class_rows = []
    for k in (1, 3, 5):
        matrix = confusion_matrix(y, all_predictions[k])
        for digit in range(10):
            support = int(matrix[digit].sum())
            predicted = int(matrix[:, digit].sum())
            correct = int(matrix[digit, digit])
            class_rows.append({"k": k, "digit": digit, "support": support,
                               "precision": correct / predicted if predicted else 0,
                               "recall": correct / support if support else 0})
    write_csv(output / "per_class_metrics.csv", class_rows, list(class_rows[0]))
    unique_pixels, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    conflicting = [int(i) for i in np.flatnonzero(counts > 1)
                   if len(np.unique(y[inverse == i])) > 1]
    overlap = len(set(map(bytes, train_X.astype(np.uint8))) & set(map(bytes, test_X.astype(np.uint8))))
    metadata = {"train_path": str(train_path.resolve()), "test_path": str(test_path.resolve()),
                "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
                "train_samples": len(train_y), "test_samples": len(test_y), "full_samples": len(y),
                "class_counts": np.bincount(y, minlength=10).tolist(), "features": X.shape[1],
                "unique_pixel_rows": len(unique_pixels), "conflicting_duplicate_groups": conflicting,
                "train_test_identical_pixel_groups": overlap,
                "distance": "Euclidean (squared for ordering)", "vote": "uniform",
                "distance_tie_break": "original training row index",
                "vote_tie_break": args.tie_break, "seed": args.seed,
                "ks": ks, "train_selected_k": chosen_k,
                "selected_k_test_accuracy": test_scores[chosen_k],
                "weka": {str(k): v for k, v in weka.items()},
                "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
                "notes": ["All-data LOO uses the provided train and test rows together as required.",
                          "Fixed test evaluation trains only on the provided training file.",
                          "k is selected by training-only LOO, never by fixed test accuracy.",
                          "Handout example accuracies are not acceptance targets.",
                          "Weka includes all neighbours tied at the kth distance."]}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plots(output, X, y, ks, all_predictions, train_scores, test_scores, chosen_k,
               test_X, test_y, test_predictions, weka)
    print(f"Saved tables, ARFF, figures and metadata to {output}", flush=True)


if __name__ == "__main__":
    main()
