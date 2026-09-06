"""Select on training folds; evaluate the chosen model once on held-out data."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_data(seed=42):
    iris = load_iris()
    train, test = train_test_split(np.arange(len(iris.target)), test_size=0.2,
                                   stratify=iris.target, random_state=seed)
    return iris, train, test


def candidate_models(seed=42):
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "nearest_neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=4, random_state=seed),
    }


def run_experiment(output="reports", seed=42):
    iris, train, test = split_data(seed)
    models = candidate_models(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    comparison = []
    for name, model in models.items():
        scores = cross_validate(model, iris.data[train], iris.target[train], cv=cv,
                                scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"})
        comparison.append({"model": name, "cv_accuracy": float(scores["test_accuracy"].mean()),
                           "cv_accuracy_std": float(scores["test_accuracy"].std()),
                           "cv_f1_macro": float(scores["test_f1_macro"].mean())})
    winner = max((row for row in comparison if row["model"] != "majority_baseline"), key=lambda row: row["cv_f1_macro"])
    model = models[winner["model"]].fit(iris.data[train], iris.target[train])
    predicted = model.predict(iris.data[test])
    matrix = confusion_matrix(iris.target[test], predicted, labels=[0,1,2])
    baseline = models["majority_baseline"].fit(iris.data[train], iris.target[train])
    report = {"seed": seed, "dataset": "Iris (scikit-learn bundled copy)",
              "train_samples": len(train), "test_samples": len(test), "selection_metric": "training CV macro F1",
              "selected_model": winner["model"], "comparison": comparison,
              "test_accuracy": float(accuracy_score(iris.target[test], predicted)),
              "baseline_test_accuracy": float(baseline.score(iris.data[test], iris.target[test])),
              "classification_report": classification_report(iris.target[test], predicted, target_names=iris.target_names, output_dict=True, zero_division=0),
              "confusion_matrix": matrix.tolist(), "feature_names": list(iris.feature_names)}
    destination = Path(output); destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    joblib.dump({"model":model,"target_names":iris.target_names,"feature_names":iris.feature_names}, destination / "model.joblib")
    _plots(destination, iris, train, matrix, comparison)
    rows = ["# Experiment results", "", f"Seed: {seed}. Training: {len(train)} samples; held-out test: {len(test)} samples.", "",
            "Model selection uses five stratified training folds and macro F1. The test split is used only after selection.", "",
            "| Model | CV accuracy | CV std | CV macro F1 |", "| --- | ---: | ---: | ---: |"]
    rows += [f"| {x['model']} | {x['cv_accuracy']:.3f} | {x['cv_accuracy_std']:.3f} | {x['cv_f1_macro']:.3f} |" for x in comparison]
    rows += ["", f"Selected: **{winner['model']}**. Test accuracy: **{report['test_accuracy']:.3f}**; baseline: **{report['baseline_test_accuracy']:.3f}**.", "",
             "![Model comparison](model-comparison.png)", "![Confusion matrix](confusion-matrix.png)", "",
             "This tiny, balanced dataset is an introduction to the workflow. A high score here does not establish performance on field measurements or other datasets."]
    (destination / "README.md").write_text("\n".join(rows) + "\n")
    return report


def _plots(destination, iris, train, matrix, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    plt.rcParams.update({"figure.dpi":150, "axes.spines.top":False, "axes.spines.right":False})
    fig, ax = plt.subplots(figsize=(8,4.5), layout="constrained")
    ax.barh([r["model"].replace("_"," ") for r in comparison], [r["cv_accuracy"] for r in comparison],
            xerr=[r["cv_accuracy_std"] for r in comparison], color=["#9ca3af","#2563eb","#14b8a6","#8b5cf6"], capsize=4)
    ax.set(xlim=(0,1.05), xlabel="Mean accuracy ± one standard deviation", title="Five-fold comparison on the training split")
    fig.savefig(destination / "model-comparison.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,5), layout="constrained")
    ConfusionMatrixDisplay(matrix, display_labels=iris.target_names).plot(ax=ax,cmap="Blues",colorbar=False)
    ax.set_title("Held-out test predictions · 30 samples")
    fig.savefig(destination / "confusion-matrix.png");plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4.5),layout="constrained")
    for label,name in enumerate(iris.target_names):
        points=iris.data[train][iris.target[train]==label]
        ax.scatter(points[:,2],points[:,3],label=name,alpha=.8,s=28)
    ax.set(xlabel=iris.feature_names[2],ylabel=iris.feature_names[3],title="Petal measurements · training split only")
    ax.legend(frameon=False);fig.savefig(destination / "training-data.png");plt.close(fig)


def predict_measurements(artifact, measurements):
    values=np.asarray(measurements,dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Provide four positive, finite measurements in centimeters.")
    # Only load artifacts you created or trust: serialized Python objects can execute code.
    bundle=joblib.load(artifact)
    probabilities=bundle["model"].predict_proba(values.reshape(1,-1))[0]
    return {"species":str(bundle["target_names"][int(np.argmax(probabilities))]),
            "probabilities":{str(name):float(prob) for name,prob in zip(bundle["target_names"],probabilities)}}
