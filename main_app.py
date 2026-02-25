import itertools
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC


st.set_page_config(page_title="Iris Clasificación", layout="wide")


@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    class_names = iris.target_names
    feature_names = list(X.columns)
    return X, y, class_names, feature_names


def get_base_models(random_state: int) -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state
        ),
    }


def maybe_pipeline(model, use_scaling: bool):
    if use_scaling:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, classes) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-score (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        y_score = decision

    if y_score is not None and y_score.shape[1] == len(classes):
        y_test_bin = label_binarize(y_test, classes=classes)
        try:
            metrics["ROC AUC (OvR)"] = roc_auc_score(
                y_test_bin, y_score, average="macro", multi_class="ovr"
            )
        except ValueError:
            metrics["ROC AUC (OvR)"] = np.nan
    else:
        y_test_bin = None
        metrics["ROC AUC (OvR)"] = np.nan

    return {
        "model": model,
        "y_pred": y_pred,
        "y_score": y_score,
        "y_test_bin": y_test_bin,
        "metrics": metrics,
    }


def plot_confusion(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    return fig


def plot_multiclass_roc(y_test_bin, y_score, class_names, title):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        score_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={score_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    return fig


def plot_multiclass_pr(y_test_bin, y_score, class_names, title):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        ax.plot(recall, precision, label=f"{class_name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.2)
    return fig


def plot_decision_boundary(
    model,
    use_scaling: bool,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    class_names: List[str],
    feature_pair: Tuple[str, str],
    test_size: float,
    random_state: int,
    title: str,
):
    x1_name, x2_name = feature_pair
    X_pair = X[[x1_name, x2_name]].values
    y_values = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_pair, y_values, test_size=test_size, random_state=random_state, stratify=y_values
    )

    db_model = maybe_pipeline(clone(model), use_scaling)
    db_model.fit(X_train, y_train)

    x_min, x_max = X_pair[:, 0].min() - 0.5, X_pair[:, 0].max() + 0.5
    y_min, y_max = X_pair[:, 1].min() - 0.5, X_pair[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 350),
        np.linspace(y_min, y_max, 350),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = db_model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    contour = ax.contourf(xx, yy, Z, alpha=0.28, cmap=plt.cm.Set2)
    _ = contour
    sns.scatterplot(
        x=X_train[:, 0],
        y=X_train[:, 1],
        hue=[class_names[idx] for idx in y_train],
        palette="Set2",
        alpha=0.45,
        marker="o",
        legend=False,
        ax=ax,
    )
    sns.scatterplot(
        x=X_test[:, 0],
        y=X_test[:, 1],
        hue=[class_names[idx] for idx in y_test],
        palette="Set1",
        marker="X",
        s=90,
        edgecolor="black",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel(x1_name)
    ax.set_ylabel(x2_name)
    ax.set_title(title)
    return fig


st.title("Clasificación Iris con Múltiples Modelos")
st.caption(
    "Selecciona modelos, métricas y visualizaciones para comparar desempeño y fronteras de decisión."
)

X, y, class_names, feature_names = load_data()
classes = np.unique(y)

with st.sidebar:
    st.header("Configuración")
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42)
    test_size = st.slider("Proporción de test", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    use_scaling = st.checkbox("Estandarizar variables", value=True)

    models_catalog = get_base_models(random_state=random_state)
    selected_model_names = st.multiselect(
        "Modelos",
        list(models_catalog.keys()),
        default=["Logistic Regression", "KNN", "SVM (RBF)"],
    )

    metric_options = [
        "Accuracy",
        "Precision (macro)",
        "Recall (macro)",
        "F1-score (macro)",
        "ROC AUC (OvR)",
    ]
    selected_metrics = st.multiselect(
        "Métricas a mostrar",
        metric_options,
        default=["Accuracy", "F1-score (macro)", "ROC AUC (OvR)"],
    )

    viz_options = [
        "Comparación de métricas",
        "Matriz de confusión",
        "Curva ROC",
        "Curva Precision-Recall",
        "Frontera de decisión",
    ]
    selected_visualizations = st.multiselect(
        "Visualizaciones",
        viz_options,
        default=["Comparación de métricas", "Matriz de confusión", "Curva ROC", "Frontera de decisión"],
    )

    feature_pairs = list(itertools.combinations(feature_names, 2))
    selected_pair = st.selectbox(
        "Par de variables (frontera de decisión)",
        feature_pairs,
        format_func=lambda p: f"{p[0]} vs {p[1]}",
    )

if not selected_model_names:
    st.warning("Selecciona al menos un modelo para continuar.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
)

results = {}
rows = []

for model_name in selected_model_names:
    estimator = maybe_pipeline(clone(models_catalog[model_name]), use_scaling=use_scaling)
    evaluation = evaluate_model(estimator, X_train, X_test, y_train, y_test, classes=classes)
    results[model_name] = evaluation
    row = {"Modelo": model_name}
    row.update(evaluation["metrics"])
    rows.append(row)

results_df = pd.DataFrame(rows).set_index("Modelo")

st.subheader("Resumen de desempeño")
if selected_metrics:
    display_df = results_df[selected_metrics].sort_values(
        by=selected_metrics[0], ascending=False
    )
else:
    display_df = results_df

st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True)

if "Comparación de métricas" in selected_visualizations and selected_metrics:
    st.subheader("Comparación de métricas")
    metrics_long = display_df.reset_index().melt(
        id_vars="Modelo", value_vars=selected_metrics, var_name="Métrica", value_name="Valor"
    )
    fig_bar, ax_bar = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=metrics_long, x="Métrica", y="Valor", hue="Modelo", ax=ax_bar)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Comparación por modelo")
    ax_bar.grid(axis="y", alpha=0.2)
    ax_bar.legend(loc="lower right", fontsize=8)
    st.pyplot(fig_bar)

for model_name in selected_model_names:
    st.markdown(f"### {model_name}")
    model_result = results[model_name]

    col1, col2 = st.columns(2)
    with col1:
        st.write("Métricas")
        model_metrics = pd.DataFrame([model_result["metrics"]], index=[model_name])
        if selected_metrics:
            st.dataframe(
                model_metrics[selected_metrics].style.format("{:.4f}"),
                use_container_width=True,
            )
        else:
            st.dataframe(model_metrics.style.format("{:.4f}"), use_container_width=True)
    with col2:
        st.write("Distribución de predicciones")
        pred_counts = pd.Series(model_result["y_pred"]).value_counts().sort_index()
        pred_labels = [class_names[idx] for idx in pred_counts.index]
        fig_pred, ax_pred = plt.subplots(figsize=(4.8, 3.4))
        ax_pred.bar(pred_labels, pred_counts.values, color="#3A7CA5")
        ax_pred.set_ylabel("Cantidad")
        ax_pred.set_title("Predicciones en test")
        ax_pred.grid(axis="y", alpha=0.2)
        st.pyplot(fig_pred)

    if "Matriz de confusión" in selected_visualizations:
        fig_cm = plot_confusion(
            y_test, model_result["y_pred"], class_names, f"Matriz de confusión - {model_name}"
        )
        st.pyplot(fig_cm)

    if (
        "Curva ROC" in selected_visualizations
        and model_result["y_score"] is not None
        and model_result["y_test_bin"] is not None
    ):
        fig_roc = plot_multiclass_roc(
            model_result["y_test_bin"],
            model_result["y_score"],
            class_names,
            f"ROC multiclase - {model_name}",
        )
        st.pyplot(fig_roc)

    if (
        "Curva Precision-Recall" in selected_visualizations
        and model_result["y_score"] is not None
        and model_result["y_test_bin"] is not None
    ):
        fig_pr = plot_multiclass_pr(
            model_result["y_test_bin"],
            model_result["y_score"],
            class_names,
            f"Precision-Recall multiclase - {model_name}",
        )
        st.pyplot(fig_pr)

    if "Frontera de decisión" in selected_visualizations:
        fig_db = plot_decision_boundary(
            model=models_catalog[model_name],
            use_scaling=use_scaling,
            X=X,
            y=y,
            feature_names=feature_names,
            class_names=class_names,
            feature_pair=selected_pair,
            test_size=test_size,
            random_state=int(random_state),
            title=f"Frontera de decisión ({selected_pair[0]} vs {selected_pair[1]}) - {model_name}",
        )
        st.pyplot(fig_db)
