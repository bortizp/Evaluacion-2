# clasificacion.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_absolute_error, mean_squared_error
)

# -----------------------------
# Utilidades
# -----------------------------
def ensure_outdir(path="out"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def build_feature_sets(df: pd.DataFrame):
    """
    Define columnas para modelado. Ajusta aquí si agregas/quitas variables.
    """
    # Posibles columnas disponibles en tu CSV (ajustables)
    cat_cols = []
    for c in ["region", "comuna", "mes"]:
        if c in df.columns:
            cat_cols.append(c)

    num_cols = [c for c in ["anio", "clientes_facturados", "e1_kwh", "e2_kwh"]
                if c in df.columns]

    # Variable objetivo de regresión
    y_reg = df["energia_kwh"].astype(float)

    # Construcción de etiqueta de clasificación (Bajo/Medio/Alto por cuantiles)
    # Nota: si ya tienes una etiqueta pre-creada, reemplaza esta sección
    etiquetas = pd.qcut(y_reg, q=3, labels=["Bajo", "Medio", "Alto"])
    df = df.copy()
    df["Consumo_Categoria"] = etiquetas
    y_clf = df["Consumo_Categoria"]

    # X común
    X = df[cat_cols + num_cols].copy()

    return X, y_reg, y_clf, cat_cols, num_cols

def make_preprocess(cat_cols, num_cols, for_linear_model=True):
    """
    Preprocesamiento:
      - Categóricas -> OneHotEncoder
      - Numéricas    -> StandardScaler (lo dejamos también para RF sin perjuicio)
    """
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")

# -----------------------------
# Clasificación
# -----------------------------
def train_classifiers(X, y, cat_cols, num_cols, outdir="out", random_state=42):
    preprocess = make_preprocess(cat_cols, num_cols, for_linear_model=True)

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Modelos
    clf_log = Pipeline(steps=[
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=200, n_jobs=None))
    ])

    clf_rf = Pipeline(steps=[
        ("prep", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state
        ))
    ])

    # Entrenar
    clf_log.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)

    # Evaluación
    results = {}
    for name, model in [("logistic", clf_log), ("rf_clf", clf_rf)]:
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "y_pred": y_pred,
            "y_test": y_test
        }

    # Matriz de confusión del mejor por F1
    best_name = max(results.keys(), key=lambda k: results[k]["f1_macro"])
    best_model = {"logistic": clf_log, "rf_clf": clf_rf}[best_name]
    cm = confusion_matrix(results[best_name]["y_test"], results[best_name]["y_pred"], labels=["Bajo","Medio","Alto"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bajo","Medio","Alto"])
    disp.plot(values_format="d")
    plt.title(f"Matriz de Confusión ({best_name})")
    out_png = os.path.join(outdir, "grafico_matriz_confusion.png")
    plt.savefig(out_png, bbox_inches="tight"); plt.close()
    return results, best_name

# -----------------------------
# Regresión
# -----------------------------
def train_regressors(X, y, cat_cols, num_cols, outdir="out", random_state=42):
    preprocess_linear = make_preprocess(cat_cols, num_cols, for_linear_model=True)
    preprocess_tree   = make_preprocess(cat_cols, num_cols, for_linear_model=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    reg_lin = Pipeline(steps=[
        ("prep", preprocess_linear),
        ("model", LinearRegression())
    ])

    reg_rf = Pipeline(steps=[
        ("prep", preprocess_tree),
        ("model", RandomForestRegressor(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state
        ))
    ])

    reg_lin.fit(X_train, y_train)
    reg_rf.fit(X_train, y_train)

    # Predicciones
    y_pred_lin = reg_lin.predict(X_test)
    y_pred_rf  = reg_rf.predict(X_test)

    # Métricas
    def metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": rmse
        }

    res_lin = metrics(y_test, y_pred_lin)
    res_rf  = metrics(y_test, y_pred_rf)

    # Gráfico real vs pred (mejor modelo por RMSE)
    best_reg_name, best_pred = ("rf_reg", y_pred_rf) if res_rf["rmse"] <= res_lin["rmse"] else ("linear", y_pred_lin)
    plt.figure()
    plt.scatter(y_test, best_pred, alpha=.5, s=10)
    lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    plt.plot(lims, lims)  # línea y=x
    plt.xlabel("Real (kWh)"); plt.ylabel("Predicho (kWh)")
    plt.title(f"Regresión – Real vs Predicho ({best_reg_name})")
    out_png = os.path.join(outdir, "grafico_regresion_real_vs_pred.png")
    plt.savefig(out_png, bbox_inches="tight"); plt.close()

    # Importancia de variables (solo RF)
    try:
        # Recuperar nombres tras OneHotEncoder
        ct: ColumnTransformer = reg_rf.named_steps["prep"]
        rf: RandomForestRegressor = reg_rf.named_steps["model"]

        feature_names = []
        for name, trans, cols in ct.transformers_:
            if name == "cat":
                ohe: OneHotEncoder = trans
                feature_names.extend(list(ohe.get_feature_names_out(cols)))
            elif name == "num":
                feature_names.extend(cols)

        importances = rf.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(20)

        plt.figure()
        imp_df[::-1].plot(kind="barh", x="feature", y="importance", legend=False)
        plt.title("Importancia de Variables (Random Forest)")
        plt.xlabel("Importancia")
        out_png = os.path.join(outdir, "grafico_importancia_variables.png")
        plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()
    except Exception:
        pass

    # Comparación simple (tabla/figura)
    comp = pd.DataFrame([
        {"modelo": "Regresión Lineal", "R2": res_lin["r2"], "MAE": res_lin["mae"], "RMSE": res_lin["rmse"]},
        {"modelo": "Random Forest (Reg)", "R2": res_rf["r2"], "MAE": res_rf["mae"], "RMSE": res_rf["rmse"]},
    ])
    comp_plot = comp.set_index("modelo")

    plt.figure()
    comp_plot[["R2"]].plot(kind="bar")
    plt.title("Comparación de Modelos (R²)")
    out_png = os.path.join(outdir, "grafico_comparacion_modelos.png")
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()

    return {"linear": res_lin, "rf_reg": res_rf}, pd.DataFrame({"y_real": y_test, "y_pred": best_pred})

# -----------------------------
# Runner principal
# -----------------------------
def run_all(df: pd.DataFrame):
    outdir = ensure_outdir("out")

    # 1) Selección de features y targets
    X, y_reg, y_clf, cat_cols, num_cols = build_feature_sets(df)

    # 2) Clasificación
    clf_results, best_clf = train_classifiers(X, y_clf, cat_cols, num_cols, outdir=outdir)

    # 3) Regresión
    reg_results, df_preds = train_regressors(X, y_reg, cat_cols, num_cols, outdir=outdir)

    # 4) Exportar predicciones
    df_preds.to_csv(os.path.join(outdir, "resultados_predicciones.csv"), index=False)

    summary = {
        "clasificacion": clf_results,
        "mejor_clasificador": best_clf,
        "regresion": reg_results
    }
    return summary

if __name__ == "__main__":
    # Permite ejecutar directo: python clasificacion.py (si ya tienes un DF limpio)
    from dataLoaderCleaner import clean_data
    df = clean_data()
    print(df['region'].unique())
    print(df)
    print(df.head())
    print(df.info())
    print(df.describe())
    resumen = run_all(df)
    print("\n=== RESUMEN ===")
    for m, r in resumen["clasificacion"].items():
        print(f"[CLF:{m}] acc={r['accuracy']:.3f} prec={r['precision_macro']:.3f} rec={r['recall_macro']:.3f} f1={r['f1_macro']:.3f}")
    for m, r in resumen["regresion"].items():
        print(f"[REG:{m}] R2={r['r2']:.3f} MAE={r['mae']:.2f} RMSE={r['rmse']:.2f}")
    print(f"Mejor clasificador: {resumen['mejor_clasificador']}")
