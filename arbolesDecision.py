import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dataLoaderCleaner


df = dataLoaderCleaner.clean_data()
print(df)

#variable objetivo
consumo = df['energia_kwh'].quantile(0.75)
df['nivel_consumo'] = np.where(df['energia_kwh'] > consumo, 1, 0)


# Incluimos 'comuna' ya que RF la puede manejar mejor que la regresion Lineal.

# Columnas a codificar incluimos 'comuna' y las otras 3
categorical_features_rf = ['region', 'comuna', 'tipo_clientes', 'tarifa']

# Convertir variables dummy
df_rf = pd.get_dummies(df, columns=categorical_features_rf, drop_first=True)

# Variables a eliminar del set final (ya no necesitamos e1/e2)
df_rf.drop(columns=['e1_kwh', 'e2_kwh'], inplace=True, errors='ignore')


print(50*"=")
print("Árbol de Decisión / Random Forest (Predicción de Consumo kWh)")

# definicion de X/Y para regresion
Y_reg_rf = df_rf['energia_kwh']
X_reg_rf = df_rf.drop(columns=['energia_kwh', 'nivel_consumo']) 

# division de datos
X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(
    X_reg_rf, Y_reg_rf, test_size=0.2, random_state=42)

# Entrenamiento del Random Forest Regressor
# Usamos pocos estimadores (n_estimators) para que el entrenamiento sea rápido.
rf_reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
rf_reg.fit(X_train_reg, Y_train_reg)

# Predicción y Evaluación
Y_pred_reg = rf_reg.predict(X_test_reg)

rmse_rf = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_reg))
r2_rf = r2_score(Y_test_reg, Y_pred_reg)

print(f"R² (Random Forest Regressor): {r2_rf:.4f}")
print(f"RMSE (Random Forest Regressor): {rmse_rf:,.2f} kWh")



print(50*"=")
print("Random Forest Classifier (Predicción Alto Consumo)")

# Definición de X/Y para Clasificación
Y_cls_rf = df_rf['nivel_consumo']
X_cls_rf = df_rf.drop(columns=['energia_kwh', 'nivel_consumo']) 

# division de datos (usando stratify para balancear las clases)
Clase_X_entrenamiento, Clase_X_prueba, Clase_Y_entrenamiento, Clase_Y_prueba = train_test_split(
    X_cls_rf, Y_cls_rf, test_size=0.2, random_state=42, stratify=Y_cls_rf)

# Entrenamiento del Random Forest Classifier
rf_cls = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
rf_cls.fit(Clase_X_entrenamiento, Clase_Y_entrenamiento)

# Predicción y Evaluación
Y_pred_cls = rf_cls.predict(Clase_X_prueba)

print("\nReporte de Clasificación:")
print(classification_report(Clase_Y_prueba, Y_pred_cls))

print(50*"=")
print("Importancia de Variables (Feature Importance)")

# Obtener la importancia de las variables del modelo de clasificación
importances = rf_cls.feature_importances_
feature_names = X_cls_rf.columns
forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Mostrar solo el top 10
print(forest_importances.head(10))

# Generar un gráfico
plt.figure(figsize=(10, 6))
sns.barplot(x=forest_importances.head(10).values, y=forest_importances.head(10).index)
plt.title("Top 10 Variables más Importantes (Random Forest)")
plt.xlabel("Importancia de la Característica")
plt.ylabel("Característica")
plt.tight_layout()
plt.show()