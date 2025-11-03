import dataLoaderCleaner
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Cargar y preparar datos
df = dataLoaderCleaner.clean_data()
X = pd.concat([
    df[['clientes_facturados', 'e1_kwh', 'e2_kwh']],
    pd.get_dummies(df[['region', 'tipo_clientes', 'tarifa']], drop_first=True)
], axis=1)
y = df['energia_kwh']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar y evaluar
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Resultados
print("\n" + "=" * 50)
print("RESULTADOS DEL MODELO DE REGRESIÓN LINEAL")
print("=" * 50)
print(f"Dataset: {df.shape[0]} filas, {X.shape[1]} características")
print(f"Train/Test: {X_train.shape[0]} / {X_test.shape[0]} muestras")
print(f"\nR² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"Intercepto: {model.intercept_:.2f}")
print("=" * 50)

# Gráfico simple
print("\nGenerando gráfico...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=15)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales (kWh)', fontsize=12)
plt.ylabel('Valores Predichos (kWh)', fontsize=12)
plt.title('Regresión Lineal: Predicción de Energía kWh', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regresion_lineal.png', dpi=300)
print("✓ Gráfico guardado: regresion_lineal.png")
plt.show()
