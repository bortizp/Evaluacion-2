# 🚀 Guía Rápida de Inicio

## Configuración Inicial (Solo una vez)

### 1. Clonar el repositorio

```bash
git clone https://github.com/bortizp/Evaluacion-2.git
cd Evaluacion-2
```

### 2. Crear entorno virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows:
.venv\Scripts\activate

# En Mac/Linux:
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 📊 Ejecutar el Análisis

### Opción 1: VS Code (Recomendado)

1. Abre la carpeta del proyecto en VS Code
2. Abre `Limpiar_datos.ipynb`
3. Selecciona el kernel de Python (`.venv` si creaste un entorno virtual)
4. Ejecuta las celdas en orden (Ctrl + Enter o el botón ▶️)

### Opción 2: Jupyter Notebook

```bash
jupyter notebook Limpiar_datos.ipynb
```

## 📝 Orden de Ejecución

Ejecuta las celdas **en orden**:

1. **Celda 1**: Título (Markdown)
2. **Celda 2**: ⚙️ Carga de datos del CSV
3. **Celda 3**: Título (Markdown)
4. **Celda 4**: 🧹 Limpieza y filtrado de datos
5. **Celda 5**: Título (Markdown)
6. **Celda 6**: 🎯 Preparación final para modelos

## ✅ Verificación

Después de ejecutar todas las celdas, deberías ver:

```
¡DataFrame final listo para los modelos!
```

Y un DataFrame con:

- ✓ 54,176 registros (después de filtrar negativos)
- ✓ 5 columnas
- ✓ Variable de clasificación creada

## 🐛 Solución de Problemas

### Error: "FileNotFoundError"

**Causa**: El archivo CSV no está en la carpeta correcta
**Solución**: Asegúrate de que `se_facturacion_clientes_regulados(in).csv` esté en la misma carpeta que el notebook

### Error: "ModuleNotFoundError: No module named 'pandas'"

**Causa**: Pandas no está instalado
**Solución**:

```bash
pip install pandas numpy
```

### Error: Caracteres extraños en los datos

**Causa**: Problema de encoding
**Solución**: Ya está resuelto en el notebook con `encoding='latin-1'`

## 📌 Próximos Pasos

Después de limpiar los datos, puedes:

1. **Crear modelos de regresión** para predecir `energia_kwh`
2. **Crear modelos de clasificación** para predecir `Consumo_Categoria`
3. **Visualizar los datos** con matplotlib o seaborn

## 💡 Tips

- Usa `df_listo.head()` para ver las primeras filas
- Usa `df_listo.describe()` para estadísticas descriptivas
- Usa `df_listo.info()` para información de columnas y tipos de datos

## 📞 Ayuda

Si tienes problemas, revisa:

- El README completo en `README.md`
- Los comentarios dentro del notebook
- La documentación de pandas: https://pandas.pydata.org/docs/
