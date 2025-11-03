# Evaluación 2 - Análisis de Consumo Energético

## 📋 Descripción del Proyecto

Este proyecto analiza datos de facturación de clientes regulados del sector energético en Chile, enfocándose específicamente en clientes residenciales para predecir patrones de consumo eléctrico.

## 🎯 Objetivo

Preparar y limpiar un dataset de consumo energético para posteriormente aplicar modelos de:

- **Regresión**: Predecir el consumo de energía (kWh)
- **Clasificación**: Categorizar el consumo en niveles (Bajo, Medio, Alto)

## 📊 Dataset

- **Archivo**: `se_facturacion_clientes_regulados(in).csv`
- **Registros totales**: 490,758
- **Período**: 2015-2024 (10 años)
- **Tipos de clientes**:
  - Residencial: 54,260 registros (11.06%)
  - No Residencial: 436,498 registros (88.94%)

## 🛠️ Dependencias

```bash
pip install pandas numpy
```

O si usas el entorno virtual del proyecto:

```bash
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Paquetes necesarios:

- `pandas` >= 2.0.0
- `numpy` >= 1.24.0

## 📓 Notebook de Limpieza

### `Limpiar_datos.ipynb`

Este notebook contiene 3 pasos principales:

#### **Paso 1: Cargar los Datos**

- Carga el archivo CSV con delimitador `;`
- Usa encoding `latin-1` para caracteres especiales
- Maneja errores de archivo no encontrado

#### **Paso 2: Limpieza y Filtrado**

- Filtra solo clientes de tipo "Residencial"
- Selecciona columnas relevantes: año, mes, comuna, energía
- Renombra columnas con caracteres especiales (BOM)
- Filtra valores negativos de energía
- **Resultado**: 54,260 registros limpios

#### **Paso 3: Preparación para Modelos**

- Convierte energía a tipo numérico
- Elimina valores nulos
- Crea variable categórica de consumo (Bajo/Medio/Alto)
- **Output**: DataFrame `df_listo` listo para modelado

## 🚀 Uso

1. **Clona el repositorio**:

```bash
git clone https://github.com/bortizp/Evaluacion-2.git
cd Evaluacion-2
```

2. **Asegúrate de tener el archivo CSV** en la misma carpeta que el notebook

3. **Abre el notebook**:

```bash
jupyter notebook Limpiar_datos.ipynb
```

O simplemente ábrelo en VS Code

4. **Ejecuta las celdas en orden**:
   - Celda 2: Carga de datos
   - Celda 4: Limpieza y filtrado
   - Celda 6: Preparación final

## 📈 Resultado Final

El DataFrame procesado (`df_listo`) contiene:

- **54,260 registros** de clientes residenciales
- **5 columnas**:
  - `anio`: Año (2015-2024)
  - `mes`: Mes (1-12)
  - `comuna`: Comuna de Chile
  - `energia_kwh`: Consumo en kWh (variable objetivo para regresión)
  - `Consumo_Categoria`: Categoría de consumo (Bajo/Medio/Alto) para clasificación
- **Sin valores nulos**
- **Sin valores negativos de energía**
- **Tipos de datos validados**

## 📝 Notas Importantes

- El dataset original contiene algunos valores negativos de energía que son filtrados automáticamente
- La columna de año tiene caracteres BOM (Byte Order Mark) que son manejados correctamente
- Los datos de clientes no residenciales son excluidos del análisis

## 👤 Autor

- **GitHub**: [@bortizp](https://github.com/bortizp)
- **Proyecto**: Evaluación 2
- **Fecha**: Noviembre 2025

## 📄 Licencia

Este proyecto es parte de una evaluación académica.
