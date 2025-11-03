# 🎤 Guía para la Presentación del Proyecto

## 📋 Checklist Pre-Presentación

### Antes del día de la presentación:

- [ ] Ejecutar TODAS las celdas del notebook `Modelos_ML.ipynb` en orden
- [ ] Verificar que se generaron los 4 gráficos PNG
- [ ] Crear la presentación PowerPoint con los resultados
- [ ] Practicar la presentación (MÁXIMO 10 minutos)
- [ ] Asegurarse de que los 3 integrantes dominen TODO el proyecto

---

## 🎯 Estructura de la Presentación (10 minutos)

### Diapositiva 1: Portada (30 segundos)

- Título: "Predicción de Consumo Energético en Clientes Residenciales"
- Nombres del equipo
- Fecha: 3 de noviembre de 2025

### Diapositiva 2: El Problema (1 minuto)

**Qué decir:**

> "El consumo de energía eléctrica en Chile es un tema contingente. Los costos de la luz han aumentado y es importante entender qué factores influyen en el consumo para tomar decisiones informadas. Nuestro proyecto busca predecir el consumo de energía y clasificar a los clientes según su nivel de uso."

**Incluir:**

- Contexto: Costos de servicios básicos en Chile
- Relevancia: Políticas de eficiencia energética

### Diapositiva 3: Fuente de Datos (1 minuto)

**Qué decir:**

> "Utilizamos datos reales de la Superintendencia de Electricidad y Combustibles de Chile. El dataset contiene 54,176 registros de clientes residenciales desde 2015 hasta 2024, con información de año, mes, comuna y consumo en kWh."

**Incluir:**

- Fuente: Gobierno de Chile (SEC)
- Período: 2015-2024 (10 años)
- Variables: año, mes, comuna, energía (kWh)

### Diapositiva 4: Modelo 1 - Regresión Lineal (1.5 minutos)

**Qué decir:**

> "El primer modelo es Regresión Lineal, que predice el consumo en kWh basado en el año, mes y comuna. Este modelo es simple e interpretable. Obtuvimos un R² de [X.XX], lo que significa que explica el [XX]% de la variación en el consumo."

**Incluir:**

- Gráfico: `grafico_regresion_lineal.png`
- Métricas: R², RMSE
- Interpretación: Qué significa el R²

### Diapositiva 5: Modelo 2 - Clasificación (1.5 minutos)

**Qué decir:**

> "El segundo modelo es Clasificación con Regresión Logística. En lugar de predecir el consumo exacto, clasificamos a los clientes en tres categorías: Bajo, Medio y Alto consumo. Esto es útil para segmentar clientes. Logramos una precisión del [XX]%."

**Incluir:**

- Gráfico: `grafico_matriz_confusion.png`
- Métrica: Accuracy
- Aplicación práctica: Segmentación de clientes

### Diapositiva 6: Modelo 3 - Random Forest (1.5 minutos)

**Qué decir:**

> "Random Forest es un conjunto de árboles de decisión que mejora la precisión. Lo más interesante es que nos muestra qué variables son más importantes. Descubrimos que la COMUNA es el factor más relevante, lo que tiene sentido porque diferentes zonas tienen distintos patrones de consumo."

**Incluir:**

- Gráfico: `grafico_importancia_variables.png`
- Métricas: R² mejorado
- Insight: La comuna es la variable más importante

### Diapositiva 7: Modelo 4 - Red Neuronal (1.5 minutos)

**Qué decir:**

> "Finalmente, implementamos una Red Neuronal con dos capas ocultas de 100 y 50 neuronas. Este es el modelo más avanzado y complejo. La red neuronal logró [explicar el desempeño]. Comparando los 4 modelos, vemos que [comparación]."

**Incluir:**

- Gráfico: `grafico_comparacion_modelos.png`
- Arquitectura: 2 capas ocultas
- Comparación con otros modelos

### Diapositiva 8: Conclusiones (1.5 minutos)

**Qué decir:**

> "En conclusión, implementamos exitosamente los 4 modelos requeridos. La comuna (ubicación) es el factor más importante. Los modelos más complejos (Random Forest y Red Neuronal) tienen mejor desempeño. Las aplicaciones prácticas incluyen planificación energética, identificación de clientes de alto consumo, y optimización de tarifas."

**Incluir:**

- Resumen de resultados
- Variable más importante: Comuna
- Aplicaciones prácticas
- Lecciones aprendidas

### Diapositiva 9: Lecciones Aprendidas (30 segundos)

**Qué decir:**

> "Lo más difícil fue la limpieza de datos y entender cómo usar cada modelo. Aprendimos que los datos reales son complejos y que diferentes modelos sirven para diferentes propósitos."

**Incluir:**

- Desafíos enfrentados
- Conocimientos adquiridos
- Mejoras futuras

---

## 💡 Consejos para la Defensa

### Durante la Presentación:

1. **Hablar con claridad y seguridad**
2. **Mirar al público, no leer las diapositivas**
3. **Usar los gráficos para apoyar tus puntos**
4. **Mantenerse dentro del tiempo (10 minutos)**

### Para las Preguntas (Q&A):

Posibles preguntas del profesor:

**P: "¿Por qué usaron Regresión Lineal?"**
R: "Porque queríamos predecir un valor numérico continuo (kWh) y es un modelo base simple para comparar con modelos más complejos."

**P: "¿Qué significa el R² de su modelo?"**
R: "El R² indica qué porcentaje de la variación en el consumo puede explicar nuestro modelo. Un R² de 0.80 significa que explica el 80% de la variación."

**P: "¿Por qué la comuna es la variable más importante?"**
R: "Porque diferentes comunas tienen distintos patrones de consumo debido a factores como clima, tipo de viviendas, nivel socioeconómico, etc."

**P: "¿Cómo dividieron los datos?"**
R: "Usamos 80% para entrenamiento y 20% para prueba, con random_state=42 para reproducibilidad."

**P: "¿Qué es una red neuronal?"**
R: "Es un modelo inspirado en el cerebro humano que tiene capas de neuronas. Cada neurona procesa información y la pasa a la siguiente capa. Nuestro modelo tiene 2 capas ocultas con 100 y 50 neuronas."

**P: "¿Cuál modelo recomendarían usar?"**
R: "Depende del objetivo. Si necesitamos interpretabilidad, Regresión Lineal. Si queremos precisión, Random Forest o Red Neuronal. Si queremos segmentar clientes, Clasificación."

**P: "¿Qué limitaciones tiene su análisis?"**
R: "No consideramos variables como temperatura, tipo de vivienda, o ingreso familiar. También los datos podrían tener errores de medición. Mejoras futuras incluirían más variables y análisis de series temporales."

---

## 📊 Archivos que Deben Tener Listos

### Archivos de Código:

1. `Limpiar_datos.ipynb` - Limpieza de datos
2. `Modelos_ML.ipynb` - Los 4 modelos implementados

### Archivos Generados:

1. `grafico_regresion_lineal.png`
2. `grafico_matriz_confusion.png`
3. `grafico_importancia_variables.png`
4. `grafico_comparacion_modelos.png`
5. `resultados_predicciones.csv`

### Presentación:

1. PowerPoint con las 9 diapositivas sugeridas

---

## ⚠️ Recordatorios Importantes

### Día de la Presentación:

- ✅ Llegar temprano
- ✅ Tener el código abierto en VS Code
- ✅ Tener los gráficos listos
- ✅ Presentación PowerPoint lista
- ✅ **IMPORTANTE:** Los 3 integrantes deben dominar TODO el proyecto

### Durante la Defensa:

- ✅ Un integrante al azar presentará
- ✅ Otro integrante al azar responderá preguntas
- ✅ Si uno falla, afecta la nota de TODO el grupo
- ✅ Deben poder explicar CUALQUIER línea de código

---

## 🎯 Criterios de Evaluación (Recordatorio)

| Criterio                     | Puntos | Qué Verificarán                       |
| ---------------------------- | ------ | ------------------------------------- |
| **Aplicación de Algoritmos** | 2 pts  | Los 4 modelos funcionan correctamente |
| **Calidad de Datos**         | 2 pts  | Dataset real y análisis coherente     |
| **Dominio de Código**        | 2 pts  | Entienden el código y los gráficos    |
| **Calidad Expositiva**       | 2 pts  | Presentación clara y en 10 minutos    |
| **Dominio Técnico (Q&A)**    | 2 pts  | Respuestas precisas a preguntas       |
| **TOTAL**                    | 10 pts |                                       |

**Fórmula de Nota:** (Puntaje / 10) × 6 + 1

---

## ✅ Última Verificación (1 Hora Antes)

- [ ] Batería del laptop cargada
- [ ] Archivos en OneDrive/USB de respaldo
- [ ] Código ejecutado sin errores
- [ ] Gráficos generados
- [ ] Presentación lista
- [ ] Los 3 integrantes revisaron TODO
- [ ] Practicaron la presentación al menos 2 veces

---

## 🎉 ¡Éxito en su Presentación!

Recuerden: Han hecho un trabajo profesional con datos reales. Confíen en su preparación y demuestren lo que aprendieron.

**"El objetivo no es la perfección, sino demostrar que entienden el ciclo completo de un proyecto de Data Science."**

---

## 📞 Contacto de Emergencia

Si tienen problemas técnicos de último minuto:

- Revisen la documentación en README.md
- Ejecuten las celdas en orden
- Verifiquen que pandas, numpy, scikit-learn, matplotlib estén instalados

**Comando de instalación rápida:**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
