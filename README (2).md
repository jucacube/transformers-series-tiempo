# Transformers para series de tiempo: pronóstico de temperatura

Actividad de la Especialización en Inteligencia Artificial (Universidad de Cundinamarca).
Se implementan, con PyTorch y sobre el dataset **Jena Climate 2009–2016**, un mecanismo de atención y un Transformer básico para pronosticar la temperatura de las próximas 24 horas a partir de las últimas 72.

## Contenido

| Archivo | Descripción |
|---|---|
| `01_atencion_series_tiempo.ipynb` | Atención de producto punto escalado (ejemplo), GRU y GRU + atención aditiva, análisis de pesos de atención |
| `02_transformer_series_tiempo.ipynb` | Transformer encoder desde cero (codificación posicional, multi-head self-attention, FFN, residual + LayerNorm) y comparación con GRU |
| `figures/` | Figuras generadas por los notebooks |
| `requirements.txt` | Dependencias (Colab ya las incluye) |

## Cómo ejecutarlo
1. Abrir cada notebook en [Google Colab](https://colab.research.google.com/) (Archivo > Abrir cuaderno > GitHub).
2. Activar GPU: Entorno de ejecución > Cambiar tipo de entorno de ejecución > T4 GPU.
3. Ejecutar todo: Entorno de ejecución > Ejecutar todo. El dataset se descarga automáticamente.

## Resumen metodológico
- **Datos:** temperatura horaria (`T (degC)`), ~70.000 registros. Entradas: temperatura + hora y día del año codificados con seno/coseno.
- **Partición cronológica** 70/15/15 y normalización z-score ajustada solo con entrenamiento.
- **Tarea:** ventana de 72 h -> predicción de 24 h.
- **Modelos:** persistencia (referencia), GRU, GRU + atención aditiva, Transformer encoder.
- **Métricas:** MAE y RMSE en °C.

## Referencias
- Vaswani et al. (2017). *Attention Is All You Need*.
- Transformers Explained Visually (Towards Data Science).
