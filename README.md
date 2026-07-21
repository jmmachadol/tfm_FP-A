# Evaluación de modelos de pronóstico para FP&A

**Trabajo de Fin de Máster — Universidad Internacional de La Rioja (UNIR)**  
Máster Universitario en Inteligencia Artificial

---

## Autores

| Autor | Responsabilidad principal |
|---|---|
| Juan Camilo Rico Ballesteros | Introducción · Módulos de modelos · Análisis de errores |
| Juan José Blanco Mendoza | Contexto y estado del arte · Evaluación y métricas · Resultados |
| José Manuel Machado Loaiza | Objetivos y metodología · Arquitectura backtesting · Configuración |

**Directora:** Marta María Arguedas Lafuente

---

## Descripción

Este repositorio contiene el código, los resultados y la memoria académica del TFM cuyo objetivo es comparar el desempeño de distintas familias de modelos de pronóstico —desde métodos estadísticos clásicos hasta arquitecturas de aprendizaje profundo— aplicados a series temporales financieras mensuales del dataset M4, bajo un protocolo de backtesting temporal con ventanas móviles (*walk-forward*) y un conjunto de métricas estadísticas y de impacto económico. Los resultados incluyen intervalos de confianza bootstrap al 95 % (B = 1 000, método percentil) para verificar la significación estadística de las diferencias entre modelos.

---

## Estructura del repositorio

```
tfm_FP-A/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias Python exactas
├── run_experiment.py            # Script principal del experimento
├── benchmark_timing.py          # Benchmark homogéneo de coste computacional (Tabla 29)
├── update_results_latex.py      # Actualiza los capítulos 5 y 6 con resultados reales
├── export_to_word.py            # Exporta la memoria LaTeX a Word
├── export_to_word.sh            # Script equivalente usando pandoc directamente
│
├── src/                         # Código fuente modular
│   ├── config.py                # Parámetros del experimento y rutas
│   ├── utils.py                 # Semillas globales y logging
│   ├── data_loader.py           # Descarga con caché del M4 Financial Monthly
│   ├── preprocessor.py          # Filtrado y submuestra estratificada
│   ├── backtesting.py           # Motor walk-forward + verificador de leakage
│   ├── evaluation.py            # Seis métricas de evaluación + bootstrap_smape_diff_ci()
│   ├── visualization.py         # Generación de las figuras del experimento
│   └── models/
│       ├── base.py              # Interfaces BaseForecaster / GlobalBaseForecaster
│       ├── baseline.py          # Seasonal Naïve
│       ├── statistical.py       # ETS, Holt-Winters, SARIMA
│       └── ml.py                # LightGBM, MLP Bottleneck, N-BEATS Interpretable
│
├── notebooks/
│   └── main.ipynb               # Orquestador interactivo del experimento
│
├── tests/                       # Pruebas automatizadas (20/20)
│   ├── test_evaluation.py       # 14 pruebas: métricas validadas manualmente
│   └── test_backtesting.py      # 6 pruebas: leakage, geometría, rolling/expanding
│
├── results/
│   ├── tables/
│   │   ├── resultados_comparativa.csv       # Métricas medias por modelo (7 modelos × 6 métricas)
│   │   ├── resultados_bootstrap_ci.csv      # IC bootstrap 95 % (modelo − SNaive, B=1000, N=200)
│   │   ├── resultados_bonferroni.csv        # IC ajustado 99,5 % (corrección de Bonferroni, 10 comparaciones)
│   │   ├── resultados_cv_temporal.csv       # CV temporal del sMAPE por serie, promediado por modelo
│   │   ├── resultados_distribucion_folds.csv # Nº de series por cantidad de folds retenidos
│   │   ├── resultados_wape_decil.csv        # WAPE medio por modelo y decil de volumen
│   │   ├── resultados_timing.csv            # Tiempos de ajuste/inferencia y tamaño por modelo
│   │   ├── entorno_hardware.csv             # Entorno de hardware del benchmark de timing
│   │   └── resultados_detalle.csv           # Resultados por serie y fold
│   └── figures/                             # PNG de las figuras del experimento
│
├── Memoria/                                 # Documento académico
│   ├── Fuente LaTeX/                        # Fuente LaTeX histórica (referencia)
│   │   ├── main.tex
│   │   ├── referencias.bib
│   │   ├── main.pdf
│   │   ├── figuras/
│   │   └── capitulos/
│   │
│   └── Versiones y Entregas/
│       ├── Entrega_1/
│       │   ├── v1.1_Entrega01_..._Corregida.docx   # Versión final entregada
│       │   └── Propuesta Sección 4.docx
│       ├── Entrega_2/
│       │   ├── TFM_Entrega2.docx                   # Versión final entregada
│       │   └── R2_Grupo 5.docx                     # Revisión de la entrega 2
│       ├── Entrega_3/
│       │   ├── TFM_Entrega3.docx                   # Versión entregada — Entrega 3
│       │   └── R3 G5.docx                          # Revisión de la entrega 3
│       ├── Predeposito_Correcciones/               # Predepósito + feedback de la directora (R3 G5)
│       │   ├── TFM_Predeposito.docx
│       │   ├── TFM_Predeposito (1).pdf
│       │   └── G5 Revisión predeposito.docx
│       └── Deposito/
│           └── TFM_Deposito.docx                   # Versión final depositada
│
├── Administrativo/               # Documentos institucionales (no forman parte del TFM)
│   ├── instrucciones.pdf         # Instrucciones UNIR (ignorado por git, no redistribuible)
│   ├── rubrica.pdf               # Rúbrica de evaluación (ignorado por git, no redistribuible)
│   ├── Plantilla Grupal.docx     # Plantilla UNIR base para las entregas
│   └── Revisión grupo 5 ESIT.docx
│
└── data/
    └── raw/                     # M4 dataset (se descarga automáticamente)
```

Cada entrega conserva únicamente su versión final entregada y, si aplica, la revisión del tribunal; los borradores intermedios (`Versiones antiguas/`) se descartaron tras la aprobación de cada hito, quedando disponibles en el historial de git si se necesitan.

---

## Instalación y uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el experimento completo

```bash
python run_experiment.py --n-series 200 --max-folds 5
```

Los datos del M4 se descargan automáticamente en `data/raw/` en la primera ejecución.  
El script genera automáticamente `results/tables/resultados_bootstrap_ci.csv` con los intervalos de confianza bootstrap al 95 % para cada modelo frente a Seasonal Naïve.

### 3. Ejecutar las pruebas automatizadas

```bash
python tests/test_evaluation.py
python tests/test_backtesting.py
```

### 4. (Opcional) Medir el coste computacional homogéneo de los siete modelos

```bash
python benchmark_timing.py
```

Genera `results/tables/resultados_timing.csv` y `results/tables/entorno_hardware.csv` (Tabla 29 de la memoria).

---

## Dataset

**M4 Competition Dataset — Financial Monthly**  
Fuente: https://github.com/Mcompetitions/M4-methods  
Licencia: MIT (acceso público)

10.987 series financieras mensuales → 8.493 tras filtrado (≥ 72 meses) → submuestra estratificada de 200 series para el experimento completo.

---

## Modelos evaluados

| Nivel | Modelo | Implementación |
|---|---|---|
| Baseline | Seasonal Naïve | NumPy (implementación propia) |
| Estadístico | ETS (AutoETS) | statsforecast |
| Estadístico | Holt-Winters | statsmodels |
| Estadístico | SARIMA (AutoARIMA) | statsforecast |
| Machine Learning | LightGBM | lightgbm |
| Deep Learning | MLP Bottleneck | PyTorch |
| Deep Learning | N-BEATS Interpretable | PyTorch |

---

## Métricas de evaluación

**Estadísticas:** MAE, RMSE, MAPE, sMAPE (estándar M4), MASE  
**Impacto económico:** WAPE, contribución al error por decil de volumen  
**Inferencia estadística:** IC bootstrap percentil 95 % (B = 1 000) para diferencia sMAPE(modelo) − sMAPE(SNaive), unidad de remuestreo = serie (N = 200), pareado por serie a través de los folds

---

## Resultados principales (N = 200 series, max\_folds = 5, SEED = 42)

| Modelo | sMAPE (%) | WAPE (%) | MASE | Diff vs SNaive (pp) | IC 95 % | Significativo |
|---|---|---|---|---|---|---|
| **ETS** | **10.005** | **10.081** | **1.017** | **−3.852** | [−4.517, −3.264] | Sí |
| SARIMA | 10.825 | 10.881 | 1.093 | −3.154 | [−3.945, −2.481] | Sí |
| Holt-Winters | 11.157 | 11.583 | 1.145 | −2.725 | [−3.515, −2.028] | Sí |
| LightGBM | 11.682 | 11.542 | 1.291 | −1.998 | [−2.572, −1.447] | Sí |
| MLP | 11.844 | 11.637 | 1.351 | −1.898 | [−2.453, −1.322] | Sí |
| N-BEATS | 12.375 | 12.074 | 1.340 | −1.065 | [−1.509, −0.596] | Sí |
| SNaive | 13.667 | 13.813 | 1.412 | 0.000 | — | — |

ETS (AutoETS) es el mejor modelo en las seis métricas. Todos los modelos superan al baseline de forma estadísticamente significativa (IC 95 % no contiene el cero).

---

## Protocolo de validación

Walk-forward *expanding window*:
- Ventana inicial de entrenamiento: 54 meses (4,5 años)
- Horizonte de evaluación: 18 meses (estándar M4 mensual)
- Desplazamiento entre iteraciones: 12 meses
- Verificación automática de ausencia de *data leakage* en cada fold
