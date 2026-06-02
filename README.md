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

Este repositorio contiene el código, los resultados y la memoria LaTeX del TFM cuyo objetivo es comparar el desempeño de distintas familias de modelos de pronóstico —desde métodos estadísticos clásicos hasta arquitecturas de aprendizaje profundo— aplicados a series temporales financieras mensuales del dataset M4, bajo un protocolo de backtesting temporal con ventanas móviles (*walk-forward*) y un conjunto de métricas estadísticas y de impacto económico.

---

## Estructura del repositorio

```
tfm_FP-A/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias Python exactas
├── run_experiment.py            # Script principal del experimento
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
│   ├── evaluation.py            # Seis métricas de evaluación validadas
│   ├── visualization.py         # Generación de las cinco figuras del experimento
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
│   ├── tables/                  # CSV con resultados numéricos
│   └── figures/                 # PNG de las figuras del experimento (fig01-fig05)
│
├── memoria/                     # Documento académico LaTeX
│   ├── main.tex                 # Archivo principal
│   ├── referencias.bib          # Bibliografía APA verificada (17 referencias)
│   ├── main.pdf                 # Memoria compilada (entregable)
│   ├── TFM_Entrega2.docx        # Exportación Word (entregable)
│   ├── figuras/                 # Figuras propias (diagrama de pipeline)
│   └── capitulos/               # Un .tex por capítulo
│       ├── resumen.tex
│       ├── organizacion.tex
│       ├── cap1_introduccion.tex
│       ├── cap2_contexto.tex
│       ├── cap3_objetivos.tex
│       ├── cap4_planteamiento.tex
│       ├── cap5_resultados.tex
│       ├── cap6_discusion.tex
│       ├── cap7_conclusiones.tex
│       ├── anexo_a.tex
│       └── acronimos.tex
│
├── Codigo/
│   └── TFE.ipynb                # Notebook exploratorio original (referencia histórica)
│
└── data/
    └── raw/                     # M4 dataset (se descarga automáticamente)
```

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

### 3. Actualizar la memoria LaTeX con los resultados

```bash
python update_results_latex.py
```

### 4. Ejecutar las pruebas automatizadas

```bash
python tests/test_evaluation.py
python tests/test_backtesting.py
```

### 5. Compilar la memoria LaTeX

Requiere MiKTeX o TeX Live con XeLaTeX y Biber (y la fuente Calibri instalada en el sistema).

```bash
cd memoria
xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex
```

### 6. Exportar la memoria a Word

```bash
bash export_to_word.sh
# o:
python export_to_word.py
```

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

---

## Protocolo de validación

Walk-forward *expanding window*:
- Ventana inicial de entrenamiento: 54 meses (4,5 años)
- Horizonte de evaluación: 18 meses (estándar M4 mensual)
- Desplazamiento entre iteraciones: 12 meses
- Verificación automática de ausencia de *data leakage* en cada fold
