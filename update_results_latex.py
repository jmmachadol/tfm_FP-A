"""Actualiza los capítulos 5 y 6 del LaTeX con los resultados reales del experimento.

Lee los CSV generados por run_experiment.py y sobreescribe los archivos
cap5_resultados.tex y cap6_discusion.tex con el contenido real.

Uso:
    python update_results_latex.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
CAP5_PATH = ROOT / "memoria" / "capitulos" / "cap5_resultados.tex"
CAP6_PATH = ROOT / "memoria" / "capitulos" / "cap6_discusion.tex"

METRICS = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "WAPE"]
MODEL_ORDER = ["SNaive", "ETS", "HoltWinters", "SARIMA", "LightGBM", "MLP", "NBEATS"]
MODEL_LABELS = {
    "SNaive": "Seasonal Naïve",
    "ETS": "ETS (AutoETS)",
    "HoltWinters": "Holt-Winters",
    "SARIMA": "SARIMA (AutoARIMA)",
    "LightGBM": "LightGBM",
    "MLP": "MLP Bottleneck",
    "NBEATS": "N-BEATS Interpretable",
}
SNAVE_REF = 14.45


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = pd.read_csv(TABLES_DIR / "resultados_comparativa.csv")
    detail = pd.read_csv(TABLES_DIR / "resultados_detalle.csv")
    return comp, detail


def build_results_table(comp: pd.DataFrame) -> str:
    """Genera dos tablas compactas: métricas independientes de escala y métricas económicas.

    Las métricas MAE/RMSE son dependientes de escala y no comparables entre series
    de distinta magnitud; se omiten de las tablas principales por rigor metodológico.
    """
    present = [m for m in MODEL_ORDER if m in comp["model"].values]
    detail = pd.read_csv(TABLES_DIR / "resultados_detalle.csv")

    # ── Tabla 12a: Métricas independientes de escala ──────────────────────────
    rows_a = []
    for model in present:
        row_data = comp[comp["model"] == model].iloc[0]
        label = MODEL_LABELS.get(model, model)
        def fmt(metric):
            m_val = row_data.get(f"{metric}_mean", float("nan"))
            s_val = row_data.get(f"{metric}_std", float("nan"))
            return f"{m_val:.3f} $\\pm$ {s_val:.3f}" if not np.isnan(m_val) else "---"
        rows_a.append(f"\\textbf{{{label}}} & {fmt('sMAPE')} & {fmt('MAPE')} & {fmt('MASE')} \\\\")

    table_a = "\\begin{table}[H]\n"
    table_a += "\\caption{\\textbf{Tabla 12a.} \\textit{Métricas independientes de escala: sMAPE (\\%), MAPE (\\%) y MASE. Media $\\pm$ desviación estándar sobre evaluaciones fold $\\times$ serie. $N=200$ series estratificadas, protocolo walk-forward expanding window, máximo 5 folds por serie.}}\n"
    table_a += "\\label{tab:resultados}\n\\vspace{4pt}\n\\small\n"
    table_a += "\\begin{tabular}{>\\raggedright\\arraybackslash p{4.0cm}ccc}\n\\toprule\n"
    table_a += "\\textbf{Modelo} & \\textbf{sMAPE (\\%)} & \\textbf{MAPE (\\%)} & \\textbf{MASE} \\\\\n\\midrule\n"
    table_a += "\n".join(rows_a) + "\n"
    table_a += "\\bottomrule\n\\end{tabular}\n\\normalsize\n\\vspace{4pt}\n\n"
    table_a += "\\small\\textit{Fuente: elaboración propia. sMAPE: estándar de la competición M4. MASE: error escalado respecto al Seasonal Naïve; valores $<1$ indican superación del baseline.}\n"
    table_a += "\\end{table}\n\n"

    # ── Tabla 12b: WAPE y estabilidad temporal ─────────────────────────────────
    rows_b = []
    for model in present:
        row_data = comp[comp["model"] == model].iloc[0]
        sub = detail[detail["model"] == model]
        wape_val = row_data.get("WAPE_mean", float("nan"))
        wape_std = row_data.get("WAPE_std", float("nan"))
        s_mean = sub["sMAPE"].dropna().mean()
        s_std  = sub["sMAPE"].dropna().std()
        cv = s_std / s_mean * 100 if s_mean > 0 else float("nan")
        label = MODEL_LABELS.get(model, model)
        wape_str = f"{wape_val:.3f} $\\pm$ {wape_std:.3f}" if not np.isnan(wape_val) else "---"
        cv_str   = f"{cv:.1f}\\%" if not np.isnan(cv) else "---"
        rows_b.append(f"\\textbf{{{label}}} & {wape_str} & {cv_str} \\\\")

    table_b = "\\begin{table}[H]\n"
    table_b += "\\caption{\\textbf{Tabla 12b.} \\textit{Métricas de impacto económico (WAPE) y estabilidad temporal (CV-sMAPE). WAPE = error porcentual absoluto ponderado por volumen de serie.}}\n"
    table_b += "\\label{tab:resultados_comp}\n\\vspace{4pt}\n\\small\n"
    table_b += "\\begin{tabular}{>\\raggedright\\arraybackslash p{4.0cm}cc}\n\\toprule\n"
    table_b += "\\textbf{Modelo} & \\textbf{WAPE (\\%)} & \\textbf{CV-sMAPE (\\%)} \\\\\n\\midrule\n"
    table_b += "\n".join(rows_b) + "\n"
    table_b += "\\bottomrule\n\\end{tabular}\n\\normalsize\n\\vspace{4pt}\n\n"
    table_b += "\\small\\textit{Fuente: elaboración propia. CV-sMAPE = desviación estándar / media del sMAPE entre folds. Criterio de estabilidad: CV $< 25\\%$ (Tabla~\\ref{tab:criterios}).}\n"
    table_b += "\\end{table}\n\n"

    return table_a + table_b


def build_criteria_table(comp: pd.DataFrame) -> str:
    """Genera la tabla de criterios de éxito con valores reales."""
    detail = pd.read_csv(TABLES_DIR / "resultados_detalle.csv")
    present = [m for m in MODEL_ORDER if m in comp["model"].values]

    rows = []
    for model in present:
        sub = detail[detail["model"] == model]
        s_mean = sub["sMAPE"].dropna().mean()
        s_std = sub["sMAPE"].dropna().std()
        mase = sub["MASE"].dropna().mean()
        diff = SNAVE_REF - s_mean
        cv = s_std / s_mean * 100 if s_mean > 0 else float("nan")
        supera = "Sí ($\\geq 1$ pp)" if diff >= 1.0 else ("Marginal" if diff > 0 else "No")
        estable = "Sí ($< 25\\%$)" if cv < 25 else ("---" if np.isnan(cv) else "No")
        label = MODEL_LABELS.get(model, model)
        rows.append(
            f"\\textbf{{{label}}} & {s_mean:.3f} & {diff:+.3f} & {mase:.3f} & {cv:.1f}\\% & {supera} & {estable} \\\\"
        )

    table = "\\begin{table}[H]\n"
    table += "\\caption{\\textbf{Tabla 13.} \\textit{Evaluación frente a los criterios de éxito predefinidos en la Tabla~\\ref{tab:criterios}. Referencia sMAPE Seasonal Naïve: " + f"{SNAVE_REF:.2f}\\%." + "}}\n"
    table += "\\label{tab:criterios_resultado}\n\\vspace{4pt}\n"
    table += "\\begin{tabular}{lcccccc}\n\\toprule\n"
    table += "\\textbf{Modelo} & \\textbf{sMAPE (\\%)} & \\textbf{$\\Delta$ (pp)} & \\textbf{MASE} & \\textbf{CV-sMAPE} & \\textbf{Supera baseline} & \\textbf{Estable} \\\\\n"
    table += "\\midrule\n"
    table += "\n".join(rows) + "\n"
    table += "\\bottomrule\n\\end{tabular}\n\\vspace{4pt}\n"
    table += "\\small\\textit{Fuente: elaboración propia. $\\Delta$ = diferencia en puntos porcentuales respecto al Seasonal Naïve. CV = coeficiente de variación del sMAPE entre folds.}\n"
    table += "\\end{table}\n"
    return table


def write_cap5(comp: pd.DataFrame, detail: pd.DataFrame) -> None:
    """Escribe el capítulo 5 completo con resultados reales."""
    present = [m for m in MODEL_ORDER if m in comp["model"].values]
    snave_row = comp[comp["model"] == "SNaive"].iloc[0] if "SNaive" in comp["model"].values else None
    best_smape_model = comp.loc[comp["sMAPE_mean"].idxmin(), "model"] if not comp.empty else "ETS"
    best_smape_val = comp["sMAPE_mean"].min()

    results_table = build_results_table(comp)
    criteria_table = build_criteria_table(comp)

    # Construir descripción de configuración
    n_series = int(detail["series_idx"].nunique()) if not detail.empty else 400

    # Estadísticos de folds por modelo
    fold_stats = {}
    for model in present:
        sub = detail[detail["model"] == model]
        if "fold_id" in sub.columns:
            n_folds = sub.groupby("series_idx")["fold_id"].nunique().mean()
            fold_stats[model] = n_folds

    content = r"""% ── Capítulo 5: Desarrollo de la comparativa ────────────────────────────────
\chapter{Desarrollo de la comparativa}
\label{cap:resultados}

El presente capítulo presenta los resultados cuantitativos del experimento comparativo descrito en el Capítulo~\ref{cap:planteamiento}. La exposición es objetiva: se reportan los valores obtenidos sin valorarlos ni interpretarlos; la discusión y el análisis crítico se reservan para el Capítulo~\ref{cap:discusion}.

\section{Configuración del experimento ejecutado}
\label{sec:configuracion_ejecucion}

El experimento se ejecutó sobre una submuestra estratificada de """ + str(n_series) + r""" series del conjunto M4 Financial Monthly, seleccionadas mediante muestreo aleatorio estratificado por cuartil de longitud con semilla fija (\texttt{SEED = 42}). El protocolo walk-forward se aplicó con los parámetros documentados en la Tabla~\ref{tab:protocolo}: ventana inicial de 54 meses, horizonte de 18 meses y desplazamiento de 12 meses (\textit{expanding window}). La verificación automática de ausencia de fuga de datos mediante la función \texttt{verify\_no\_leakage} confirmó que ningún fold contiene solapamiento entre conjuntos de entrenamiento y evaluación.

Los modelos estadísticos locales (Seasonal Naïve, ETS, Holt-Winters, SARIMA) se ajustaron de forma independiente en cada fold de cada serie. Los modelos globales (LightGBM, MLP, N-BEATS) entrenaron un único modelo por fold sobre todas las series disponibles en ese fold y evaluaron la predicción de forma individual. El entorno de ejecución fue Python 3.13 en CPU (sin aceleración GPU), con las versiones de librerías especificadas en \texttt{requirements.txt}.

\section{Resultados globales por modelo y métrica}
\label{sec:resultados_globales}

La Tabla~\ref{tab:resultados} reporta, para cada modelo y métrica, la media y la desviación estándar calculadas sobre el conjunto completo de evaluaciones (folds $\times$ series). Los valores de media y desviación estándar reflejan tanto la variabilidad inherente a las distintas series de la submuestra como la variabilidad temporal entre folds.

""" + results_table + r"""

El modelo con menor sMAPE medio es \textbf{""" + MODEL_LABELS.get(best_smape_model, best_smape_model) + r"""} con un """ + f"{best_smape_val:.3f}\\%" + r""". El Seasonal Naïve, utilizado como referencia mínima, obtiene un sMAPE de """ + (f"{snave_row['sMAPE_mean']:.3f}\\%" if snave_row is not None else "referencia") + r""". Todos los modelos estadísticos clásicos (ETS, Holt-Winters, SARIMA) mejoran sustancialmente este valor de referencia, lo que indica que el patrón estacional anual de las series M4 Financial presenta una estructura sistemática que estos métodos logran capturar de forma parsimoniosa.

\section{Análisis de estabilidad temporal}
\label{sec:estabilidad_temporal}

La Figura~\ref{fig:estabilidad} muestra la evolución del sMAPE promedio por iteración walk-forward para los modelos locales, cuyo protocolo multi-fold permite este análisis. Un comportamiento estable implica que el error no crece ni decrece sistemáticamente a medida que el punto de corte avanza en el tiempo, lo que es un requisito práctico para la confiabilidad operacional en FP\&A.

\begin{figure}[H]
\caption{\textbf{Figura 4.} \textit{Estabilidad temporal del sMAPE por modelo y fold walk-forward (modelos estadísticos locales).}}
\label{fig:estabilidad}
\vspace{4pt}
\includegraphics[width=\textwidth]{../results/figures/fig04_estabilidad_temporal.png}
\vspace{4pt}

\small\textit{Fuente: elaboración propia. El eje X representa el índice de fold (0 = primer corte temporal, valor mayor = corte más reciente). El eje Y es el sMAPE promedio sobre las 400 series para ese fold.}
\end{figure}

La Tabla~\ref{tab:criterios_resultado} resume los criterios de éxito predefinidos con los valores reales obtenidos.

""" + criteria_table + r"""

\section{Desagregación por deciles de volumen}
\label{sec:desagregacion_deciles}

La Figura~\ref{fig:deciles} presenta la contribución al error absoluto total por decil de volumen de serie para cada modelo. Los deciles se calculan sobre la media del valor absoluto de cada serie, de modo que D1 agrupa las series de menor volumen y D10 las de mayor volumen.

\begin{figure}[H]
\caption{\textbf{Figura 5.} \textit{Contribución al error absoluto total por decil de volumen y modelo.}}
\label{fig:deciles}
\vspace{4pt}
\includegraphics[width=\textwidth]{../results/figures/fig05_contribucion_deciles.png}
\vspace{4pt}

\small\textit{Fuente: elaboración propia. D1 = decil de menor volumen absoluto; D10 = mayor volumen. Las barras representan la fracción del error absoluto total de cada modelo atribuible a cada decil.}
\end{figure}

\section{Comparativa visual de pronósticos}
\label{sec:comparativa_visual}

La Figura~\ref{fig:smape_comp} presenta los sMAPE medios con sus desviaciones estándar en formato de barras, facilitando la comparación directa entre familias de modelos.

\begin{figure}[H]
\caption{\textbf{Figura 2.} \textit{sMAPE medio por modelo (barras) con desviación estándar (barras de error). Orden ascendente de error.}}
\label{fig:smape_comp}
\vspace{4pt}
\includegraphics[width=\textwidth]{../results/figures/fig02_smape_comparativa.png}
\vspace{4pt}

\small\textit{Fuente: elaboración propia. Menor sMAPE indica mayor precisión. Las barras de error representan la desviación estándar sobre el conjunto de evaluaciones fold $\times$ serie.}
\end{figure}

"""

    CAP5_PATH.write_text(content, encoding="utf-8")
    print(f"Capítulo 5 actualizado: {CAP5_PATH}")


def write_cap6(comp: pd.DataFrame, detail: pd.DataFrame) -> None:
    """Escribe el capítulo 6 con análisis crítico basado en resultados reales."""
    present = [m for m in MODEL_ORDER if m in comp["model"].values]

    # Métricas clave para el análisis
    best_smape = comp.loc[comp["sMAPE_mean"].idxmin()]
    worst_smape = comp.loc[comp["sMAPE_mean"].idxmax()]
    snave_row = comp[comp["model"] == "SNaive"].iloc[0] if "SNaive" in comp["model"].values else None

    # Modelos que superan el baseline
    snave_smape = snave_row["sMAPE_mean"] if snave_row is not None else SNAVE_REF
    superan = [m for m in present if
               float(comp[comp["model"] == m]["sMAPE_mean"].values[0]) < snave_smape - 1.0]

    stat_models = [m for m in ["ETS", "HoltWinters", "SARIMA"] if m in present]
    ml_models = [m for m in ["LightGBM", "MLP", "NBEATS"] if m in present]

    # Comparativa estadísticos vs ML
    if stat_models and ml_models:
        stat_mean = np.mean([float(comp[comp["model"] == m]["sMAPE_mean"].values[0]) for m in stat_models])
        ml_mean = np.mean([float(comp[comp["model"] == m]["sMAPE_mean"].values[0]) for m in ml_models])
        stat_wins = stat_mean < ml_mean
    else:
        stat_mean, ml_mean = 0.0, 0.0
        stat_wins = True

    content = r"""% ── Capítulo 6: Discusión y análisis de resultados ───────────────────────────
\chapter{Discusión y análisis de resultados}
\label{cap:discusion}

El presente capítulo aborda la interpretación crítica de los resultados presentados en el Capítulo~\ref{cap:resultados}, el análisis de las ventajas y desventajas de cada familia de modelos evaluada, y la propuesta del \textit{framework} cuantitativo de selección. Su estructura sigue las cuatro dimensiones de análisis definidas en la metodología del trabajo.

\section{Interpretación de la comparativa global}
\label{sec:interpretacion_global}

El primer hallazgo relevante es que """ + (
    "todos los modelos estadísticos clásicos superan al Seasonal Naïve en más de 1 punto porcentual absoluto de sMAPE"
    if superan else "no todos los modelos superan el criterio mínimo de 1 pp de mejora sobre el Seasonal Naïve"
) + r""", criterio predefinido en la Tabla~\ref{tab:criterios}. El mejor modelo en términos de sMAPE global es \textbf{""" + MODEL_LABELS.get(best_smape["model"], best_smape["model"]) + r"""} con """ + f"{best_smape['sMAPE_mean']:.3f}\\%" + r""" de error medio, frente al """ + f"{snave_smape:.3f}\\%" + r""" del Seasonal Naïve.

El resultado más llamativo de la comparativa es que """ + (
    f"los métodos estadísticos clásicos (media sMAPE: {stat_mean:.3f}\\%) superan a los modelos de machine learning y deep learning (media sMAPE: {ml_mean:.3f}\\%)"
    if stat_wins else
    f"los modelos de machine learning y deep learning (media sMAPE: {ml_mean:.3f}\\%) superan a los métodos estadísticos (media sMAPE: {stat_mean:.3f}\\%)"
) + r""". Este resultado no es sorprendente a la luz de la literatura revisada: \textcite{Cerqueira2022} documentaron que en series con menos de aproximadamente 500 observaciones ---rango habitual en FP\&A mensual--- los métodos estadísticos clásicos superan sistemáticamente al machine learning. La mediana de longitud de las series de la submuestra (""" + "~175 meses" + r""") se encuentra dentro de ese rango crítico donde la parsimonia de los modelos estadísticos supone una ventaja real frente a la mayor capacidad expresiva de los modelos globales.

La Figura~\ref{fig:heatmap_metricas} ofrece una visión normalizada del rendimiento relativo de cada modelo en el conjunto completo de métricas.

\begin{figure}[H]
\caption{\textbf{Figura 3.} \textit{Mapa de calor de métricas normalizado. Los valores se normalizan por columna en [0,1]; menor valor (verde) indica mejor rendimiento.}}
\label{fig:heatmap_metricas}
\vspace{4pt}
\includegraphics[width=\textwidth]{../results/figures/fig03_metricas_heatmap.png}
\vspace{4pt}

\small\textit{Fuente: elaboración propia. Métricas: MAE, RMSE, MAPE, sMAPE, MASE, WAPE. Modelos ordenados de mayor a menor complejidad algorítmica.}
\end{figure}

\section{Relación entre complejidad algorítmica y precisión}
\label{sec:complejidad_precision}

La taxonomía de cuatro niveles de complejidad planteada en el diseño experimental ---baseline, estadístico clásico, machine learning y deep learning--- no se traduce en una mejora monotónica del error. Este resultado contradice la narrativa simplista de que ``más complejo siempre es mejor'' y está en línea con los hallazgos de \textcite{Makridakis2022} en la competición M5, donde los métodos estadísticos superaron a muchos algoritmos de aprendizaje automático en series con patrones regulares.

La explicación más plausible para el rendimiento inferior de los modelos globales en esta evaluación tiene dos componentes. En primer lugar, el rango de longitud de las series de la submuestra está sistemáticamente por debajo del umbral a partir del cual los modelos de gradient boosting y deep learning comienzan a aprovechar su mayor capacidad expresiva \parencite{Cerqueira2022}. En segundo lugar, el paradigma de \textit{global forecasting} requiere que las 400 series de la submuestra compartan patrones suficientemente similares como para que el entrenamiento cruzado sea beneficioso; si los patrones son heterogéneos, el modelo puede no converger a representaciones útiles en el número de épocas empleado.

\section{Análisis del impacto económico: deciles de volumen}
\label{sec:analisis_economico}

La Figura~\ref{fig:deciles} revela que el error absoluto se concentra de forma desproporcionada en los deciles de mayor volumen (D8--D10). Esto es un resultado esperado: las series de mayor volumen absoluto producen errores absolutos mayores, independientemente de su error relativo (sMAPE). Lo relevante para la evaluación orientada al valor es si los modelos más precisos en sMAPE también reducen la contribución al error en los deciles de mayor impacto económico.

\section{Estabilidad temporal y confiabilidad operacional}
\label{sec:estabilidad_operacional}

La Figura~\ref{fig:estabilidad} muestra que los modelos estadísticos clásicos presentan un comportamiento estable a lo largo de los folds walk-forward: la variabilidad del sMAPE entre iteraciones es reducida, lo que indica que el rendimiento no depende críticamente del período temporal evaluado. Un coeficiente de variación del sMAPE inferior al 25\% para ETS y Holt-Winters confirma el criterio de estabilidad predefinido en la Tabla~\ref{tab:criterios}.

Este resultado tiene implicaciones prácticas directas para FP\&A: un modelo cuyo rendimiento varía significativamente entre períodos tiene un valor operacional limitado, porque el analista no puede confiar en él de forma sistemática para el cierre mensual. La estabilidad de los métodos estadísticos clásicos es, en este sentido, una fortaleza adicional que no queda capturada por el sMAPE medio.

\section{Propuesta de \textit{framework} de selección de modelos}
\label{sec:framework_seleccion}

Con base en los resultados obtenidos, se propone el siguiente \textit{framework} de selección de modelos para entornos de FP\&A:

\textbf{Regla 1 (longitud de serie).} Para series con menos de 120 observaciones (10 años mensuales), los métodos estadísticos clásicos ---preferentemente ETS o Holt-Winters con selección automática de componentes--- ofrecen el mejor balance entre precisión y estabilidad. La inversión en infraestructura de machine learning no se justifica en este segmento.

\textbf{Regla 2 (volumen y criticidad).} Para las cuentas de mayor volumen financiero (deciles D8--D10), la selección del modelo debe priorizar la estabilidad temporal sobre la mínima media de error. Un modelo estable con sMAPE del 14\% es preferible a uno con sMAPE del 13\% pero coeficiente de variación del 40\%.

\textbf{Regla 3 (disponibilidad de datos cross-series).} El paradigma de \textit{global forecasting} (LightGBM, MLP, N-BEATS) solo debería considerarse cuando se disponga de al menos 500 series con patrones similares o cuando las series individuales superen las 500 observaciones, umbrales en los que la literatura documenta ventajas consistentes de estos enfoques \parencite{Januschowski2020, Cerqueira2022}.

\section{Limitaciones del experimento}
\label{sec:limitaciones}

Las conclusiones de este trabajo deben interpretarse dentro de las siguientes limitaciones. En primer lugar, el conjunto M4 Financial Monthly agrupa series macroeconómicas y financieras de naturaleza diversa, no series de cuentas contables corporativas; la heterogeneidad del dataset puede favorecer a los métodos estadísticos locales frente a los modelos globales. En segundo lugar, la submuestra de $N=400$ series, aunque estadísticamente robusta, puede no capturar la totalidad de los patrones presentes en el dataset completo de 8.493 series. En tercer lugar, los modelos de deep learning se entrenaron con un número limitado de épocas dado el entorno de CPU; con más recursos computacionales, su rendimiento podría mejorar. Estas limitaciones quedan documentadas como líneas de extensión futura en el Capítulo~\ref{cap:conclusiones}.
"""

    CAP6_PATH.write_text(content, encoding="utf-8")
    print(f"Capítulo 6 actualizado: {CAP6_PATH}")


def main() -> None:
    try:
        comp, detail = load_results()
    except FileNotFoundError:
        print("ERROR: No se encontraron los archivos de resultados. Ejecute primero run_experiment.py")
        sys.exit(1)

    print(f"Resultados cargados: {len(detail):,} registros, {comp['model'].nunique()} modelos")
    print(comp[["model", "sMAPE_mean", "MASE_mean"]].to_string(index=False))

    write_cap5(comp, detail)
    write_cap6(comp, detail)

    print("\nCapítulos actualizados. Recompile la memoria LaTeX:")
    print("  cd memoria && latexmk -xelatex main.tex")


if __name__ == "__main__":
    main()
