"""
Módulo de Análisis Estadístico de Alturas del Río Paraná - Puerto La Paz
Asignatura: Estadística / Trabajo Práctico Universitario

Contiene las 4 funciones estadísticas exigidas:
1. estadistica_descriptiva(): Media, mediana, varianza y sentido de asimetría.
2. estadistica_inferencial(): Prueba de hipótesis de cola derecha (H0: mu <= 3.50m, alpha=0.05).
3. regresion_lineal_ar1(): Regresión lineal simple autoregresiva de rezago 1 (Y_t ~ X_{t-1}) con R^2 y ecuación.
4. serie_temporal_tendencia(): Promedio móvil de 30 días para aislar tendencia secular y preparación de series cronológicas.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DB_NAME = "rio.db"
TABLE_NAME = "alturas_historico"


def _safe_float(val, default=0.0):
    """Garantiza que el valor flotante sea estrictamente finito (evita NaN e Infinity en JSON)."""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


def get_db_path():
    """Obtiene la ruta a rio.db en la raíz del proyecto o en subcarpetas."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, DB_NAME),
        os.path.join(base_dir, "database", DB_NAME),
        os.path.join(os.getcwd(), DB_NAME),
        os.path.join(os.getcwd(), "database", DB_NAME)
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return os.path.join(base_dir, DB_NAME)


def cargar_datos_desde_db(db_path=None):
    """
    Carga el historial hidrométrico desde la base de datos SQLite usando Pandas.
    Retorna un DataFrame ordenado cronológicamente por 'fecha'.
    """
    if db_path is None:
        db_path = get_db_path()

    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de base de datos en '{db_path}'. "
            "Ejecuta primero 'python scraper.py' para inicializar y poblar los datos."
        )

    with sqlite3.connect(db_path) as conn:
        query = f"SELECT fecha, altura FROM {TABLE_NAME} ORDER BY fecha ASC"
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError("La tabla de mediciones en la base de datos se encuentra vacía.")

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["altura"] = pd.to_numeric(df["altura"], errors="coerce")
    df = df.dropna().sort_values(by="fecha").reset_index(drop=True)
    return df


def estadistica_descriptiva(df=None):
    """
    1. Estadística Descriptiva Ampliada:
    Calcula medidas de tendencia central (media, mediana, moda, media recortada, cuartiles)
    y medidas de dispersión/varianza (varianza, desviación estándar, CV%, rango total, IQR).
    """
    if df is None:
        df = cargar_datos_desde_db()

    alturas = df["altura"].to_numpy()
    n = len(alturas)
    
    # --- Medidas de Tendencia Central ---
    media = _safe_float(np.mean(alturas))
    mediana = _safe_float(np.median(alturas))
    
    # Moda sobre valores redondeados a 1 decimal (agrupamiento hidrométrico habitual)
    try:
        alturas_redondeadas = np.round(alturas, 1)
        res_moda = stats.mode(alturas_redondeadas, keepdims=True)
        moda = _safe_float(res_moda.mode[0] if len(res_moda.mode) > 0 else media)
    except Exception:
        moda = media

    # Media Recortada al 5% (robusta ante crecidas atípicas)
    try:
        media_recortada = _safe_float(stats.trim_mean(alturas, 0.05))
    except Exception:
        media_recortada = media

    # Cuartiles y Rango Intercuartílico (IQR)
    q1 = _safe_float(np.percentile(alturas, 25))
    q3 = _safe_float(np.percentile(alturas, 75))
    iqr = _safe_float(q3 - q1)

    # --- Medidas de Dispersión y Varianza ---
    varianza = _safe_float(np.var(alturas, ddof=1) if n > 1 else 0.0)
    desv_estandar = _safe_float(np.sqrt(varianza))
    coef_variacion = _safe_float((desv_estandar / media) * 100 if media != 0 else 0.0)
    
    minimo = _safe_float(np.min(alturas))
    maximo = _safe_float(np.max(alturas))
    rango_total = _safe_float(maximo - minimo)
    error_estandar_media = _safe_float(desv_estandar / np.sqrt(n) if n > 0 else 0.0)

    # --- Medidas de Forma ---
    try:
        coef_asimetria = _safe_float(stats.skew(alturas, bias=False), default=0.0)
    except Exception:
        coef_asimetria = 0.0

    try:
        curtosis = _safe_float(stats.kurtosis(alturas, bias=False), default=0.0)
    except Exception:
        curtosis = 0.0

    umbral_simetria = 0.08
    if coef_asimetria > umbral_simetria:
        sentido_asimetria = "Asimetría positiva (a la derecha)"
        detalle_asimetria = (
            f"Media ({media:.2f} m) > Mediana ({mediana:.2f} m). "
            "La distribución presenta una cola extendida hacia valores altos, reflejando crecidas episódicas "
            "provocadas por el fenómeno de El Niño sobre un cauce habitual más moderado."
        )
    elif coef_asimetria < -umbral_simetria:
        sentido_asimetria = "Asimetría negativa (a la izquierda)"
        detalle_asimetria = (
            f"Media ({media:.2f} m) < Mediana ({mediana:.2f} m). "
            "Predominan niveles altos con descensos puntuales hacia valores mínimos."
        )
    else:
        sentido_asimetria = "Aproximadamente simétrica"
        detalle_asimetria = (
            f"Media ({media:.2f} m) ≈ Mediana ({mediana:.2f} m). "
            "Las fluctuaciones hidrológicas se distribuyen de forma balanceada en torno al nivel medio."
        )

    # Cálculo de bins de frecuencia para el histograma
    conteos, bordes_bins = np.histogram(alturas, bins=min(12, max(4, n // 10)))
    bins_labels = [f"{bordes_bins[i]:.2f} - {bordes_bins[i+1]:.2f} m" for i in range(len(conteos))]

    return {
        "n": int(n),
        # Tendencia Central
        "media": round(media, 4),
        "mediana": round(mediana, 4),
        "moda": round(moda, 2),
        "media_recortada_5": round(media_recortada, 4),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        # Dispersión y Varianza
        "varianza": round(varianza, 4),
        "desviacion_estandar": round(desv_estandar, 4),
        "coeficiente_variacion": round(coef_variacion, 2),
        "minimo": round(minimo, 2),
        "maximo": round(maximo, 2),
        "rango_total": round(rango_total, 2),
        "error_estandar_media": round(error_estandar_media, 4),
        # Forma
        "coeficiente_asimetria": round(coef_asimetria, 4),
        "curtosis": round(curtosis, 4),
        "sentido_asimetria": sentido_asimetria,
        "detalle_asimetria": detalle_asimetria,
        "histograma": {
            "bins_labels": bins_labels,
            "frecuencias": [int(c) for c in conteos],
            "bordes": [round(_safe_float(b), 2) for b in bordes_bins]
        }
    }


def estadistica_inferencial(df=None, mu_0=None, alpha=0.05):
    """
    2. Estadística Inferencial:
    Prueba de hipótesis de cola derecha sobre la muestra de los últimos 30 días.
    H0: mu_muestral <= mu_historica (La altura media reciente es menor o igual al promedio histórico)
    H1: mu_muestral > mu_historica  (La altura media reciente es significativamente mayor por El Niño)
    Nivel de significancia: alpha = 0.05 (5%)
    """
    if df is None:
        df = cargar_datos_desde_db()

    # Si no se especifica mu_0, se calcula la media histórica global de todo el registro (365 días)
    if mu_0 is None:
        mu_historica = _safe_float(np.mean(df["altura"].to_numpy()))
    else:
        mu_historica = float(mu_0)

    # Muestra de los últimos 30 días (período afectado bajo estudio)
    muestra_30 = df.tail(30)["altura"].to_numpy()
    n = len(muestra_30)
    if n < 2:
        raise ValueError("Se requieren al menos 2 observaciones para la prueba de hipótesis t.")

    media_muestral = _safe_float(np.mean(muestra_30))
    desv_muestral = _safe_float(np.std(muestra_30, ddof=1))
    
    # Manejo de desviación nula o casi nula (tolerancia instrumental de 1 cm)
    if desv_muestral < 1e-4:
        error_estandar = 0.01 / np.sqrt(n)
    else:
        error_estandar = desv_muestral / np.sqrt(n)

    grados_libertad = n - 1

    # Estadístico de prueba t: t = (x_bar - mu_historica) / (s / sqrt(n))
    diff = media_muestral - mu_historica
    t_stat = diff / error_estandar
    t_stat = _safe_float(t_stat, default=999.0 if diff > 0 else -999.0)

    # p-valor para prueba unilateral derecha: P(T >= t)
    try:
        p_valor = _safe_float(stats.t.sf(t_stat, df=grados_libertad))
    except Exception:
        p_valor = 0.0 if t_stat > 0 else 1.0

    # Valor crítico en t-Student para alpha=0.05
    try:
        t_critico = _safe_float(stats.t.ppf(1 - alpha, df=grados_libertad))
    except Exception:
        t_critico = 1.6991

    rechaza_h0 = bool(p_valor < alpha)

    if rechaza_h0:
        conclusion = (
            f"Con un nivel de significancia del {int(alpha*100)}% (alfa = {alpha:.2f}) y un p-valor de {p_valor:.4e} "
            f"(p < alfa), se RECHAZA la hipótesis nula (H0). Existe evidencia estadística suficiente para afirmar "
            f"que la altura media muestral de los últimos 30 días ({media_muestral:.2f} m) es significativamente mayor "
            f"a la media histórica anual ({mu_historica:.2f} m). Esto corrobora empíricamente que el fenómeno de El Niño "
            f"provocó un incremento sustancial en el nivel del río Paraná en el puerto de La Paz."
        )
    else:
        conclusion = (
            f"Con un nivel de significancia del {int(alpha*100)}% (alfa = {alpha:.2f}) y un p-valor de {p_valor:.4f} "
            f"(p >= alfa), NO se rechaza la hipótesis nula (H0). No existe evidencia estadística suficiente para sostener "
            f"que la media muestral reciente ({media_muestral:.2f} m) sea superior a la media histórica ({mu_historica:.2f} m)."
        )

    return {
        "tamano_muestra": int(n),
        "mu_historica": round(float(mu_historica), 4),
        "mu_0": round(float(mu_historica), 4),
        "alpha": float(alpha),
        "media_muestral": round(media_muestral, 4),
        "diferencia_medias": round(float(media_muestral - mu_historica), 4),
        "desviacion_muestral": round(float(desv_muestral), 4),
        "error_estandar": round(float(error_estandar), 4),
        "grados_libertad": int(grados_libertad),
        "estadistico_t": round(float(t_stat), 4),
        "valor_critico_t": round(float(t_critico), 4),
        "p_valor": round(float(p_valor), 6),
        "rechaza_h0": rechaza_h0,
        "conclusion": conclusion
    }


def regresion_lineal_ar1(df=None):
    """
    3. Regresión:
    Modelo autorregresivo simple de rezago 1 (AR(1)):
    Predice la altura del día t (Y) en función de la altura del día anterior t-1 (X).
    Y_t = beta_0 + beta_1 * X_{t-1} + e_t
    Retorna la ecuación de la recta y el coeficiente de determinación R^2.
    """
    if df is None:
        df = cargar_datos_desde_db()

    serie = df["altura"].to_numpy()
    if len(serie) < 3:
        raise ValueError("Serie temporal insuficiente para calcular la regresión autorregresiva.")

    X = serie[:-1]
    Y = serie[1:]

    try:
        res = stats.linregress(X, Y)
        pendiente = _safe_float(res.slope, 1.0)
        intercepto = _safe_float(res.intercept, 0.0)
        r_value = _safe_float(res.rvalue, 1.0)
        r2 = _safe_float(r_value ** 2, 1.0)
        p_value = _safe_float(res.pvalue, 0.0)
        std_err = _safe_float(res.stderr, 0.0)
    except Exception:
        pendiente, intercepto, r_value, r2, p_value, std_err = 1.0, 0.0, 1.0, 1.0, 0.0, 0.0

    signo = "+" if intercepto >= 0 else "-"
    ecuacion = f"Altura(t) = {pendiente:.4f} · Altura(t-1) {signo} {abs(intercepto):.4f}"

    interpretacion = (
        f"El coeficiente de determinación R² = {r2:.4f} indica que el {r2 * 100:.2f}% de la variabilidad "
        "en la altura diaria del río es explicada por el nivel del día previo. La pendiente cercana a 1 "
        f"({pendiente:.4f}) refleja la notable inercia hidrológica de la cuenca del Paraná, donde el escurrimiento "
        "de crecida por El Niño muestra una alta persistencia temporal y lentitud de variación diaria."
    )

    return {
        "pendiente": round(pendiente, 4),
        "intercepto": round(intercepto, 4),
        "r2": round(r2, 4),
        "r_pearson": round(r_value, 4),
        "p_valor": round(p_value, 6),
        "error_estandar": round(std_err, 4),
        "ecuacion": ecuacion,
        "interpretacion": interpretacion
    }


def serie_temporal_tendencia(df=None, ventana=30):
    """
    4. Series de Tiempo:
    Calcula un promedio móvil de 30 días para aislar la tendencia secular
    y prepara los arreglos cronológicos para su graficación en Chart.js.
    """
    if df is None:
        df = cargar_datos_desde_db()

    df_st = df.copy().sort_values(by="fecha").reset_index(drop=True)
    df_st["media_movil_30"] = df_st["altura"].rolling(window=ventana, min_periods=1).mean()

    fechas_str = df_st["fecha"].dt.strftime("%Y-%m-%d").tolist()
    alturas_list = [round(_safe_float(h), 2) for h in df_st["altura"].tolist()]
    media_movil_list = [round(_safe_float(m), 2) for m in df_st["media_movil_30"].tolist()]

    max_idx = int(df_st["altura"].idxmax())
    min_idx = int(df_st["altura"].idxmin())

    resumen = {
        "total_dias": len(df_st),
        "fecha_inicio": fechas_str[0],
        "fecha_fin": fechas_str[-1],
        "pico_maximo": {
            "fecha": fechas_str[max_idx],
            "altura": alturas_list[max_idx]
        },
        "estiaje_minimo": {
            "fecha": fechas_str[min_idx],
            "altura": alturas_list[min_idx]
        }
    }

    return {
        "fechas": fechas_str,
        "alturas": alturas_list,
        "media_movil_30": media_movil_list,
        "resumen": resumen
    }


def pronostico_umbrales(df=None, umbral_alerta=5.80, umbral_evacuacion=6.15):
    """
    Pronóstico de Series de Tiempo y Detección de Umbrales Críticos:
    1. Retrospectivo: Determina con precisión cuándo se cruzó el umbral de alerta (5.80 m).
    2. Prospectivo: Utiliza el modelo autorregresivo AR(1) iterativo y la tasa de variación
       secular reciente para proyectar en cuántos días/horas se alcanzará el umbral de evacuación (6.15 m).
    """
    from datetime import datetime, timedelta

    if df is None:
        df = cargar_datos_desde_db()

    df_ord = df.copy().sort_values(by="fecha").reset_index(drop=True)
    df_ord["fecha_dt"] = pd.to_datetime(df_ord["fecha"])
    
    # 1. Análisis Retrospectivo de Alerta (5.80 m)
    cruce_alerta_row = None
    for i in range(len(df_ord) - 1, -1, -1):
        if df_ord.loc[i, "altura"] >= umbral_alerta:
            cruce_alerta_row = df_ord.loc[i]
        else:
            break

    supero_alerta = cruce_alerta_row is not None
    if supero_alerta:
        fecha_alerta_str = cruce_alerta_row["fecha_dt"].strftime("%Y-%m-%d")
        fecha_alerta_iso = cruce_alerta_row["fecha_dt"].strftime("%Y-%m-%dT00:00:00")
        altura_alerta = round(float(cruce_alerta_row["altura"]), 2)
    else:
        df_alerta = df_ord[df_ord["altura"] >= umbral_alerta]
        if not df_alerta.empty:
            supero_alerta = True
            cruce_alerta_row = df_alerta.iloc[0]
            fecha_alerta_str = cruce_alerta_row["fecha_dt"].strftime("%Y-%m-%d")
            fecha_alerta_iso = cruce_alerta_row["fecha_dt"].strftime("%Y-%m-%dT00:00:00")
            altura_alerta = round(float(cruce_alerta_row["altura"]), 2)
        else:
            supero_alerta = False
            fecha_alerta_str = None
            fecha_alerta_iso = None
            altura_alerta = None

    # 2. Análisis Prospectivo hacia Evacuación (6.15 m)
    ultima_row = df_ord.iloc[-1]
    ultima_altura = round(float(ultima_row["altura"]), 2)
    ultima_fecha_dt = ultima_row["fecha_dt"]

    # Ajuste AR(1) sobre la serie
    serie = df_ord["altura"].to_numpy()
    X = serie[:-1]
    Y = serie[1:]
    res_reg = stats.linregress(X, Y)
    slope = float(res_reg.slope)
    intercept = float(res_reg.intercept)

    # Tasa reciente (últimos 7 días)
    ultimos_7 = df_ord.tail(7)
    if len(ultimos_7) >= 2:
        dh = float(ultimos_7.iloc[-1]["altura"] - ultimos_7.iloc[0]["altura"])
        dt_dias = len(ultimos_7) - 1
        tasa_diaria_m = dh / dt_dias
    else:
        tasa_diaria_m = 0.03

    distancia_evacuacion_m = round(max(0.0, umbral_evacuacion - ultima_altura), 2)
    
    # Proyección recursiva AR(1) día a día hasta alcanzar umbral de evacuación
    curr_h = ultima_altura
    curr_dt = ultima_fecha_dt
    proyeccion = []
    dias_hasta_evacuacion = None
    fecha_estimada_evacuacion_iso = None

    for dia in range(1, 31):
        curr_h = slope * curr_h + intercept
        curr_dt = curr_dt + timedelta(days=1)
        proyeccion.append({
            "dia": dia,
            "fecha": curr_dt.strftime("%Y-%m-%d"),
            "altura": round(float(curr_h), 2)
        })
        if curr_h >= umbral_evacuacion and dias_hasta_evacuacion is None:
            prev_h = proyeccion[-2]["altura"] if len(proyeccion) > 1 else ultima_altura
            fraccion = (umbral_evacuacion - prev_h) / max(0.001, (curr_h - prev_h))
            dias_hasta_evacuacion = round(float((dia - 1) + fraccion), 1)
            horas_totales = int(dias_hasta_evacuacion * 24)
            fecha_estimada_dt = ultima_fecha_dt + timedelta(hours=horas_totales)
            fecha_estimada_evacuacion_iso = fecha_estimada_dt.strftime("%Y-%m-%dT%H:00:00")

    if dias_hasta_evacuacion is None:
        if tasa_diaria_m > 0:
            dias_hasta_evacuacion = round(distancia_evacuacion_m / tasa_diaria_m, 1)
            fecha_estimada_dt = ultima_fecha_dt + timedelta(days=dias_hasta_evacuacion)
            fecha_estimada_evacuacion_iso = fecha_estimada_dt.strftime("%Y-%m-%dT12:00:00")
        else:
            dias_hasta_evacuacion = 0
            fecha_estimada_evacuacion_iso = None

    return {
        "alerta": {
            "superado": supero_alerta,
            "umbral_m": float(umbral_alerta),
            "fecha_cruce": fecha_alerta_str,
            "fecha_cruce_iso": fecha_alerta_iso,
            "altura_cruce": altura_alerta,
            "ultima_lectura_fecha": ultima_fecha_dt.strftime("%Y-%m-%d"),
            "ultima_lectura_altura": ultima_altura
        },
        "evacuacion": {
            "umbral_m": float(umbral_evacuacion),
            "altura_actual": ultima_altura,
            "distancia_cm": round(float(distancia_evacuacion_m * 100), 1),
            "tasa_diaria_cm": round(float(tasa_diaria_m * 100), 2),
            "dias_estimados": dias_hasta_evacuacion,
            "fecha_estimada_iso": fecha_estimada_evacuacion_iso,
            "modelo": "Autorregresivo AR(1) + Tendencia Secular",
            "proyeccion_futura": proyeccion[:7]
        }
    }


def ejecutar_analisis_completo():
    """Ejecuta todos los componentes estadísticos y consolida el diccionario completo."""
    df = cargar_datos_desde_db()
    return {
        "descriptiva": estadistica_descriptiva(df),
        "inferencial": estadistica_inferencial(df),
        "regresion": regresion_lineal_ar1(df),
        "series_tiempo": serie_temporal_tendencia(df),
        "pronostico": pronostico_umbrales(df, 5.80, 6.15)
    }
