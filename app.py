"""
Servidor Web Flask y API REST para el Dashboard Estadístico del Río Paraná
Asignatura: Estadística / Trabajo Práctico Universitario

Rutas:
- GET /           : Renderiza el dashboard principal (templates/index.html)
- GET /api/datos  : Ejecuta las funciones de analisis.py y devuelve el JSON estructurado
- POST /api/actualizar : Ejecuta nuevamente el scraper para refrescar rio.db
"""

import os
import sys
import importlib.util
from flask import Flask, jsonify, render_template

# Resolver y cargar los módulos directamente para evitar conflictos con carpetas homónimas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _cargar_modulo(nombre, archivo):
    ruta = os.path.join(BASE_DIR, archivo)
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    import analisis
    if not hasattr(analisis, "get_db_path"):
        analisis = _cargar_modulo("analisis_mod", "analisis.py")
except Exception:
    analisis = _cargar_modulo("analisis_mod", "analisis.py")

try:
    import scraper
    if not hasattr(scraper, "scrapear_alturas"):
        scraper = _cargar_modulo("scraper_mod", "scraper.py")
except Exception:
    scraper = _cargar_modulo("scraper_mod", "scraper.py")

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    """Ruta principal que sirve la interfaz visual del dashboard."""
    return render_template("index.html")


def _sanitizar_json(obj):
    """Limpia recursivamente cualquier NaN o Infinity para cumplir estrictamente con RFC 8259 JSON."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitizar_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitizar_json(v) for v in obj]
    return obj


@app.route("/api/datos", methods=["GET"])
def api_datos():
    """
    Ruta API que ejecuta las 4 funciones estadísticas y devuelve los resultados consolidados.
    Si la base de datos aún no existe, intenta inicializarla automáticamente mediante el scraper.
    """
    try:
        db_path = analisis.get_db_path()
        if not os.path.isfile(db_path):
            app.logger.warning("Base de datos no detectada. Inicializando scraper de emergencia...")
            df_inicial = scraper.scrapear_alturas()
            scraper.guardar_en_sqlite(df_inicial)

        # Ejecución de los 4 módulos estadísticos
        df = analisis.cargar_datos_desde_db()
        descriptiva = analisis.estadistica_descriptiva(df)
        inferencial = analisis.estadistica_inferencial(df, alpha=0.05)
        regresion = analisis.regresion_lineal_ar1(df)
        series_tiempo = analisis.serie_temporal_tendencia(df, ventana=30)
        pronostico = analisis.pronostico_umbrales(df, 5.80, 6.15)

        respuesta = {
            "status": "success",
            "metadata": {
                "puerto": "Puerto La Paz (Entre Ríos)",
                "rio": "Río Paraná",
                "fenomeno": "El Niño (ENSO)",
                "total_registros": len(df),
                "fecha_inicio": df["fecha"].min().strftime("%Y-%m-%d"),
                "fecha_fin": df["fecha"].max().strftime("%Y-%m-%d"),
                "umbral_alerta_m": 5.80,
                "umbral_evacuacion_m": 6.15
            },
            "descriptiva": descriptiva,
            "inferencial": inferencial,
            "regresion": regresion,
            "series_tiempo": series_tiempo,
            "pronostico": pronostico
        }
        return jsonify(_sanitizar_json(respuesta)), 200

    except Exception as e:
        app.logger.error(f"Error procesando estadísticas: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error al procesar los análisis estadísticos: {str(e)}"
        }), 500


@app.route("/api/actualizar", methods=["POST"])
def api_actualizar():
    """Ruta para forzar una nueva extracción mediante el scraper."""
    try:
        df = scraper.scrapear_alturas()
        scraper.guardar_en_sqlite(df)
        return jsonify({
            "status": "success",
            "message": f"Base de datos actualizada con {len(df)} registros."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Fallo al actualizar scraper: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("=" * 70)
    print(" INICIANDO SERVIDOR WEB FLASK - TP ESTADÍSTICA (LA PAZ)")
    print(" Acceso local: http://127.0.0.1:5000")
    print(" API Datos:    http://127.0.0.1:5000/api/datos")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=True)
