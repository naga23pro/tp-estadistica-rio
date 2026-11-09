"""
Carga dinámica de funciones desde analisis.py para resolver el conflicto de nombres
entre la carpeta 'analisis/' y el módulo 'analisis.py'.
"""

import importlib.util
import os
import sys

_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_archivo_py = os.path.join(_raiz, "analisis.py")

if os.path.isfile(_archivo_py):
    spec = importlib.util.spec_from_file_location("analisis_modulo_raiz", _archivo_py)
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    for _attr_name in dir(_mod):
        if not _attr_name.startswith("__"):
            globals()[_attr_name] = getattr(_mod, _attr_name)
