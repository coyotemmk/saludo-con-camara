#!/bin/bash

# ========================================================================
# Saludo con Cámara - Start Script
# ========================================================================

# Detectar la ruta del script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Cambiar a la carpeta del proyecto
cd "$SCRIPT_DIR" || exit 1


# Elegir intérprete de Python
PYTHON_BIN=""
if [ -x "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "⚠️  No se encontró Python. Instala python3 y vuelve a intentarlo."
    exit 1
fi

echo "✓ Usando Python: $PYTHON_BIN"

# Verificar que existe el modelo
if [ ! -f "models/hand_landmarker.task" ]; then
    echo "⚠️  No se encontró models/hand_landmarker.task"
    echo "Ejecuta setup.sh para descargar el modelo."
    exit 1
fi

# Ejecutar la aplicación
echo "🤖 Iniciando Saludo con Cámara..."
"$PYTHON_BIN" app.py
