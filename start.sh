#!/bin/bash

# ========================================================================
# Saludo con Cámara - Start Script
# ========================================================================

# Detectar la ruta del script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Cambiar a la carpeta del proyecto
cd "$SCRIPT_DIR" || exit 1

# Activar entorno virtual
if [ -d "venv" ]; then
    echo "✓ Activando entorno virtual..."
    source venv/bin/activate
else
    echo "⚠️  No se encontró entorno virtual. Ejecuta setup.sh primero."
    exit 1
fi

# Verificar que existe el modelo
if [ ! -f "models/hand_landmarker.task" ]; then
    echo "⚠️  No se encontró models/hand_landmarker.task"
    echo "Ejecuta setup.sh para descargar el modelo."
    exit 1
fi

# Ejecutar la aplicación
echo "🤖 Iniciando Saludo con Cámara..."
python app.py
