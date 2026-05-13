#!/bin/bash

# =========================================
# Script de Instalación de Piper
# Para usar voces de calidad con la app
# =========================================

echo "======================================"
echo "Instalando Piper TTS"
echo "======================================"

# Detectar arquitectura
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)
        ARCH_NAME="x86_64"
        ;;
    armv7l)
        ARCH_NAME="armv7l"
        ;;
    aarch64)
        ARCH_NAME="arm64"
        ;;
    *)
        echo "Arquitectura no soportada: $ARCH"
        exit 1
        ;;
esac

# URL de descarga
VERSION="2024.1.31"
FILENAME="piper_linux_${ARCH_NAME}.tar.gz"
URL="https://github.com/rhasspy/piper/releases/download/${VERSION}/${FILENAME}"

echo "Descargando Piper para $ARCH ($ARCH_NAME)..."
echo "URL: $URL"

# Crear directorio temporal
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Descargar
if ! curl -L -o "$FILENAME" "$URL"; then
    echo "❌ Error descargando Piper"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "✓ Descarga completada"

# Extraer
echo "Extrayendo archivos..."
tar xzf "$FILENAME"

if [ ! -f "piper" ]; then
    echo "❌ El archivo piper no se encontró en el tarball"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Hacer ejecutable
chmod +x piper

# Instalar en /usr/local/bin
echo "Instalando en /usr/local/bin/..."
sudo mv piper /usr/local/bin/piper

# Limpiar
rm -rf "$TEMP_DIR"

# Verificar instalación
if command -v piper &> /dev/null; then
    echo "✅ Piper instalado correctamente!"
    piper --version
else
    echo "❌ Error: Piper no se encontró después de la instalación"
    echo "Intenta ejecutar: which piper"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ Piper TTS listo para usar"
echo "======================================"
echo ""
echo "Ahora ejecuta: python app.py"
