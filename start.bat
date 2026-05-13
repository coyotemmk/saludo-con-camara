@echo off
REM ========================================================================
REM Saludo con Cámara - Start Script para Windows
REM ========================================================================

setlocal enabledelayedexpansion

REM Cambiar a la carpeta del script
cd /d "%~dp0" || exit /b 1

echo.
echo =====================================
echo  🤖 Iniciando Saludo con Camara
echo =====================================
echo.

REM Buscar Python en el entorno virtual primero
if exist "venv\Scripts\python.exe" (
    set PYTHON_BIN=venv\Scripts\python.exe
    echo ✓ Python (venv) encontrado
) else if exist ".\venv\Scripts\python.exe" (
    set PYTHON_BIN=.\venv\Scripts\python.exe
    echo ✓ Python (venv) encontrado
) else (
    REM Si no está en venv, buscar en el sistema
    python --version >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_BIN=python
        echo ✓ Python (sistema) encontrado
    ) else (
        echo.
        echo ❌ Error: Python no encontrado o no está en el PATH
        echo.
        echo Opciones para resolver:
        echo 1. Ejecuta primero: setup.bat
        echo 2. O descarga Python desde: https://www.python.org/downloads/
        echo    (marca "Add Python to PATH" durante la instalación)
        echo.
        pause
        exit /b 1
    )
)

REM Verificar si el modelo existe
if not exist "models\hand_landmarker.task" (
    echo ⚠️  Advertencia: Modelo de Hand Landmarker no encontrado
    echo Se descargará automáticamente (puede tardar 1-2 minutos)
)

echo.
echo 🔄 Iniciando aplicación...
echo.

REM Ejecutar la aplicación
!PYTHON_BIN! app.py

if errorlevel 1 (
    echo.
    echo ❌ Error al ejecutar la app
    echo.
)

pause
