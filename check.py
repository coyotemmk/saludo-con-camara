#!/usr/bin/env python
"""
Validador de Saludo con Cámara - Verifica que todas las dependencias estén instaladas
"""

import sys
import importlib
from pathlib import Path

def check_module(name, import_name=None):
    """Verifica si un módulo está instalado."""
    if import_name is None:
        import_name = name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {name}")
        return True
    except ImportError:
        print(f"❌ {name} - NO INSTALADO")
        return False

def check_file(path, description):
    """Verifica si un archivo existe."""
    if Path(path).exists():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - NO ENCONTRADO")
        return False

def main():
    print("=" * 60)
    print("🔍 Validador: Saludo con Cámara")
    print("=" * 60)
    print()
    
    # Verificar versión de Python
    print(f"📌 Python {sys.version.split()[0]}")
    if sys.version_info >= (3, 8):
        print("✅ Versión de Python compatible\n")
    else:
        print("❌ Se requiere Python 3.8+\n")
        return False
    
    # Verificar módulos Python
    print("📦 Módulos Python:")
    modules_ok = all([
        check_module("OpenCV", "cv2"),
        check_module("MediaPipe", "mediapipe"),
        check_module("pyttsx3"),
        check_module("NumPy", "numpy"),
        check_module("PIL", "PIL"),
    ])
    print()
    
    # Verificar archivos y carpetas
    print("📂 Archivos y carpetas:")
    files_ok = all([
        check_file("app.py", "app.py (aplicación principal)"),
        check_file("requirements.txt", "requirements.txt (dependencias)"),
        check_file("faces", "faces/ (carpeta de animaciones)"),
    ])
    print()
    
    # Verificar modelo
    print("🤖 Modelos:")
    model_ok = check_file("models/hand_landmarker.task", 
                          "models/hand_landmarker.task (Hand Landmarker)")
    if not model_ok:
        print("   💡 Se descargará automáticamente al ejecutar app.py\n")
    else:
        print()
    
    # Resultado final
    print("=" * 60)
    if modules_ok and files_ok:
        print("✅ ¡TODO OK! La app debería funcionar.")
        print("\nPara ejecutar:")
        print("  - Windows: python app.py")
        print("  - Linux:   ./start.sh")
    else:
        print("❌ Hay problemas. Ejecuta:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return modules_ok and files_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
