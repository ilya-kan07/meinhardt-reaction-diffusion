import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

def get_resource_files():
    project_root = Path.cwd()
    resource_dir = project_root / 'resources' / 'solver'

    if not resource_dir.exists():
        print(f"[WARNING] Папка ресурсов не найдена: {resource_dir}")
        return []

    files = []
    for src in resource_dir.rglob('*'):
        if src.is_file():
            files.append((str(src), 'resources/solver'))
    return files

# Собираем ресурсы (картинки и т.д.)
datas = get_resource_files()

# Добавляем blosc2 (очень важно для Windows!)
datas += collect_data_files('blosc2')

# Если будут проблемы с numpy — можно добавить явно
# datas += collect_data_files('numpy')

a = Analysis(
    ['run_solver.py'],  # ← твой главный файл запуска
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'sqlite3',
        'blosc2',
        'numpy.core._multiarray_umath',
        'numpy.core.multiarray',
        # иногда нужны дополнительные
        'numpy.linalg._umath_linalg',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MeinhardtSolver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                    # опционально
    console=False,               # GUI без консоли
    icon='resources/solver/solver.ico',  # если есть
    # version='version.txt',     # если есть
)
