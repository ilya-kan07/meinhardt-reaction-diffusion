from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

project_root = Path.cwd()

# Иконка необязательна: если файла нет, сборка не падает
icon_path = project_root / 'resources' / 'bifurcation' / 'bifurcation.ico'
icon = str(icon_path) if icon_path.exists() else None


def get_resource_files():
    """Ресурсы bifurcation (если появятся). Сейчас в папке только иконка."""
    resource_dir = project_root / 'resources' / 'bifurcation'
    if not resource_dir.exists():
        return []
    return [(str(src), 'resources/bifurcation')
            for src in resource_dir.rglob('*') if src.is_file()]


datas = get_resource_files()

# blosc2 нужен модулю solver/src/database.py (общая БД с solver)
datas += collect_data_files('blosc2')
binaries = collect_dynamic_libs('blosc2')

a = Analysis(
    ['run_bifurcation.py'],
    pathex=[str(project_root)],   # чтобы нашлись пакеты bifurcation/ и solver/
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'sqlite3',
        'blosc2',
        'scipy.optimize',
        'scipy.interpolate',
        'openpyxl',
        'pandas',
        # matplotlib (TkAgg) подгружает PIL.ImageTk динамически — PyInstaller этого не видит
        'PIL.ImageTk',
        'PIL._tkinter_finder',
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
    name='MeinhardtBifurcation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,               # GUI без консоли
    icon=icon,
    version='bifurcation_version.txt',
)
