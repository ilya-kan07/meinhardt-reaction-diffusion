import os
from pathlib import Path

block_cipher = None

def get_resource_files():
    resource_dir = Path('resources/analysis')
    files = []
    for file_path in resource_dir.rglob('*'):
        if file_path.is_file():
            dest = str(file_path.relative_to(Path('resources')))
            files.append((str(file_path), os.path.dirname(dest) or '.'))
    return files

a = Analysis(
    ['run_analysis.py'],
    pathex=[],
    binaries=[],
    datas=get_resource_files(),
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
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
    name='MeinhardtAnalysis',
    debug=False,
    console=False,
    icon='resources/analysis/analysis.ico',
    version='version.txt',
)
