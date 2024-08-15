# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['APP_main_v11.py'],
    pathex=['C:\\ProgramData\\anaconda3'],
    binaries=[],
    datas=[('C:\\ProgramData\\anaconda3\\tcl\\tkdnd2.8', 'tkdnd2.8')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow','bokeh','sqlite3','sphinx','lxml','qtpy','tables','nacl','scipy','notebook','argon2','botocore','parso','docutils','nbformat','jsonschema','cryptography','cloudpickle','llvmlite','platformdirs','lib2to3','jedi','dask','lz4','h5py','pyarrow','keyring','scikit-learn','sqlalchemy','bcrypt','anyio','pytest','pygments','pycparser','difflib','traitlets','jinja2','openpyxl'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='APP_main_v11',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
