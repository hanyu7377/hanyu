# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Kodiak_U%_model_v4.py'],
    pathex=['C:\\ProgramData\\anaconda3'],
    binaries=[],
    datas=[('C:\\ProgramData\\anaconda3\\tcl\\tkdnd2.8','.'),
              ('site1_model_iteration_8.h5', '.')],
    hiddenimports=['tensorflow', 'keras', 'numpy', 'tkinterdnd2', 'tkinter',"scipy"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["flask",'bokeh','sqlite3','sphinx','lxml','qtpy','tables','nacl','notebook','argon2','botocore','parso','docutils','nbformat','jsonschema','cryptography','cloudpickle','llvmlite','platformdirs','lib2to3','jedi','dask','lz4','pyarrow','keyring','sqlalchemy','bcrypt','anyio','pytest','pygments','pycparser','traitlets','jinja2'],
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
    name='Kodiak_U%_model_v4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

