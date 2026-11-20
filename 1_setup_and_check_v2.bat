@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set LOG=check_v2.log
echo TurnSense v2 setup and check > %LOG%
echo [%date% %time%] start >> %LOG%
if exist ".venv\Scripts\python.exe" goto havevenv

rem ---- find a 64-bit Python (the neural VAD needs onnxruntime, which has no 32-bit build) ----
set PY=
for %%V in (3.12 3.13 3.11 3.10) do (
  if not defined PY (
    py -%%V-64 -c "import struct,sys;sys.exit(0 if struct.calcsize('P')==8 else 1)" >nul 2>&1 && set PY=py -%%V-64
  )
)
if not defined PY (
  for %%P in ("%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe") do (
    if not defined PY if exist %%P ( %%P -c "import struct,sys;sys.exit(0 if struct.calcsize('P')==8 else 1)" >nul 2>&1 && set PY=%%P )
  )
)
if not defined PY (
  echo No 64-bit Python found: installing Python 3.12 64-bit with winget... >> %LOG%
  echo Installing 64-bit Python 3.12 with winget, please wait...
  winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements >> %LOG% 2>&1
  if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
if not defined PY (
  echo No 64-bit Python available: falling back to the 32-bit one, v2 will run with the v1 BASE settings. >> %LOG%
  set PY="%LOCALAPPDATA%\Programs\Python\Python312-32\python.exe"
)
echo Using Python: !PY! >> %LOG%
!PY! -m venv .venv >> %LOG% 2>&1
:havevenv
".venv\Scripts\python.exe" -c "import struct,sys;print('python',sys.version.split()[0], struct.calcsize('P')*8,'bit')" >> %LOG% 2>&1
echo Installing packages (numpy, websockets, onnxruntime)...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q numpy "websockets>=13" onnxruntime pytest >> %LOG% 2>&1
".venv\Scripts\python.exe" -c "import numpy, websockets; print('DEPS OK numpy', numpy.__version__, 'websockets', websockets.__version__)" >> %LOG% 2>&1
".venv\Scripts\python.exe" -c "import onnxruntime as o; print('ONNXRUNTIME OK', o.__version__)" >> %LOG% 2>&1

echo Offline check on the 24 dev calls...
".venv\Scripts\python.exe" offline_replay.py --out runs\offline_v2 >> %LOG% 2>&1
echo Real-time check on one call over WebSocket (about 30 s)...
".venv\Scripts\python.exe" -m ttbench.play --calls GI-D01 >> %LOG% 2>&1
echo [%date% %time%] done >> %LOG%
echo.
type %LOG%
echo.
echo Check finished. You can close this window.
pause
