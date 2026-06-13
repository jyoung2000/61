@echo off
REM Inflect Studio launcher (Windows). Creates a venv, installs deps once, runs.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip >nul

if not exist ".venv\.deps_installed" (
    echo Installing PyTorch ^(CUDA 12.1^) ...
    python -c "import torch" 2>nul || python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 || goto :error
    echo Installing requirements ^(this can take a while the first time^) ...
    python -m pip install -r requirements.txt || goto :error
    echo done > ".venv\.deps_installed"
)

python -m inflect
goto :eof

:error
echo.
echo Setup failed. See the messages above.
exit /b 1
