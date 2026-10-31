@echo off
REM ============================================================================
REM Francisv2 GPT-SoVITS Headless Streaming API Server Launcher
REM Free to use with attribution to Francisv2 by Francis Pike
REM ============================================================================

if "%GPT_SOVITS_ROOT%"=="" set "GPT_SOVITS_ROOT=C:\GPT-SoVITS"

if not exist "%GPT_SOVITS_ROOT%\api_v2.py" (
    echo [ERROR] Could not find "%GPT_SOVITS_ROOT%\api_v2.py".
    echo Set GPT_SOVITS_ROOT to your GPT-SoVITS folder path and try again.
    pause
    exit /b 1
)

set "PATH=%GPT_SOVITS_ROOT%\runtime;%PATH%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo Syncing Francisv2 weights and reference audio...
"%GPT_SOVITS_ROOT%\runtime\python.exe" "%~dp0download_weights.py" --gpt-sovits-dir "%GPT_SOVITS_ROOT%"

echo Starting GPT-SoVITS api_v2.py on 0.0.0.0:9880 ...
cd /d "%GPT_SOVITS_ROOT%"
"%GPT_SOVITS_ROOT%\runtime\python.exe" api_v2.py -a 0.0.0.0 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
