@echo off
REM S³ — Spectral-Spatial-Smooth Fine-Tuning
REM Complete workflow from SVD to benchmarks (Windows batch equivalent of Makefile)

setlocal enabledelayedexpansion

REM Default variables
set VENV=.venv
set PYTHON=%VENV%\Scripts\python.exe
set PIP=%VENV%\Scripts\pip.exe
set MODEL=Qwen/Qwen2.5-7B-Instruct
set K=128
set CONFIG=experiments/configs/s3_standard.yaml
set DEVICE=cuda:0
set ABLATION=s3_full
set CKPT=./checkpoints/s3_qwen7b_alpaca/checkpoint.pt

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="help" goto :help
if /i "%~1"=="venv" goto :venv
if /i "%~1"=="svd" goto :svd
if /i "%~1"=="train" goto :train
if /i "%~1"=="train-monitored" goto :train_monitored
if /i "%~1"=="train-ablation" goto :train_ablation
if /i "%~1"=="evaluate" goto :evaluate
if /i "%~1"=="all" goto :all
if /i "%~1"=="clean" goto :clean
if /i "%~1"=="test" goto :test
if /i "%~1"=="setup" goto :setup
if /i "%~1"=="MODEL" set MODEL=%~2&shift&shift&goto :parse_args
if /i "%~1"=="K" set K=%~2&shift&shift&goto :parse_args
if /i "%~1"=="CONFIG" set CONFIG=%~2&shift&shift&goto :parse_args
if /i "%~1"=="DEVICE" set DEVICE=%~2&shift&shift&goto :parse_args
if /i "%~1"=="ABLATION" set ABLATION=%~2&shift&shift&goto :parse_args
if /i "%~1"=="CKPT" set CKPT=%~2&shift&shift&goto :parse_args
shift&goto :parse_args

:end_parse

REM Check if virtual environment exists
:ensure_venv
if not exist "%VENV%" (
    echo [Error] Virtual environment not found. Run: bat_s3.bat venv
    exit /b 1
)
goto :eof

REM Help command
:help
echo S³ Fine-Tuning Pipeline (Windows)
echo.
echo Usage:
echo   bat_s3.bat venv          Create virtual environment + install dependencies
echo   bat_s3.bat svd           Pre-compute randomized SVD
echo   bat_s3.bat train         Train S³ on Alpaca
echo   bat_s3.bat train-monitored  Train with full monitoring
echo   bat_s3.bat evaluate      Run benchmarks on trained model
echo   bat_s3.bat all           Full pipeline: venv ^→ svd ^→ train ^→ evaluate
echo   bat_s3.bat clean         Remove checkpoints and results
echo   bat_s3.bat test          Run unit tests
echo.
echo Variables (use VAR=value syntax):
echo   MODEL              HuggingFace model (default: Qwen/Qwen2.5-7B-Instruct)
echo   K                  SVD rank (default: 128)
echo   CONFIG             YAML config (default: experiments/configs/s3_standard.yaml)
echo   DEVICE             GPU device (default: cuda:0)
echo   ABLATION           Ablation study name (default: s3_full)
echo   CKPT               Checkpoint path (default: ./checkpoints/s3_qwen7b_alpaca/checkpoint.pt)
echo.
echo Examples:
echo   bat_s3.bat svd MODEL=Qwen/Qwen2.5-7B-Instruct K=128
echo   bat_s3.bat train DEVICE=cuda:0
echo   bat_s3.bat train-ablation ABLATION=s3_no_stb
goto :eof

REM Create virtual environment and install dependencies
:venv
echo [Venv] Creating virtual environment...
if not exist "%VENV%" (
    python -m venv %VENV%
    echo [Venv] Created %VENV%
) else (
    echo [Venv] Already exists: %VENV%
)
echo [Venv] Installing dependencies...
%PIP% install --upgrade pip
%PIP% install -r requirements.txt
echo [Venv] Ready.
goto :eof

REM Compute randomized SVD
:svd
call :ensure_venv
echo [SVD] Computing randomized SVD for %MODEL% (k=%K%)...
%PYTHON% src/utils/randomized_svd.py ^
    --model %MODEL% ^
    --k %K% ^
    --output ./svd_factors/ ^
    --device cpu
goto :eof

REM Train S³
:train
call :ensure_venv
echo [Train] Starting frugal training...
%PYTHON% train_s3.py ^
    --config %CONFIG% ^
    --device %DEVICE%
goto :eof

REM Train with monitoring
:train_monitored
call :ensure_venv
echo [Train+Monitor] Starting frugal training with FULL system monitoring...
echo [Monitor] Logs: results/monitoring/
%PYTHON% train_s3.py ^
    --config %CONFIG% ^
    --device %DEVICE%
goto :eof

REM Ablation studies
:train_ablation
call :ensure_venv
echo [Train] Ablation: %ABLATION%...
%PYTHON% train_s3.py ^
    --config experiments/configs/s3_standard.yaml ^
    --device %DEVICE% ^
    --ablation %ABLATION%
goto :eof

REM Evaluate trained model
:evaluate
call :ensure_venv
echo [Eval] Running benchmarks...
%PYTHON% train_s3.py ^
    --config %CONFIG% ^
    --device %DEVICE% ^
    --eval_only ^
    --checkpoint %CKPT%
goto :eof

REM Full pipeline
:all
call :venv
call :svd
call :train
call :evaluate
echo [S³] Full pipeline complete.
goto :eof

REM Run tests
:test
call :ensure_venv
echo [Test] Running unit tests...
%VENV%\Scripts\pytest.exe tests/ -v --tb=short
goto :eof

REM Clean checkpoints and results
:clean
echo [Clean] Removing checkpoints and results...
if exist ".\checkpoints\" rmdir /s /q .\checkpoints
if exist ".\results\" rmdir /s /q .\results
echo [Clean] Done.
goto :eof

REM Setup (venv + svd)
:setup
call :venv
call :svd
echo [Setup] Ready for training.
goto :eof

endlocal