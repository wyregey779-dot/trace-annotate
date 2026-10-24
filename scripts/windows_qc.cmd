@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "REPO_DIR=%~dp0.."
set "TARGET=%~f1"
set "N_STEPS=%~2"
set "NO_OPEN=%~3"
if "%~1"=="" (
    echo Usage: drag an annotation JSON/JSONL here, or run this file with input [expected-step-count] [--no-open].
    exit /b 2
)
if /I "%N_STEPS%"=="--no-open" (
    set "NO_OPEN=--no-open"
    set "N_STEPS="
)
if not exist "%TARGET%" (
    echo [FAIL] Input does not exist.
    exit /b 2
)
if not exist "%REPO_DIR%\trace_annotate\cli.py" (
    echo [FAIL] TraceAnnotate source is missing.
    exit /b 2
)
where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Python is not available on PATH.
    exit /b 2
)
set "REPORT_ROOT=%REPO_DIR%\..\qc_runs"
if defined TRACE_QC_REPORT_ROOT set "REPORT_ROOT=%TRACE_QC_REPORT_ROOT%"
if not exist "%REPORT_ROOT%" mkdir "%REPORT_ROOT%"
if not exist "%REPORT_ROOT%" (
    echo [FAIL] Cannot create report directory.
    exit /b 2
)
:choose_report
set "RUN_DIR=%REPORT_ROOT%\run-%RANDOM%-%RANDOM%"
if exist "%RUN_DIR%" goto choose_report
mkdir "%RUN_DIR%"
if errorlevel 1 exit /b 2
set "REPORT=%RUN_DIR%\qc_report.html"
pushd "%REPO_DIR%"
if errorlevel 1 exit /b 2
if defined N_STEPS (
    python -B -m trace_annotate.cli embodied-pipeline "%TARGET%" --auto-idle --nsteps "%N_STEPS%" --report "%REPORT%"
) else (
    python -B -m trace_annotate.cli embodied-pipeline "%TARGET%" --auto-idle --report "%REPORT%"
)
set "QC_EXIT=%ERRORLEVEL%"
popd
if not exist "%REPORT%" (
    echo [FAIL] No new report was produced. Old reports were not opened.
    if "%QC_EXIT%"=="0" exit /b 2
    exit /b %QC_EXIT%
)
echo Report: "%REPORT%"
if not "%QC_EXIT%"=="0" (
    echo [FAIL] Validation or review gate did not pass. Exit code: %QC_EXIT%
    exit /b %QC_EXIT%
)
echo [OK] Automated gates passed for this batch. Video content requires independent review.
if /I not "%NO_OPEN%"=="--no-open" start "" "%REPORT%"
exit /b 0
