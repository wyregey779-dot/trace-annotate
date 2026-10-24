@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
set "REPO_DIR=%~dp0.."
set "ANNOTATOR_HTML=%REPO_DIR%\trace_annotate\embodied\ui\annotator.html"
if not exist "%ANNOTATOR_HTML%" (
    echo [FAIL] Timeline editor HTML is missing.
    exit /b 2
)
if /I "%~1"=="--check" (
    echo [OK] Editor file exists. Browser execution has not been checked.
    exit /b 0
)
start "" "%ANNOTATOR_HTML%"
if errorlevel 1 (
    echo [FAIL] Browser launch request failed.
    exit /b 1
)
echo [OK] Browser launch requested. Confirm the editor loads before annotating.
exit /b 0
