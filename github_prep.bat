@echo off
echo Cleaning up project for GitHub...

REM Remove backup directory
echo Removing backup directory...
rmdir /S /Q backup 2>nul

REM Remove pycache
echo Removing Python cache files...
rmdir /S /Q __pycache__ 2>nul
for /d /r %%d in (*__pycache__*) do @if exist "%%d" rd /s /q "%%d"

REM Remove IDE specific folders
echo Removing IDE specific files...
rmdir /S /Q .qodo 2>nul

REM Remove temporary files
echo Removing temporary and generated files...
del /F /Q sentiment_probabilities.png 2>nul

REM Handle checkpoint directories - remove large model files but keep directories
echo Handling model checkpoints...
if not exist checkpoints mkdir checkpoints
if not exist improved_checkpoints mkdir improved_checkpoints

echo Removing large model checkpoint files from git tracking...
del /F /Q checkpoints\*.pth 2>nul
del /F /Q improved_checkpoints\*.pth 2>nul

REM Keep only metadata and training metrics in checkpoints
echo Keeping only metadata and useful visualizations...
echo. > checkpoints\.gitkeep
echo. > improved_checkpoints\.gitkeep

REM Handle duplicate test files
echo Handling duplicate test files...
echo Comparing test_system.py and test_full_system.py

REM We're keeping both for now, but making the difference clear
echo. > test_system.py.info
echo This file provides basic unit tests for the model and data loading >> test_system.py.info
echo. > test_full_system.py.info
echo This file provides integration tests for the full system workflow >> test_full_system.py.info

REM Move cleanup.bat to backup before removing
if exist cleanup.bat (
    echo Moving old cleanup.bat to cleanup.bat.bak...
    rename cleanup.bat cleanup.bat.bak
)

echo.
echo Cleanup complete! Project is now ready for GitHub.
echo.
echo Final tasks:
echo   1. Review files (especially test_system.py and test_full_system.py)
echo   2. Commit changes to Git using:
echo      git add .
echo      git commit -m "Prepare project for GitHub"
echo      git push
echo. 