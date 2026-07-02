@echo off
echo Compiling benchmarks for Ryzen 5 3600X (Zen 2)...
gcc src\benchmark.c -o bin\benchmark.exe -O3 -march=znver2 -Wall
if %ERRORLEVEL% equ 0 (
    echo Compilation successful. Running benchmarks...
    echo.
    bin\benchmark.exe
) else (
    echo Compilation failed.
)
