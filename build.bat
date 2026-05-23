@echo off
echo Compiling benchmarks for Ryzen 5 3600X (Zen 2)...
gcc benchmark.c -o benchmark.exe -O3 -march=znver2 -Wall
if %ERRORLEVEL% equ 0 (
    echo Compilation successful. Running benchmarks...
    echo.
    benchmark.exe
) else (
    echo Compilation failed.
)
