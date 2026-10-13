param(
    [Parameter(Mandatory=$true)][string]$QtPath,
    [Parameter(Mandatory=$true)][string]$CompilerPath,
    [string]$BuildDirectory = (Join-Path $PSScriptRoot 'build'),
    [switch]$SkipTests
)
$ErrorActionPreference = 'Stop'
$qtResolved = (Resolve-Path -LiteralPath $QtPath).Path
$compilerResolved = (Resolve-Path -LiteralPath $CompilerPath).Path
$compilerExecutable = Join-Path $compilerResolved 'g++.exe'
if (-not (Test-Path -LiteralPath $compilerExecutable)) { throw "Compiler missing: $compilerExecutable" }
if (-not (Test-Path -LiteralPath (Join-Path $qtResolved 'lib/cmake/Qt6/Qt6Config.cmake'))) { throw 'QtPath must point to the Qt SDK directory (e.g. mingw_64).' }
$previousPath = $env:PATH
try {
    $env:PATH = "$compilerResolved;$(Join-Path $qtResolved 'bin');$previousPath"
    $testsEnabled = if ($SkipTests) { 'OFF' } else { 'ON' }
    & cmake -S $PSScriptRoot -B $BuildDirectory -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_PREFIX_PATH=$qtResolved" "-DCMAKE_CXX_COMPILER=$compilerExecutable" "-DBUILD_TESTING=$testsEnabled"
    if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
    & cmake --build $BuildDirectory --parallel 1
    if ($LASTEXITCODE -ne 0) { throw 'Desktop compilation failed.' }
    if (-not $SkipTests) {
        & ctest --test-dir $BuildDirectory --output-on-failure --parallel 1
        if ($LASTEXITCODE -ne 0) { throw 'Desktop tests failed.' }
    }
    Write-Output (Join-Path $BuildDirectory 'silicon-chat.exe')
} finally { $env:PATH = $previousPath }
