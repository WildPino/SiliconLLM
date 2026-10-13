param(
    [Parameter(Mandatory=$true)][string]$QtPath,
    [Parameter(Mandatory=$true)][string]$CompilerPath,
    [string]$BuildDirectory = (Join-Path $PSScriptRoot 'build'),
    [string]$Destination = (Join-Path $PSScriptRoot 'dist/SiliconLLM')
)
$ErrorActionPreference = 'Stop'
$qtResolved = (Resolve-Path -LiteralPath $QtPath).Path
$compilerResolved = (Resolve-Path -LiteralPath $CompilerPath).Path
$executable = Join-Path $BuildDirectory 'silicon-chat.exe'
if (-not (Test-Path -LiteralPath $executable)) { throw 'Build silicon-chat.exe before packaging.' }
if (Test-Path -LiteralPath $Destination) { throw 'Destination already exists. Choose a new empty destination to avoid mixing releases.' }
New-Item -ItemType Directory -Path $Destination | Out-Null
Copy-Item -LiteralPath $executable -Destination $Destination
$packagedExecutable = Join-Path $Destination 'silicon-chat.exe'
$previousPath = $env:PATH
try {
    $env:PATH = "$compilerResolved;$(Join-Path $qtResolved 'bin');$previousPath"
    & (Join-Path $qtResolved 'bin/windeployqt.exe') --release --no-translations --no-opengl-sw --no-system-d3d-compiler --skip-plugin-types 'generic,networkinformation,tls' $packagedExecutable
    if ($LASTEXITCODE -ne 0) { throw 'Qt deployment failed; this directory is incomplete.' }
    foreach ($runtimeDll in @('libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll')) {
        Copy-Item -LiteralPath (Join-Path $compilerResolved $runtimeDll) -Destination $Destination
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'resources/catalog.example.json') -Destination (Join-Path $Destination 'catalog.example.json')
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'resources/PORTABLE_README.md') -Destination (Join-Path $Destination 'README.md')
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'resources/licenses') -Destination (Join-Path $Destination 'dependency-licenses') -Recurse
    $repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $packagedDocs = Join-Path $Destination 'docs/local-chat'
    New-Item -ItemType Directory -Path $packagedDocs | Out-Null
    Copy-Item -LiteralPath (Join-Path $repositoryRoot 'docs/local-chat/RUNTIME_PROTOCOL.md') -Destination $packagedDocs
    Copy-Item -LiteralPath (Join-Path $repositoryRoot 'LICENSE') -Destination (Join-Path $Destination 'LICENSE.txt')
    $qtLicenseDirectory = Join-Path $qtResolved 'licenses'
    if (Test-Path -LiteralPath $qtLicenseDirectory) { Copy-Item -LiteralPath $qtLicenseDirectory -Destination (Join-Path $Destination 'qt-licenses') -Recurse }
    Write-Output $packagedExecutable
} finally { $env:PATH = $previousPath }
