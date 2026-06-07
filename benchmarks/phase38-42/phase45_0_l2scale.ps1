# Phase 45.0 - L2 dynamics scale measurement (calibration for 45.A homeostasis)
#
# Phase 45 changes the L2 WRITE/MAINTENANCE with threshold knobs (norm-clamp,
# update-if-novel, anti-stale-damp, internal-refractory). Those thresholds need the
# SCALE of the L2 state norm, the per-step delta, and how often boundaries are
# redundant or the state goes stale - otherwise they are blind guesses (the 44.E
# caveat-3 trap, x4). This phase MEASURES that scale first. No new C: it reuses the
# generator's --telemetry (already logs l2norm col8, l2delta col9, gate col10).
#
# Output: ||L2|| percentiles, l2delta percentiles, write magnitude (delta on boundary
# steps) vs drift (delta off-boundary), staleness prevalence, redundancy proxy ->
# recommended absolute thresholds for 45.A.
#
# Run:  .\benchmarks\phase38-42\phase45_0_l2scale.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase45_0'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# Configs with active L2 (from earlier phases): D1 (near-stable) + delta champion.
$configs = @(
    @{ label='D1';    file=($WDIR + '\phase44f_F0.bin') },           # mix0.5 scale0.5, near-stable
    @{ label='delta'; file=($WDIR + '\phase44d_D3_delta_s50.bin') }  # teacher-forced champion (unstable)
)
$TEMPS   = @('0.65','0.55')
$NSEEDS  = 4
$GEN_LEN = 2000
$WARMUP  = 5000
$RNG     = 12345

Write-Host ''
Write-Host '=== COMPILING (generator with --telemetry) ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

$present = @($configs | Where-Object { Test-Path $_.file })
if ($present.Count -eq 0) { Write-Error 'No L2 weights present (need 44.D/44.F weights). Run those tribunals first.'; exit 1 }

$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

# -------- Generate telemetry --------------------------------------------------
Write-Host ''
Write-Host ('=== Telemetry: {0} configs x {1} temps x {2} seeds ===' -f $present.Count,$TEMPS.Count,$NSEEDS) -ForegroundColor Yellow
foreach ($c in $present) {
    foreach ($tp in $TEMPS) {
        $si = 0
        foreach ($off in $SEEDS) {
            $si++
            $tel = $RDIR + '\tel_' + $c.label + '_T' + $tp + '_s' + $si + '.tsv'
            $sf  = $RDIR + '\tel_' + $c.label + '_T' + $tp + '_s' + $si + '_stats.txt'
            $al = ('"{0}" "{1}" --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6} --telemetry "{7}"' -f `
                   $TS_DATA,$c.file,$GEN_LEN,$WARMUP,$tp,$off,$RNG,$tel)
            $p = Start-Process -FilePath "$BINDIR\phase44_generator.exe" -ArgumentList $al `
                    -RedirectStandardOutput ($RDIR+'\_null.bin') -RedirectStandardError $sf -NoNewWindow -PassThru -Wait
            Write-Host ('  {0,-7} T{1} s{2} exit={3}' -f $c.label,$tp,$si,$p.ExitCode)
        }
    }
}

# -------- Stats helpers -------------------------------------------------------
function Pctl([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=$a|Sort-Object; $idx=[int][math]::Floor($q*($s.Count-1)); return [math]::Round($s[$idx],4) }
function Mean([double[]]$a){ if($a.Count -eq 0){return 0.0}; [math]::Round(($a|Measure-Object -Average).Average,4) }

# -------- Analyse per (config,temp) -------------------------------------------
Write-Host ''
Write-Host '======== L2 dynamics scale (per config / temp) ========' -ForegroundColor Cyan
$reco = @{}
foreach ($c in $present) {
    foreach ($tp in $TEMPS) {
        $l2n=@(); $l2dAll=@(); $l2dBound=@(); $l2dDrift=@()
        $si=0
        foreach ($off in $SEEDS) {
            $si++
            $tel = $RDIR + '\tel_' + $c.label + '_T' + $tp + '_s' + $si + '.tsv'
            if (-not (Test-Path $tel)) { continue }
            foreach ($line in [System.IO.File]::ReadLines($tel)) {
                if ($line.Length -eq 0 -or $line[0] -eq '#' -or $line[0] -eq 's') { continue }
                $f = $line -split "`t"; if ($f.Count -lt 14) { continue }
                $n=[double]$f[8]; $d=[double]$f[9]; $g=[int]$f[10]
                $l2n += $n; $l2dAll += $d
                if ($g -eq 1) { $l2dBound += $d } else { $l2dDrift += $d }
            }
        }
        if ($l2n.Count -eq 0) { continue }
        $nMean=Mean $l2n; $nP50=Pctl $l2n 0.50; $nP90=Pctl $l2n 0.90; $nP99=Pctl $l2n 0.99
        $dMean=Mean $l2dAll; $dP50=Pctl $l2dAll 0.50; $dP90=Pctl $l2dAll 0.90
        $wMean=Mean $l2dBound; $wP50=Pctl $l2dBound 0.50      # write magnitude (boundary delta)
        $drMean=Mean $l2dDrift                                  # drift between boundaries
        # staleness: fraction of steps with delta below 25% of median delta
        $staleThr = 0.25*$dP50
        $stale = @($l2dAll | Where-Object { $_ -lt $staleThr }).Count
        $stalePct = [math]::Round(100.0*$stale/$l2dAll.Count,1)
        Write-Host ''
        Write-Host ('  {0}  T={1}   (n={2} steps)' -f $c.label,$tp,$l2n.Count) -ForegroundColor Yellow
        Write-Host ('    ||L2||   mean={0,7}  p50={1,7}  p90={2,7}  p99={3,7}' -f $nMean,$nP50,$nP90,$nP99)
        Write-Host ('    l2delta  mean={0,7}  p50={1,7}  p90={2,7}' -f $dMean,$dP50,$dP90)
        Write-Host ('    write(boundary delta) mean={0,7} p50={1,7}  | drift(off) mean={2,7}' -f $wMean,$wP50,$drMean)
        Write-Host ('    staleness (delta < 25%*p50={0}) = {1}% of steps' -f ([math]::Round($staleThr,4)),$stalePct)
        $reco[$c.label+'_'+$tp] = [PSCustomObject]@{ nP90=$nP90; nP99=$nP99; dP50=$dP50; wP50=$wP50; staleThr=$staleThr }
    }
}

# -------- Recommended 45.A thresholds (from D1 @ T=0.55 if present) ------------
Write-Host ''
Write-Host '======== Recommended 45.A thresholds (anchor: D1 @ T=0.55) ========' -ForegroundColor Cyan
$anchor = $reco['D1_0.55']
if ($null -eq $anchor) { $anchor = ($reco.Values | Select-Object -First 1) }
if ($null -ne $anchor) {
    Write-Host ('  NC norm-clamp Cmax   ~ p90..p99 ||L2||   = {0} .. {1}' -f $anchor.nP90,$anchor.nP99) -ForegroundColor Green
    Write-Host ('  AS anti-stale eps    ~ 0.25*p50 delta    = {0}     (+ stale_k ~ 16-32 step)' -f ([math]::Round($anchor.staleThr,4))) -ForegroundColor Green
    Write-Host ('  UN update-if-novel   ~ 0.5..1.0 * write   = {0} .. {1}' -f ([math]::Round(0.5*$anchor.wP50,4)),$anchor.wP50) -ForegroundColor Green
    Write-Host ('  IR internal-refr thr ~ p99 ||L2||         = {0}     (+ refr ~ 8-16 step)' -f $anchor.nP99) -ForegroundColor Green
    Write-Host ('  WN write-normalize   target norm ~ p50 write = {0}' -f $anchor.wP50) -ForegroundColor Green
}
Write-Host ''
Write-Host '  Confronta D1 (near-stable) vs delta (champion instabile): se ||L2|| o staleness'
Write-Host '  sono molto piu alti su delta, e li che nasce la pressione fissa -> 45.A mira li.'
Write-Host ''
Write-Host '  Questi numeri calibrano i knob 45.A (self-calibrating dai dati, non a caso).'
Write-Host ('  Files: ' + $RDIR)
