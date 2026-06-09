# Phase 45.0 - L2 dynamics scale measurement (calibration for 45.A homeostasis)
#
# UPDATE (event-conditioned): 45.0's first pass showed L2 is EVENT-SPARSE (it only
# moves on boundary writes), so GLOBAL percentiles over all steps are invalid - they
# are dominated by the long flat stretches between writes. This version conditions
# every statistic on the actual write events and reports nonzero-only quantiles.
#
# It also drops raw ||L2|| as a "pressure" proxy: D1 sits at ||L2||~128k yet is the
# STABLE config, while the delta champion sits at ||L2||~13 and is the UNSTABLE one.
# Raw norm is not the cause. The discriminating signals are scale-free:
#   - relative state move  l2delta/l2norm  (D1 ~1e-4 frozen vs delta ~1.0 volatile)
#   - logit-space ratio    col 'ratio'     (L2/SEE logit-norm, already scale-free)
# plus the write-vs-realized-move gap (wnorm vs l2delta) that reveals the EMA regime.
#
# Telemetry columns (added wnorm/wrote/gfired in the generator, aligned with l2delta):
#   8 l2norm  9 l2delta  10 gate  11 Lwin  ... 14 wnorm  15 wrote  16 gfired
#   wnorm  = ||p||, raw projection norm of the incoming write (pre-EMA)
#   wrote  = 1 if l2_evolve actually updated L2 (gate fired AND cooldown clear)
#   gfired = the gate decision that evolve saw
#   l2delta = ||L2_t - L2_{t-1}||, the REALIZED state change from that same evolve
#
# 45.A thresholds must come from: p25/p50 of nonzero writes; p10/p25 of nonzero
# boundary deltas; the run-length of missed-update gaps; and the EFFECTIVE ratio/gamma
# - never from a global p50 diluted by flat steps.
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

# Configs with active L2: D1 (near-stable) + delta champion (unstable).
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
Write-Host '=== COMPILING (generator with --telemetry + write diag) ===' -ForegroundColor Cyan
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
function Mean([double[]]$a){ if($a.Count -eq 0){return 0.0}; [math]::Round(($a|Measure-Object -Average).Average,4) }
function Q([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=@($a|Sort-Object); $idx=[int][math]::Floor($q*($s.Count-1)); [math]::Round($s[$idx],4) }
# nonzero quantile band p10/p25/p50/p75/p90/p99 as a formatted string
function QBand([double[]]$a){
    if($a.Count -eq 0){ return '(empty)' }
    ('p10={0} p25={1} p50={2} p75={3} p90={4} p99={5}' -f (Q $a 0.10),(Q $a 0.25),(Q $a 0.50),(Q $a 0.75),(Q $a 0.90),(Q $a 0.99))
}

# Parse one telemetry file into ordered per-step columns (one seed).
function Read-Tel([string]$path){
    $l2d=New-Object System.Collections.Generic.List[double]
    $l2n=New-Object System.Collections.Generic.List[double]
    $wn =New-Object System.Collections.Generic.List[double]
    $wr =New-Object System.Collections.Generic.List[int]
    $gf =New-Object System.Collections.Generic.List[int]
    $rt =New-Object System.Collections.Generic.List[double]
    $gm =New-Object System.Collections.Generic.List[double]
    $fl =New-Object System.Collections.Generic.List[int]
    $lw =New-Object System.Collections.Generic.List[double]
    foreach ($line in [System.IO.File]::ReadLines($path)) {
        if ($line.Length -eq 0 -or $line[0] -eq '#' -or $line[0] -eq 's') { continue }
        $f = $line -split "`t"; if ($f.Count -lt 17) { continue }
        $rt.Add([double]$f[5]); $gm.Add([double]$f[6]); $fl.Add([int]$f[4])
        $l2n.Add([double]$f[8]); $l2d.Add([double]$f[9]); $lw.Add([math]::Abs([double]$f[11]))
        $wn.Add([double]$f[14]); $wr.Add([int]$f[15]); $gf.Add([int]$f[16])
    }
    return [PSCustomObject]@{ l2d=$l2d; l2n=$l2n; wn=$wn; wr=$wr; gf=$gf; rt=$rt; gm=$gm; fl=$fl; lw=$lw }
}

# -------- Analyse per (config,temp), event-conditioned ------------------------
Write-Host ''
Write-Host '======== L2 event-conditioned dynamics (per config / temp) ========' -ForegroundColor Cyan
$reco = @{}
foreach ($c in $present) {
    foreach ($tp in $TEMPS) {
        # pooled scalars + per-seed gap lengths
        $P_l2n=@(); $P_rt=@(); $P_gm=@(); $P_lw=@(); $P_fl=@()
        $P_l2d_gf1=@(); $P_l2d_gf0=@(); $P_l2d_nz=@()
        $P_wn_wr1=@(); $P_wn_nz=@(); $P_rel_nz=@()
        $gaps=@()
        $nSteps=0; $nGate=0; $nWrote=0; $nCdBlocked=0; $nFlip=0
        $si=0
        foreach ($off in $SEEDS) {
            $si++
            $tel = $RDIR + '\tel_' + $c.label + '_T' + $tp + '_s' + $si + '.tsv'
            if (-not (Test-Path $tel)) { continue }
            $d = Read-Tel $tel
            $n = $d.l2d.Count; if ($n -eq 0) { continue }
            $nSteps += $n
            $writeIdx = New-Object System.Collections.Generic.List[int]
            for ($k=0; $k -lt $n; $k++) {
                $dd=$d.l2d[$k]; $nn=$d.l2n[$k]; $ww=$d.wn[$k]; $wf=$d.wr[$k]; $gff=$d.gf[$k]
                $P_l2n += $nn; $P_rt += $d.rt[$k]; $P_gm += $d.gm[$k]; $P_lw += $d.lw[$k]; $P_fl += $d.fl[$k]
                if ($d.fl[$k] -eq 1) { $nFlip++ }
                if ($gff -eq 1) { $nGate++;  $P_l2d_gf1 += $dd } else { $P_l2d_gf0 += $dd }
                if ($wf  -eq 1) { $nWrote++; $P_wn_wr1 += $ww; $writeIdx.Add($k) }
                if ($gff -eq 1 -and $wf -eq 0) { $nCdBlocked++ }
                if ($dd -gt 0) { $P_l2d_nz += $dd; if ($nn -gt 1e-9) { $P_rel_nz += ($dd/$nn) } }
                if ($ww -gt 0) { $P_wn_nz += $ww }
            }
            # gap = consecutive-zero run length between nonzero writes (this seed only)
            for ($k=1; $k -lt $writeIdx.Count; $k++) { $gaps += ($writeIdx[$k]-$writeIdx[$k-1]-1) }
        }
        if ($nSteps -eq 0) { continue }
        $gateFreq = [math]::Round(100.0*$nGate/$nSteps,2)
        $writeFreq= [math]::Round(100.0*$nWrote/$nSteps,2)
        $evPerK   = [math]::Round(1000.0*$nWrote/$nSteps,2)
        $flipRate = [math]::Round(100.0*$nFlip/$nSteps,2)

        Write-Host ''
        Write-Host ('  {0}  T={1}   (n={2} steps, {3} seeds)' -f $c.label,$tp,$nSteps,$NSEEDS) -ForegroundColor Yellow
        Write-Host ('    events: gate(gfired=1)={0} ({1}%)  write(wrote=1)={2} ({3}%)  cooldown-blocked={4}  events/1000step={5}' -f `
            $nGate,$gateFreq,$nWrote,$writeFreq,$nCdBlocked,$evPerK)
        Write-Host ('    l2delta | gfired=1 : mean={0,-9} {1}' -f (Mean $P_l2d_gf1),(QBand $P_l2d_gf1))
        $gf0nz = @($P_l2d_gf0 | Where-Object { $_ -gt 0 }).Count
        Write-Host ('    l2delta | gfired=0 : mean={0,-9} nonzero={1}/{2}  (>0 only if non-boundary decay is on)' -f (Mean $P_l2d_gf0),$gf0nz,$P_l2d_gf0.Count)
        Write-Host ('    l2delta | >0       : mean={0,-9} {1}   (n={2})' -f (Mean $P_l2d_nz),(QBand $P_l2d_nz),$P_l2d_nz.Count)
        Write-Host ('    wnorm   | wrote=1  : mean={0,-9} {1}' -f (Mean $P_wn_wr1),(QBand $P_wn_wr1))
        Write-Host ('    wnorm   | >0       : mean={0,-9} (n={1})' -f (Mean $P_wn_nz),$P_wn_nz.Count)
        if ($gaps.Count -gt 0) {
            Write-Host ('    gap(zeros between writes): mean={0} p50={1} p90={2} max={3}' -f (Mean $gaps),(Q $gaps 0.50),(Q $gaps 0.90),(Q $gaps 1.0))
        } else {
            Write-Host '    gap(zeros between writes): (no internal gaps)'
        }
        Write-Host  '    SCALE-FREE PRESSURE:' -ForegroundColor DarkCyan
        Write-Host ('      rel move l2delta/l2norm | >0 : p25={0} p50={1} p90={2}   (D1 frozen ~1e-4, delta volatile ~1)' -f (Q $P_rel_nz 0.25),(Q $P_rel_nz 0.50),(Q $P_rel_nz 0.90))
        Write-Host ('      logit ratio (col5)           : mean={0} p50={1} p90={2}' -f (Mean $P_rt),(Q $P_rt 0.50),(Q $P_rt 0.90))
        Write-Host ('    gamma mean={0}  flip-rate={1}%  |Lwin| mean={2}  ||L2|| raw p50={3} (NOT pressure)' -f `
            (Mean $P_gm),$flipRate,(Mean $P_lw),(Q $P_l2n 0.50))

        $reco[$c.label+'_'+$tp] = [PSCustomObject]@{
            wnNzP25=(Q $P_wn_nz 0.25); wnNzP50=(Q $P_wn_nz 0.50);
            dNzP10=(Q $P_l2d_nz 0.10); dNzP25=(Q $P_l2d_nz 0.25); dNzP50=(Q $P_l2d_nz 0.50);
            relP50=(Q $P_rel_nz 0.50); relP90=(Q $P_rel_nz 0.90);
            ratioMean=(Mean $P_rt); gammaMean=(Mean $P_gm); flipRate=$flipRate; lwMean=(Mean $P_lw);
            gapP90=(Q $gaps 0.90); l2nP50=(Q $P_l2n 0.50);
            writeFreq=$writeFreq
        }
    }
}

# -------- D1 vs delta discriminator (scale-free) ------------------------------
Write-Host ''
Write-Host '======== D1 (stable) vs delta (unstable) - scale-free discriminators ========' -ForegroundColor Cyan
Write-Host ('  {0,-14} {1,12} {2,12}' -f 'signal','D1@T0.55','delta@T0.55')
Write-Host ('  ' + '-'*40)
$a = $reco['D1_0.55']; $b = $reco['delta_0.55']
if ($null -ne $a -and $null -ne $b) {
    Write-Host ('  {0,-14} {1,12} {2,12}' -f 'relMove p50',  $a.relP50,   $b.relP50)
    Write-Host ('  {0,-14} {1,12} {2,12}' -f 'relMove p90',  $a.relP90,   $b.relP90)
    Write-Host ('  {0,-14} {1,12} {2,12}' -f 'logitRatio',   $a.ratioMean,$b.ratioMean)
    Write-Host ('  {0,-14} {1,12} {2,12}' -f 'gamma',        $a.gammaMean,$b.gammaMean)
    Write-Host ('  {0,-14} {1,12} {2,12}' -f 'flip-rate %',  $a.flipRate, $b.flipRate)
    Write-Host ('  {0,-14} {1,12} {2,12}' -f '|Lwin| mean',  $a.lwMean,   $b.lwMean)
    Write-Host ('  {0,-14} {1,12} {2,12}' -f 'write d nzP50',$a.dNzP50,   $b.dNzP50)
    Write-Host ('  {0,-14} {1,12} {2,12}' -f 'wnorm nzP50',  $a.wnNzP50,  $b.wnNzP50)
    Write-Host ('  {0,-14} {1,12} {2,12}' -f 'l2norm p50',   $a.l2nP50,   $b.l2nP50)
    Write-Host ('  {0,-14} {1,12} {2,12}' -f 'write freq %', $a.writeFreq,$b.writeFreq)
    Write-Host ''
    Write-Host '  Read: if relMove is tiny on D1 and ~1 on delta, the instability is L2 VOLATILITY' -ForegroundColor Gray
    Write-Host '  (small-norm state reoriented ~100% per write), NOT high raw norm. 45.A then targets' -ForegroundColor Gray
    Write-Host '  bounding the relative move (update-if-novel / write-normalize), not clamping ||L2||.' -ForegroundColor Gray
}

# -------- Recommended 45.A thresholds (anchor: D1 @ T=0.55, nonzero only) ------
Write-Host ''
Write-Host '======== Recommended 45.A thresholds (anchor: D1 @ T=0.55, EVENT-conditioned) ========' -ForegroundColor Cyan
$anchor = $reco['D1_0.55']
if ($null -eq $anchor) { $anchor = ($reco.Values | Select-Object -First 1) }
if ($null -ne $anchor) {
    Write-Host ('  WN write-normalize target ~ p25..p50 nonzero wnorm = {0} .. {1}' -f $anchor.wnNzP25,$anchor.wnNzP50) -ForegroundColor Green
    Write-Host ('  UN update-if-novel  thr   ~ p10..p25 nonzero l2delta = {0} .. {1}' -f $anchor.dNzP10,$anchor.dNzP25) -ForegroundColor Green
    Write-Host ('  AS anti-stale stale_k     ~ gap p90 (missed-update run) = {0} writes apart' -f $anchor.gapP90) -ForegroundColor Green
    Write-Host  '  IR internal-refractory    ~ on write stress wnorm >= p90; refr ~ gap-scale' -ForegroundColor Green
    Write-Host ('  NC norm-clamp             ~ DEPRIORITIZED: raw ||L2|| p50={0} is NOT the pressure axis;' -f $anchor.l2nP50) -ForegroundColor DarkYellow
    Write-Host  '                              if used at all, clamp the RELATIVE move, not ||L2||.' -ForegroundColor DarkYellow
    Write-Host ('  effective ratio={0}  gamma={1}  (use REAL values for any gamma-based knob, not global p50)' -f $anchor.ratioMean,$anchor.gammaMean) -ForegroundColor Green
}
Write-Host ''
Write-Host '  These calibrate 45.A from the events themselves. Build 45.A only after reading them.'
Write-Host ('  Files: ' + $RDIR)
