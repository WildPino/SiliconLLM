# Phase 43 - C2.A Confirmation Run (robust word-gate before SEE-V3 promotion)
#
# The deterministic re-check named C2.A (26c Oja eta1e-3) the winner: first
# 26-cell config to pass val BPB + full word-gate. Before promoting SEE-V3 we
# confirm it is not a single-RNG fluke: 16 seeds x 2 rng-seeds = 32 samples per
# config, worst-case gate, with per-rng breakdown to expose rng-sensitivity.
#
# Configs: C2.A (candidate) vs c20 (SEE-V1S) vs SEE-V2 (13c) - generation only.
#
# Gate (applied to C2.A, over ALL 32 samples - worst case):
#   val BPB <= 2.2597   name-ish worst <= 20   word-run worst <= 5
#   top bigram <= 8   alt loop <= 2   0.8 <= self_BPB <= 2.0
#
# Decision: C2.A passes on BOTH rng-seeds (and combined) -> promote
#   SEE-V3 = SEE-V1S c20 + 26c Oja eta=1e-3  (weights/phase43c2_C2A.bin).
#   Else -> close Phase 43, dual baseline, Phase 44 structural jump.
#
# Run:  .\benchmarks\phase38-42\phase43_confirm_c2a.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase43_confirm_c2a'
$TS_DATA = $ROOT + '\experiments\phase41a\corpora\tinystories_64mb.txt'
$C2LOG   = $ROOT + '\results\phase43c2\phase43c2_train.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16
$RNGS    = @(12345, 67890)   # two independent sampling streams
$GEN_LEN = 2000
$WARMUP  = 5000
$T       = '0.65'
$MINLEN  = 2
$BPB_LO  = 0.8
$BPB_HI  = 2.0
$SEE_V2_BPB = 2.2617

$GATE_NAMEISH_WST = 20.0
$GATE_WORDRUN_WST = 5.0
$GATE_TOPBI       = 8.0
$GATE_ALTLOOP     = 2.0
$GATE_BPB_IMPROVE = 0.002

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile generator ---------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING generator ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

function Get-LogBPB([string]$log, [string]$sfx) {
    if (-not (Test-Path $log)) { return $null }
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $log | Select-Object -Last 1
    if ($line) { return [double](($line.Line -replace '.*BPB=','').Trim()) }
    return $null
}
$c2a_bpb = Get-LogBPB $C2LOG '_C2A.bin'

$gen_configs = @(
    @{ label='c20';    weights=$WDIR+'\phase43h_c20.bin';    isbase=$true;  valbpb=2.2656 },
    @{ label='SEE-V2'; weights=$WDIR+'\phase43c_eta1e3.bin'; isbase=$true;  valbpb=$SEE_V2_BPB },
    @{ label='C2.A';   weights=$WDIR+'\phase43c2_C2A.bin';   isbase=$false; valbpb=$c2a_bpb }
)
foreach ($c in $gen_configs) { if (-not (Test-Path $c.weights)) { Write-Error "Missing weights: $($c.weights)"; exit 1 } }

# -------- Word-level metrics --------------------------------------------------
function Get-WordMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -eq 0) { return $null }
    $text = [System.Text.Encoding]::ASCII.GetString($bytes) -replace '[^a-zA-Z]', ' '
    $toks = @($text.ToLower() -split '\s+' | Where-Object { $_.Length -ge $MINLEN })
    $nt = $toks.Count
    if ($nt -lt 4) { return $null }
    $wf = @{}
    foreach ($w in $toks) { $wf[$w] = ($wf[$w] -as [int]) + 1 }
    $name_ct = 0
    foreach ($w in $toks) { if ($NAME_SET.ContainsKey($w)) { $name_ct++ } }
    $nameish = [math]::Round($name_ct / $nt * 100.0, 1)
    $run = 1; $maxrun = 1
    for ($j = 1; $j -lt $nt; $j++) { if ($toks[$j] -eq $toks[$j-1]) { $run++ } else { $run = 1 }; if ($run -gt $maxrun) { $maxrun = $run } }
    $bf = @{}
    for ($j = 0; $j -lt $nt-1; $j++) { $k = $toks[$j]+' '+$toks[$j+1]; $bf[$k] = ($bf[$k] -as [int]) + 1 }
    $top_bi = if ($bf.Count) { ($bf.Values | Measure-Object -Maximum).Maximum } else { 0 }
    $alt = 0; $altmax = 0
    for ($j = 0; $j -lt $nt-2; $j++) { if ($toks[$j] -eq $toks[$j+2]) { $alt++ } else { $alt = 0 }; if ($alt -gt $altmax) { $altmax = $alt } }
    return [PSCustomObject]@{ nameish=$nameish; maxrun=$maxrun; top_bi=$top_bi; altloop=$altmax }
}

function Avg([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }
function Mx ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }

# -------- Generate across seeds x rng -----------------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

# agg[label][rng] = @{mets=@(); bpbs=@()}
$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{}; foreach ($rng in $RNGS) { $agg[$cfg.label][$rng] = @{ mets=@(); bpbs=@() } } }

Write-Host ''
Write-Host ('=== Confirmation: {0} seeds x {1} rng = {2} samples/config, T={3} ===' -f $NSEEDS,$RNGS.Count,($NSEEDS*$RNGS.Count),$T) -ForegroundColor Yellow
Write-Host ''
foreach ($rng in $RNGS) {
    $si = 0
    foreach ($off in $SEEDS) {
        $si++
        foreach ($cfg in $gen_configs) {
            $lbl = $cfg.label + '_r' + $rng + '_s' + $si
            $tf  = $RDIR + '\gen_' + $lbl + '.txt'
            $sf  = $RDIR + '\gen_' + $lbl + '_stats.txt'
            Write-Host ('  {0,-8} rng{1} seed{2,-2} ...' -f $cfg.label,$rng,$si) -NoNewline
            $cmdLine = "`"$BINDIR\phase43_generator.exe`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --warmup $WARMUP --temp $T --seed-start $off --rng-seed $rng > `"$tf`" 2>`"$sf`""
            cmd /c $cmdLine
            if ($LASTEXITCODE -ne 0) { Write-Host ' FAILED' -ForegroundColor Red; continue }
            Write-Host ' OK' -ForegroundColor Green
            $m = Get-WordMetrics $tf
            if ($m) { $agg[$cfg.label][$rng].mets += $m }
            $bl = Select-String 'self_BPB' $sf | Select-Object -Last 1
            if ($bl) { $agg[$cfg.label][$rng].bpbs += [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
        }
    }
}

# -------- Per-rng + combined worst-case table ---------------------------------
Write-Host ''
Write-Host '======================================================================' -ForegroundColor Cyan
Write-Host '  Phase 43 C2.A Confirmation - worst-case word-gate'
Write-Host '======================================================================'
Write-Host ('  gate: valBPB<={0}  nameWst<={1}  runWst<={2}  topBi<={3}  altLp<={4}  self in [{5},{6}]' -f `
    ($SEE_V2_BPB-$GATE_BPB_IMPROVE),$GATE_NAMEISH_WST,$GATE_WORDRUN_WST,$GATE_TOPBI,$GATE_ALTLOOP,$BPB_LO,$BPB_HI)
Write-Host ''
Write-Host ('  {0,-8} {1,5} {2,9} {3,8} {4,7} {5,7} {6,7} {7,9} {8,6}' -f `
    'config','rng','valBPB','nameWst','runWst','topBi','altLp','selfBPB','gate')
Write-Host ('  ' + '-'*78)

# returns combined worst metrics row for a config across all rng
function Worst-Row($lbl) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $bs=@()
    foreach ($rng in $RNGS) {
        foreach ($m in @($agg[$lbl][$rng].mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop }
        foreach ($b in @($agg[$lbl][$rng].bpbs)) { $bs+=[double]$b }
    }
    return [PSCustomObject]@{ ni_wst=(Mx $ni); mr_wst=(Mx $mr); bi_wst=(Mx $bi); al_wst=(Mx $al); self=(Avg $bs) }
}

function Gate-Pass($valbpb,$w) {
    return ($null -ne $valbpb) -and ($valbpb -le ($SEE_V2_BPB-$GATE_BPB_IMPROVE)) -and `
           ($w.ni_wst -le $GATE_NAMEISH_WST) -and ($w.mr_wst -le $GATE_WORDRUN_WST) -and `
           ($w.bi_wst -le $GATE_TOPBI) -and ($w.al_wst -le $GATE_ALTLOOP) -and `
           ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
}

$c2a_per_rng_pass = @{}
foreach ($cfg in $gen_configs) {
    # per-rng rows
    foreach ($rng in $RNGS) {
        $ni=@(); $mr=@(); $bi=@(); $al=@(); $bs=@()
        foreach ($m in @($agg[$cfg.label][$rng].mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop }
        foreach ($b in @($agg[$cfg.label][$rng].bpbs)) { $bs+=[double]$b }
        $w = [PSCustomObject]@{ ni_wst=(Mx $ni); mr_wst=(Mx $mr); bi_wst=(Mx $bi); al_wst=(Mx $al); self=(Avg $bs) }
        $gate = 'ref'
        if (-not $cfg.isbase) {
            $p = Gate-Pass $cfg.valbpb $w
            $gate = if ($p) {'PASS'} else {'fail'}
            $c2a_per_rng_pass[$rng] = $p
        }
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        Write-Host ('  {0,-8} {1,5} {2,9} {3,7} {4,7} {5,7} {6,7} {7,9} {8,6}' -f `
            $cfg.label,$rng,$vb,('{0:F1}' -f $w.ni_wst),('{0:F0}' -f $w.mr_wst),('{0:F0}' -f $w.bi_wst),('{0:F0}' -f $w.al_wst),('{0:F2}' -f $w.self),$gate)
    }
    # combined row
    $wc = Worst-Row $cfg.label
    $gate = 'ref'
    if (-not $cfg.isbase) { $gate = if (Gate-Pass $cfg.valbpb $wc) {'PASS'} else {'fail'} }
    $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
    Write-Host ('  {0,-8} {1,5} {2,9} {3,7} {4,7} {5,7} {6,7} {7,9} {8,6}' -f `
        $cfg.label,'ALL',$vb,('{0:F1}' -f $wc.ni_wst),('{0:F0}' -f $wc.mr_wst),('{0:F0}' -f $wc.bi_wst),('{0:F0}' -f $wc.al_wst),('{0:F2}' -f $wc.self),$gate)
    Write-Host ('  ' + '-'*78)
}

# -------- Verdict -------------------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto confirmation: C2.A ===' -ForegroundColor Yellow
$wc = Worst-Row 'C2.A'
$combined_pass = Gate-Pass $c2a_bpb $wc
$both_rng_pass = $true
foreach ($rng in $RNGS) { if (-not $c2a_per_rng_pass[$rng]) { $both_rng_pass = $false } }

if ($combined_pass -and $both_rng_pass) {
    Write-Host '  -> C2.A CONFERMATO: passa il gate su entrambi gli rng e combinato.' -ForegroundColor Green
    Write-Host '     PROMUOVI SEE-V3 = SEE-V1S c20 + 26c Oja eta=1e-3' -ForegroundColor Green
    Write-Host '     weights/phase43c2_C2A.bin' -ForegroundColor Green
} elseif ($combined_pass -and -not $both_rng_pass) {
    Write-Host '  -> C2.A passa il combinato ma NON entrambi gli rng singolarmente.' -ForegroundColor Yellow
    Write-Host '     Borderline / rng-sensitive. Allarga (4 rng o 32 seed) prima di SEE-V3.' -ForegroundColor Yellow
} else {
    Write-Host '  -> C2.A FALLISCE il confirmation robusto.' -ForegroundColor Red
    Write-Host '     CHIUDI Phase 43. Baseline duale SEE-V1S c20 / SEE-V2 13c.' -ForegroundColor Red
    Write-Host '     Phase 44 = salto strutturale: memoria gerarchica / boundary-gated.' -ForegroundColor Red
}

Write-Host ''
Write-Host ('  Files: ' + $RDIR)
