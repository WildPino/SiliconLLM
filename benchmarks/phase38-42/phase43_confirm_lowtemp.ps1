# Phase 43 - C2.A Low-Temperature Confirmation (final microphase)
#
# At T=0.65, C2.A passes BPB + name-ish + word-run + self, but MISSES the
# worst-case loop gate: topBi=11 (>8) and altLp=3 (>2). It is the SEE-V3
# CANDIDATE / best-so-far, NOT a full promotion at T=0.65.
#
# Architecturally it crushes the baselines by an order of magnitude:
#   c20 ALL:    topBi 153, altLp 18
#   SEE-V2 ALL: topBi 111, altLp 13
#   C2.A ALL:   topBi 11,  altLp 3
# i.e. 26-cell microplasticity works; the byte-level loop gate is just tighter
# than T=0.65 decoding can satisfy. This microphase asks ONE question: does a
# slightly cooler decode (T<=0.60) bring C2.A inside the full gate WITHOUT
# changing BPB (BPB is teacher-forced, temp-independent -> always 2.2593).
#
# Configs: c20 vs SEE-V2 vs C2.A. Temps: 0.60, 0.55. 16 seeds x 2 rng = 32/temp.
#
# Decision:
#   C2.A passes full gate at T=0.60 -> promote SEE-V3 = C2.A, stable temp T<=0.60
#   passes only at T=0.55           -> SEE-V3 = C2.A, stable temp T<=0.55
#   fails both                      -> close Phase 43:
#       official compression baseline = C2.A (best BPB+attractor reduction)
#       official stable-generation baseline = none / SEE-V2 per gate
#       Phase 44 = boundary-gated hierarchical memory (structural jump)
#
# NO more routing/Oja tweaks. This is the last local experiment of Phase 43.
#
# Run:  .\benchmarks\phase38-42\phase43_confirm_lowtemp.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase43_lowtemp'
$TS_DATA = $ROOT + '\experiments\phase41a\corpora\tinystories_64mb.txt'
$C2LOG   = $ROOT + '\results\phase43c2\phase43c2_train.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16
$RNGS    = @(12345, 67890)
$TEMPS   = @('0.60', '0.55')
$GEN_LEN = 2000
$WARMUP  = 5000
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

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING generator ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed'; exit 1 }
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

# -------- Generate: temp x rng x seed x config --------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

# agg[label][temp] = @{mets=@(); bpbs=@()}  (32 samples folded over rng)
$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{}; foreach ($tp in $TEMPS) { $agg[$cfg.label][$tp] = @{ mets=@(); bpbs=@() } } }

Write-Host ''
Write-Host ('=== Low-temp confirmation: temps {0}, {1} seeds x {2} rng = {3}/temp ===' -f ($TEMPS -join ','),$NSEEDS,$RNGS.Count,($NSEEDS*$RNGS.Count)) -ForegroundColor Yellow
Write-Host ''
foreach ($tp in $TEMPS) {
    foreach ($rng in $RNGS) {
        $si=0
        foreach ($off in $SEEDS) {
            $si++
            foreach ($cfg in $gen_configs) {
                $lbl = $cfg.label + '_T' + $tp + '_r' + $rng + '_s' + $si
                $tf  = $RDIR + '\gen_' + $lbl + '.txt'
                $sf  = $RDIR + '\gen_' + $lbl + '_stats.txt'
                Write-Host ('  {0,-8} T{1} rng{2} s{3,-2} ...' -f $cfg.label,$tp,$rng,$si) -NoNewline
                $cmdLine = "`"$BINDIR\phase43_generator.exe`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --warmup $WARMUP --temp $tp --seed-start $off --rng-seed $rng > `"$tf`" 2>`"$sf`""
                cmd /c $cmdLine
                if ($LASTEXITCODE -ne 0) { Write-Host ' FAILED' -ForegroundColor Red; continue }
                Write-Host ' OK' -ForegroundColor Green
                $m = Get-WordMetrics $tf
                if ($m) { $agg[$cfg.label][$tp].mets += $m }
                $bl = Select-String 'self_BPB' $sf | Select-Object -Last 1
                if ($bl) { $agg[$cfg.label][$tp].bpbs += [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
            }
        }
    }
}

function Worst($lbl,$tp) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $bs=@()
    foreach ($m in @($agg[$lbl][$tp].mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop }
    foreach ($b in @($agg[$lbl][$tp].bpbs)) { $bs+=[double]$b }
    return [PSCustomObject]@{ ni_wst=(Mx $ni); mr_wst=(Mx $mr); bi_wst=(Mx $bi); al_wst=(Mx $al); self=(Avg $bs) }
}
function Gate-Pass($valbpb,$w) {
    return ($null -ne $valbpb) -and ($valbpb -le ($SEE_V2_BPB-$GATE_BPB_IMPROVE)) -and `
           ($w.ni_wst -le $GATE_NAMEISH_WST) -and ($w.mr_wst -le $GATE_WORDRUN_WST) -and `
           ($w.bi_wst -le $GATE_TOPBI) -and ($w.al_wst -le $GATE_ALTLOOP) -and `
           ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
}

# -------- Table ---------------------------------------------------------------
Write-Host ''
Write-Host '======================================================================' -ForegroundColor Cyan
Write-Host '  Phase 43 C2.A Low-Temp Confirmation - worst-case over 32 samples'
Write-Host '======================================================================'
Write-Host ('  gate: valBPB<={0}  nameWst<={1}  runWst<={2}  topBi<={3}  altLp<={4}  self in [{5},{6}]' -f `
    ($SEE_V2_BPB-$GATE_BPB_IMPROVE),$GATE_NAMEISH_WST,$GATE_WORDRUN_WST,$GATE_TOPBI,$GATE_ALTLOOP,$BPB_LO,$BPB_HI)
Write-Host ''
Write-Host ('  {0,-8} {1,5} {2,9} {3,8} {4,7} {5,7} {6,7} {7,9} {8,6}' -f `
    'config','temp','valBPB','nameWst','runWst','topBi','altLp','selfBPB','gate')
Write-Host ('  ' + '-'*80)
foreach ($tp in $TEMPS) {
    foreach ($cfg in $gen_configs) {
        $w = Worst $cfg.label $tp
        $gate='ref'
        if (-not $cfg.isbase) { $gate = if (Gate-Pass $cfg.valbpb $w) {'PASS'} else {'fail'} }
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        $col = if ($gate -eq 'PASS') {'Green'} elseif ($gate -eq 'fail') {'DarkGray'} else {'Gray'}
        Write-Host ('  {0,-8} {1,5} {2,9} {3,8} {4,7} {5,7} {6,7} {7,9} {8,6}' -f `
            $cfg.label,$tp,$vb,('{0:F1}' -f $w.ni_wst),('{0:F0}' -f $w.mr_wst),('{0:F0}' -f $w.bi_wst),('{0:F0}' -f $w.al_wst),('{0:F2}' -f $w.self),$gate) -ForegroundColor $col
    }
    Write-Host ('  ' + '-'*80)
}

# -------- Verdict -------------------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto low-temp: C2.A ===' -ForegroundColor Yellow
$promote_temp=$null
foreach ($tp in $TEMPS) {   # TEMPS ordered high->low; first pass = warmest passing temp
    if (Gate-Pass $c2a_bpb (Worst 'C2.A' $tp)) { $promote_temp=$tp; break }
}
if ($null -ne $promote_temp) {
    Write-Host ('  -> C2.A PASSA il gate completo a T={0} (BPB invariato {1:F4}).' -f $promote_temp,$c2a_bpb) -ForegroundColor Green
    Write-Host ('     PROMUOVI SEE-V3 = C2.A (SEE-V1S c20 + 26c Oja eta=1e-3),' ) -ForegroundColor Green
    Write-Host ('     stable operating temp T<={0}. weights/phase43c2_C2A.bin' -f $promote_temp) -ForegroundColor Green
} else {
    Write-Host '  -> C2.A non passa il gate completo nemmeno a T=0.60/0.55.' -ForegroundColor Red
    Write-Host '     CHIUDI Phase 43:' -ForegroundColor Red
    Write-Host '       compression baseline (BPB)     = C2.A 2.2593 (miglior BPB+attractor)' -ForegroundColor Red
    Write-Host '       stable-generation baseline      = nessuno pieno / SEE-V2 borderline' -ForegroundColor Red
    Write-Host '     Phase 44 = boundary-gated hierarchical memory (salto strutturale).' -ForegroundColor Red
    Write-Host '     NON altri routing/Oja tweak: il limite e closed-loop, non microplasticita.' -ForegroundColor Red
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
