# Phase 44.C - Entropy-high Delta Calibration Tribunal
#
# 44.B finding: delta-write is the right axis (D_delta 2.2214 BPB, -0.0379 vs
# C2.A; B_delta 2.2276, -0.0317) but generation still loops (D_delta T=0.65
# topBi 13 / altLp 3 FAIL; T=0.55 worse). Delta is right but uncalibrated.
#
# 44.C calibrates the delta on the two axes that SURVIVE per-dim z-normalization:
#   alpha (EMA pole = temporal structure) and mix (write direction).
#   src = SEE_t - mix * SEE_prev_boundary.  mix0=absolute, mix1=full delta.
#   gain is a proven no-op (scalar on a linear write, cancelled by normalization)
#   and is skipped - the math is the proof.
#
# 8 configs (entropy-high gate, clamp 2.0, no decay, no cooldown):
#   D_H0     a0.99   mix0.00   (absolute control = reproduce 44.D_hi ~2.2526)
#   D_delta  a0.99   mix1.00   (full delta control = reproduce 44.B D_delta ~2.2214)
#   D_a995   a0.995  mix1.00
#   D_a9975  a0.9975 mix1.00
#   D_a999   a0.999  mix1.00
#   D_mix25  a0.99   mix0.25
#   D_mix50  a0.99   mix0.50
#   D_mix75  a0.99   mix0.75
#
# Promotion gate (per user): val BPB <= 2.2543 (>=0.005 vs C2.A) AND topBi <= 8
#   AND altLp <= 2 AND nameWst <= 20 AND runWst <= 5 (self in [0.8,2.0]).
#
# Run:  .\benchmarks\phase38-42\phase44c_delta.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase44c'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$WP      = $WDIR + '\phase44c'
$C2A_BPB = 2.2593
if (-not (Test-Path $C2A_W)) { Write-Error "Missing C2.A: $C2A_W"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16
$RNGS    = @(12345, 67890)
$TEMPS   = @('0.65', '0.55')
$GEN_LEN = 2000
$WARMUP  = 5000
$MINLEN  = 2
$BPB_LO  = 0.8
$BPB_HI  = 2.0
$PROMO_BPB = $C2A_BPB - 0.005   # 2.2543
$G_NAME = 20.0; $G_RUN = 5.0; $G_BI = 8.0; $G_AL = 2.0

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44c_delta.c' @SRC_CORE -o "$BINDIR\phase44c_delta.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44c_delta'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Train ---------------------------------------------------------------
Write-Host ''
Write-Host '=== Phase 44.C - Training 8 delta-calibration configs ===' -ForegroundColor Yellow
Write-Host '  256D features, 8 configs -> long run. Alpha axis then mix axis.'
Write-Host ''
$TRAIN_OUT = $RDIR + '\phase44c_train.txt'
& "$BINDIR\phase44c_delta.exe" $TS_DATA $WP $C2A_W 2>&1 | Tee-Object $TRAIN_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
Write-Host 'Training complete.' -ForegroundColor Green

function Get-TrainBPB([string]$sfx) {
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $TRAIN_OUT | Select-Object -Last 1
    if ($line) { return [double](($line.Line -replace '.*BPB=','').Trim()) }
    return $null
}

$sfxs = @('_D_H0','_D_delta','_D_a995','_D_a9975','_D_a999','_D_mix25','_D_mix50','_D_mix75')
$gen_configs = @( @{ label='C2.A'; exe='phase43_generator.exe'; weights=$C2A_W; valbpb=$C2A_BPB; ref=$true } )
foreach ($s in $sfxs) {
    $gen_configs += @{ label=$s.TrimStart('_'); exe='phase44_generator.exe'; weights=$WP+$s+'.bin'; valbpb=(Get-TrainBPB ($s+'.bin')); ref=$false }
}

# -------- Word metrics --------------------------------------------------------
function Get-WordMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -eq 0) { return $null }
    $text = [System.Text.Encoding]::ASCII.GetString($bytes) -replace '[^a-zA-Z]', ' '
    $toks = @($text.ToLower() -split '\s+' | Where-Object { $_.Length -ge $MINLEN })
    $nt = $toks.Count; if ($nt -lt 4) { return $null }
    $wf = @{}; foreach ($w in $toks) { $wf[$w] = ($wf[$w] -as [int]) + 1 }
    $name_ct = 0; foreach ($w in $toks) { if ($NAME_SET.ContainsKey($w)) { $name_ct++ } }
    $nameish = [math]::Round($name_ct / $nt * 100.0, 1)
    $run = 1; $maxrun = 1
    for ($j = 1; $j -lt $nt; $j++) { if ($toks[$j] -eq $toks[$j-1]) { $run++ } else { $run = 1 }; if ($run -gt $maxrun) { $maxrun = $run } }
    $bf = @{}; for ($j = 0; $j -lt $nt-1; $j++) { $k = $toks[$j]+' '+$toks[$j+1]; $bf[$k] = ($bf[$k] -as [int]) + 1 }
    $top_bi = if ($bf.Count) { ($bf.Values | Measure-Object -Maximum).Maximum } else { 0 }
    $alt = 0; $altmax = 0
    for ($j = 0; $j -lt $nt-2; $j++) { if ($toks[$j] -eq $toks[$j+2]) { $alt++ } else { $alt = 0 }; if ($alt -gt $altmax) { $altmax = $alt } }
    return [PSCustomObject]@{ nameish=$nameish; maxrun=$maxrun; top_bi=$top_bi; altloop=$altmax }
}
function Mx ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }
function Av ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }

# -------- Generate ------------------------------------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{}; foreach ($tp in $TEMPS) { $agg[$cfg.label][$tp] = @{ mets=@(); bpbs=@() } } }

Write-Host ''
Write-Host ('=== Word-gate: temps {0}, {1}x{2}={3}/temp ===' -f ($TEMPS -join ','),$NSEEDS,$RNGS.Count,($NSEEDS*$RNGS.Count)) -ForegroundColor Yellow
Write-Host ''
foreach ($tp in $TEMPS) {
    foreach ($rng in $RNGS) {
        $si=0
        foreach ($off in $SEEDS) {
            $si++
            foreach ($cfg in $gen_configs) {
                if (-not (Test-Path $cfg.weights)) { continue }
                $lbl=$cfg.label+'_T'+$tp+'_r'+$rng+'_s'+$si
                $tf=$RDIR+'\gen_'+$lbl+'.txt'; $sf=$RDIR+'\gen_'+$lbl+'_stats.txt'
                Write-Host ('  {0,-11} T{1} r{2} s{3,-2} ...' -f $cfg.label,$tp,$rng,$si) -NoNewline
                $cmd = "`"$BINDIR\$($cfg.exe)`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --warmup $WARMUP --temp $tp --seed-start $off --rng-seed $rng > `"$tf`" 2>`"$sf`""
                cmd /c $cmd
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
    return [PSCustomObject]@{ ni=(Mx $ni); mr=(Mx $mr); bi=(Mx $bi); al=(Mx $al); self=(Av $bs) }
}

# -------- Tables --------------------------------------------------------------
foreach ($tp in $TEMPS) {
    Write-Host ''
    Write-Host ('======== T={0} : worst-case over 32 samples ========' -f $tp) -ForegroundColor Cyan
    Write-Host ('  {0,-11} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f 'config','valBPB','dBPB','nameWst','runWst','topBi','altLp','selfBPB','gate')
    Write-Host ('  ' + '-'*82)
    foreach ($cfg in $gen_configs) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { Write-Host ('  {0,-11} [no data]' -f $cfg.label); continue }
        $w = Worst $cfg.label $tp
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        $db = if ($null -ne $cfg.valbpb) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$C2A_BPB) } else { '-' }
        $gate='ref'
        if (-not $cfg.ref) {
            $pass = ($null -ne $cfg.valbpb) -and ($cfg.valbpb -le $PROMO_BPB) -and ($w.ni -le $G_NAME) -and ($w.mr -le $G_RUN) -and ($w.bi -le $G_BI) -and ($w.al -le $G_AL) -and ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
            $gate = if ($pass) {'PASS'} else {'fail'}
        }
        $col = if ($gate -eq 'PASS') {'Green'} elseif ($gate -eq 'fail') {'Gray'} else {'DarkCyan'}
        Write-Host ('  {0,-11} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f `
            $cfg.label,$vb,$db,('{0:F1}' -f $w.ni),('{0:F0}' -f $w.mr),('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F2}' -f $w.self),$gate) -ForegroundColor $col
    }
}

# -------- Control sanity (reproduce 44.A/44.B reference points) ---------------
Write-Host ''
$dh0 = Get-TrainBPB '_D_H0.bin'; $ddelta = Get-TrainBPB '_D_delta.bin'
if ($null -ne $dh0)    { Write-Host ('  Control absolute: D_H0    BPB={0:F4} (should match 44.D_hi ~2.2526)' -f $dh0) -ForegroundColor Cyan }
if ($null -ne $ddelta) { Write-Host ('  Control delta:    D_delta BPB={0:F4} (should match 44.B D_delta ~2.2214)' -f $ddelta) -ForegroundColor Cyan }

# -------- Verdict: best PASSING config ----------------------------------------
Write-Host ''
Write-Host '=== Verdetto 44.C ===' -ForegroundColor Yellow
$win=$null; $win_bpb=$null; $win_tp=''
foreach ($cfg in $gen_configs) {
    if ($cfg.ref -or ($null -eq $cfg.valbpb)) { continue }
    foreach ($tp in $TEMPS) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { continue }
        $w = Worst $cfg.label $tp
        $pass = ($cfg.valbpb -le $PROMO_BPB) -and ($w.ni -le $G_NAME) -and ($w.mr -le $G_RUN) -and ($w.bi -le $G_BI) -and ($w.al -le $G_AL) -and ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
        if ($pass -and ($null -eq $win_bpb -or $cfg.valbpb -lt $win_bpb)) { $win_bpb=$cfg.valbpb; $win=$cfg.label; $win_tp=$tp }
    }
}

# Diagnostic: did any braked/calibrated delta keep BPB strong (<=PROMO) at all?
$any_strong_bpb = $false
foreach ($cfg in $gen_configs) {
    if ($cfg.ref -or ($null -eq $cfg.valbpb)) { continue }
    if ($cfg.label -ne 'D_H0' -and $cfg.valbpb -le $PROMO_BPB) { $any_strong_bpb = $true }
}

if ($null -ne $win) {
    Write-Host ('  -> PROMOSSO: {0} @ T={1}  valBPB {2:F4} ({3:+0.000;-0.000;0.000} vs C2.A)' -f $win,$win_tp,$win_bpb,($win_bpb-$C2A_BPB)) -ForegroundColor Green
    Write-Host '     Delta calibrato: la memoria gerarchica a eventi risolve gli attractor' -ForegroundColor Green
    Write-Host '     mantenendo il guadagno BPB. SEE-V4 = C2.A + boundary-gated L2-delta memory.' -ForegroundColor Green
    Write-Host '     Primo contesto gerarchico stabile basato su DIFFERENZE, non snapshot.' -ForegroundColor Green
} elseif ($any_strong_bpb) {
    Write-Host '  -> Nessuna config passa il gate pieno, MA il BPB resta forte (delta regge).' -ForegroundColor Yellow
    Write-Host '     Diagnosi: il segnale c''e (delta comprime), ma il readout lineare su L2 lo' -ForegroundColor Yellow
    Write-Host '     usa in modo instabile in closed-loop. Serve REGOLARIZZAZIONE del readout L2,' -ForegroundColor Yellow
    Write-Host '     non nuova memoria. Prossimo: weight-decay/dropout mirato sulle 64 col L2,' -ForegroundColor Yellow
    Write-Host '     o readout L2 a rango ridotto. NON aggiungere altra memoria.' -ForegroundColor Yellow
    Write-Host '     Leggi gli assi: alpha alto (memoria piu lenta) o mix basso (meno delta)' -ForegroundColor Yellow
    Write-Host '     abbassano topBi/altLp? Quello indica la direzione della calibrazione.' -ForegroundColor Yellow
} else {
    Write-Host '  -> Il BPB crolla appena si frena/calibra il delta.' -ForegroundColor Yellow
    Write-Host '     Diagnosi: L2 porta segnale solo nella forma piu instabile (full delta a0.99).' -ForegroundColor Yellow
    Write-Host '     Calibrarlo lo spegne -> il guadagno -0.0379 era inseparabile dalla' -ForegroundColor Yellow
    Write-Host '     instabilita. La proiezione random fissa e il tetto: prossimo step =' -ForegroundColor Yellow
    Write-Host '     L2 trainable / structured projection (non piu fixed random).' -ForegroundColor Yellow
}
Write-Host ''
Write-Host '  Messaggio architetturale: la memoria gerarchica funziona ma deve memorizzare'
Write-Host '  DIFFERENZE/EVENTI, non snapshot. 44.C dice se quel segnale e calibrabile a'
Write-Host '  stabilita, o se istante per istante e inseparabile dal rumore.'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
