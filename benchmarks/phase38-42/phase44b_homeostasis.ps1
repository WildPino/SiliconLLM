# Phase 44.B - L2 Homeostasis Tribunal
#
# 44.A proved boundary-gated slow memory helps BPB (ws -0.0094, entropy -0.0067)
# but generation still loops (topBi ~12, altLp ~3 at T=0.65; worse at 0.55 = not
# thermal). L2 accumulates useful context AND attractors. 44.B adds BRAKES on L2
# (not new gates) to the two survivors: 44.D_hi (entropy, alpha0.99) and
# 44.B_a95 (whitespace, alpha0.95). Priority candidate: 44.D_hi + clamp/decay.
#
# Brakes: L2 norm clamp 1.5 | non-boundary decay 0.995 | cooldown 16 |
#         delta-write | stack(all). Magic 0x5345453D.
#
# Promotion gate (per user): val BPB <= 2.2543 (>=0.005 vs C2.A) AND topBi <= 8
#   AND altLp <= 2 AND nameWst <= 20 AND runWst <= 5 (self in [0.8,2.0]).
#
# Run:  .\benchmarks\phase38-42\phase44b_homeostasis.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase44b'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$WP      = $WDIR + '\phase44b'
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
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44b_homeostasis.c' @SRC_CORE -o "$BINDIR\phase44b_homeostasis.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44b_homeostasis'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Train ---------------------------------------------------------------
Write-Host ''
Write-Host '=== Phase 44.B - Training 12 homeostasis configs ===' -ForegroundColor Yellow
Write-Host '  WARNING: 256D features, 12 configs -> long run. D-family runs first.'
Write-Host ''
$TRAIN_OUT = $RDIR + '\phase44b_train.txt'
& "$BINDIR\phase44b_homeostasis.exe" $TS_DATA $WP $C2A_W 2>&1 | Tee-Object $TRAIN_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
Write-Host 'Training complete.' -ForegroundColor Green

function Get-TrainBPB([string]$sfx) {
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $TRAIN_OUT | Select-Object -Last 1
    if ($line) { return [double](($line.Line -replace '.*BPB=','').Trim()) }
    return $null
}

$sfxs = @('_D_H0','_D_clamp15','_D_decay995','_D_cool16','_D_delta','_D_stack',
          '_B_H0','_B_clamp15','_B_decay995','_B_cool16','_B_delta','_B_stack')
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

# -------- Control sanity (D_H0 / B_H0 should match 44.A) ----------------------
Write-Host ''
$dh0 = Get-TrainBPB '_D_H0.bin'; $bh0 = Get-TrainBPB '_B_H0.bin'
if ($null -ne $dh0) { Write-Host ('  44.A reproduction: D_H0 BPB={0:F4} (should match 44.D ~2.2526)' -f $dh0) -ForegroundColor Cyan }
if ($null -ne $bh0) { Write-Host ('                     B_H0 BPB={0:F4} (should match 44.B ~2.2499)' -f $bh0) -ForegroundColor Cyan }

# -------- Verdict: best PASSING config ----------------------------------------
Write-Host ''
Write-Host '=== Verdetto 44.B ===' -ForegroundColor Yellow
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
if ($null -ne $win) {
    Write-Host ('  -> PROMOSSO: {0} @ T={1}  valBPB {2:F4} ({3:+0.000;-0.000;0.000} vs C2.A)' -f $win,$win_tp,$win_bpb,($win_bpb-$C2A_BPB)) -ForegroundColor Green
    Write-Host '     Omeostasi su L2 risolve gli attractor mantenendo il guadagno BPB.' -ForegroundColor Green
    Write-Host '     SEE-V4 = C2.A + boundary-gated L2 + homeostasis. Primo contesto gerarchico stabile.' -ForegroundColor Green
} else {
    Write-Host '  -> Nessuna config passa il gate pieno.' -ForegroundColor Yellow
    Write-Host '     Confronta i brake: quale riduce di piu topBi/altLp senza perdere BPB?' -ForegroundColor Yellow
    Write-Host '     - se clamp/decay aiutano ma non bastano: L2 fixed-projection e il tetto -> L2 trainable / structured projection.' -ForegroundColor Yellow
    Write-Host '     - se delta-write abbatte i loop: il problema era contesto assoluto accumulato -> espandi delta-write.' -ForegroundColor Yellow
    Write-Host '     - se BPB crolla coi brake: i brake cancellano il segnale -> brake piu morbidi (decay 0.999, clamp 2.0).' -ForegroundColor Yellow
}
Write-Host ''
Write-Host '  Messaggio: il silicio ha accettato memoria gerarchica; ora chiede'
Write-Host '  omeostasi su L2 - quanto lasciare accumulare, non solo quando aggiornare.'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
