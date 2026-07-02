# Phase 43.C3 - Plastic Homeostasis Tribunal
#
# 43.C2 verdict: 26 raw plastic cells find signal (C2.B BPB 2.2572) but fall into
# tunnels (name worst 50.5%, word-run 30, topBi 74, altLoop 28). C2.A was NEAR.
# Silicon said: yes to more memory, but only if plasticity stays constrained.
#
# 43.C3 brakes the Oja projection toward the identity base:
#   h = h_base + beta*(h_oja - h_base);  beta = plastic_blend in [0,1]
# More capacity (26 cells) WITH a homeostatic brake.
#
# base:    SEE-V2 = 13c eta1e-3 (weights/phase43c_eta1e3.bin)
# variants (self-describing magic 0x5345453A, n_oja + blend in header):
#   C3.A  26c eta1e-3 blend0.25
#   C3.B  26c eta1e-3 blend0.50
#   C3.C  26c eta3e-4 blend0.25
#   C3.D  26c eta3e-4 blend0.50
#
# Promotion gate (per variant) - IDENTICAL to 43.C2:
#   val BPB <= 2.2597 (>=0.002 better than SEE-V2)
#   name-ish worst <= 20%   word-run worst <= 5   top bigram <= 8   alt loop <= 2
#   0.8 <= self_BPB <= 2.0
#
# Run:  .\benchmarks\phase38-42\phase43c3_homeostasis.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase43c3'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$BASE_W = $WDIR + '\phase43c_eta1e3.bin'   # SEE-V2
$C20_W  = $WDIR + '\phase43h_c20.bin'      # warm-start readout source
$WP     = $WDIR + '\phase43c3'             # output prefix -> phase43c3_C3A.bin etc.
$SEE_V2_BPB = 2.2617

foreach ($w in @($BASE_W, $C20_W)) {
    if (-not (Test-Path $w)) { Write-Error "Missing weights: $w"; exit 1 }
}

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16
$GEN_LEN = 2000
$WARMUP  = 5000
$T       = '0.65'
$MINLEN  = 2
$BPB_LO  = 0.8
$BPB_HI  = 2.0

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
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
Write-Host '  phase43c3_homeostasis.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43c3_homeostasis.c' @SRC_CORE -o "$BINDIR\phase43c3_homeostasis.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43c3_homeostasis'; exit 1 }
Write-Host '  phase43_generator.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Train ---------------------------------------------------------------
Write-Host ''
Write-Host '=== Phase 43.C3 - Training 4 variants (C3.A/B/C/D) ===' -ForegroundColor Yellow
Write-Host '  Warm start readout from phase43h_c20.bin; clamp +/-2.0; magic 0x5345453A'
Write-Host ''
$TRAIN_OUT = $RDIR + '\phase43c3_train.txt'
& "$BINDIR\phase43c3_homeostasis.exe" $TS_DATA $WP $C20_W 2>&1 | Tee-Object $TRAIN_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
Write-Host ''
Write-Host 'Training complete.' -ForegroundColor Green

function Get-TrainBPB([string]$sfx) {
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $TRAIN_OUT | Select-Object -Last 1
    if ($line) { return [double](($line.Line -replace '.*BPB=','').Trim()) }
    return $null
}

# -------- Generation configs --------------------------------------------------
$gen_configs = @(
    @{ label='SEE-V2_13'; weights=$BASE_W;        isbase=$true;  valbpb=$SEE_V2_BPB },
    @{ label='C3.A_b25';  weights=$WP+'_C3A.bin'; isbase=$false; valbpb=(Get-TrainBPB '_C3A.bin') },
    @{ label='C3.B_b50';  weights=$WP+'_C3B.bin'; isbase=$false; valbpb=(Get-TrainBPB '_C3B.bin') },
    @{ label='C3.C_b25';  weights=$WP+'_C3C.bin'; isbase=$false; valbpb=(Get-TrainBPB '_C3C.bin') },
    @{ label='C3.D_b50';  weights=$WP+'_C3D.bin'; isbase=$false; valbpb=(Get-TrainBPB '_C3D.bin') }
)

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
    $top_word = [math]::Round(($wf.Values | Measure-Object -Maximum).Maximum / $nt * 100.0, 1)

    $name_ct = 0
    foreach ($w in $toks) { if ($NAME_SET.ContainsKey($w)) { $name_ct++ } }
    $nameish = [math]::Round($name_ct / $nt * 100.0, 1)

    $run = 1; $maxrun = 1
    for ($j = 1; $j -lt $nt; $j++) {
        if ($toks[$j] -eq $toks[$j-1]) { $run++ } else { $run = 1 }
        if ($run -gt $maxrun) { $maxrun = $run }
    }
    $bf = @{}
    for ($j = 0; $j -lt $nt-1; $j++) { $k = $toks[$j]+' '+$toks[$j+1]; $bf[$k] = ($bf[$k] -as [int]) + 1 }
    $top_bi = if ($bf.Count) { ($bf.Values | Measure-Object -Maximum).Maximum } else { 0 }
    $alt = 0; $altmax = 0
    for ($j = 0; $j -lt $nt-2; $j++) {
        if ($toks[$j] -eq $toks[$j+2]) { $alt++ } else { $alt = 0 }
        if ($alt -gt $altmax) { $altmax = $alt }
    }
    $top_words = $wf.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 6 |
        ForEach-Object { "$($_.Key)=$($_.Value)" }
    return [PSCustomObject]@{
        top_word=$top_word; nameish=$nameish; maxrun=$maxrun; top_bi=$top_bi; altloop=$altmax
        top_words=($top_words -join ' ')
    }
}

# -------- Generate + score ----------------------------------------------------
Write-Host ''
Write-Host ('=== Generation word-gate: {0} seeds, T={1} ===' -f $NSEEDS, $T) -ForegroundColor Yellow
Write-Host ''

$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{ mets=@(); bpbs=@() } }

$si = 0
foreach ($off in $SEEDS) {
    $si++
    foreach ($cfg in $gen_configs) {
        if (-not (Test-Path $cfg.weights)) { Write-Host ('  {0} seed{1}: weights missing, skip' -f $cfg.label,$si) -ForegroundColor Red; continue }
        $lbl = $cfg.label + '_s' + $si
        $tf  = $RDIR + '\gen_' + $lbl + '.txt'
        $sf  = $RDIR + '\gen_' + $lbl + '_stats.txt'
        Write-Host ('  {0,-10} seed{1,-2} (off={2}) ...' -f $cfg.label, $si, $off) -NoNewline
        $cmdLine = "`"$BINDIR\phase43_generator.exe`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --warmup $WARMUP --temp $T --seed-start $off > `"$tf`" 2>`"$sf`""
        cmd /c $cmdLine
        if ($LASTEXITCODE -ne 0) { Write-Host ' FAILED' -ForegroundColor Red; continue }
        Write-Host ' OK' -ForegroundColor Green
        $m = Get-WordMetrics $tf
        if ($m) { $agg[$cfg.label].mets += $m }
        $bl = Select-String 'self_BPB' $sf | Select-Object -Last 1
        if ($bl) { $agg[$cfg.label].bpbs += [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
    }
}

function Avg([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }
function Mx ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }

# -------- Summary table -------------------------------------------------------
Write-Host ''
Write-Host '======================================================================' -ForegroundColor Cyan
Write-Host ('  Phase 43.C3 - Plastic Homeostasis ({0} seeds, T={1})' -f $NSEEDS, $T)
Write-Host '======================================================================'
Write-Host ('  gate: valBPB<=SEE-V2-{0}  nameWst<={1}  runWst<={2}  topBi<={3}  altLp<={4}  self in [{5},{6}]' -f `
    $GATE_BPB_IMPROVE,$GATE_NAMEISH_WST,$GATE_WORDRUN_WST,$GATE_TOPBI,$GATE_ALTLOOP,$BPB_LO,$BPB_HI)
Write-Host ''
Write-Host ('  {0,-10} {1,9} {2,8} {3,14} {4,8} {5,7} {6,7} {7,9} {8,6}' -f `
    'config','valBPB','dBPB','nameish(av/wst)','runWst','topBi','altLp','selfBPB','gate')
Write-Host ('  ' + '-'*88)

$rows = @{}
foreach ($cfg in $gen_configs) {
    $ms = @($agg[$cfg.label].mets); $bs = @($agg[$cfg.label].bpbs)
    if ($ms.Count -eq 0) { Write-Host ('  {0,-10} [no data]' -f $cfg.label); continue }

    $ni = $ms | ForEach-Object { [double]$_.nameish }
    $mr = $ms | ForEach-Object { [double]$_.maxrun }
    $bi = $ms | ForEach-Object { [double]$_.top_bi }
    $al = $ms | ForEach-Object { [double]$_.altloop }

    $vb = $cfg.valbpb
    $row = [PSCustomObject]@{
        valbpb=$vb; ni_avg=(Avg $ni); ni_wst=(Mx $ni); mr_wst=(Mx $mr)
        bi_wst=(Mx $bi); al_wst=(Mx $al); selfbpb=(Avg $bs); isbase=$cfg.isbase
    }
    $rows[$cfg.label] = $row

    $dbpb = if ($vb -ne $null) { $vb - $SEE_V2_BPB } else { $null }

    $gate = '-'
    if (-not $cfg.isbase) {
        $bpb_ok = ($vb -ne $null) -and ($vb -le ($SEE_V2_BPB - $GATE_BPB_IMPROVE))
        $ni_ok  = $row.ni_wst -le $GATE_NAMEISH_WST
        $mr_ok  = $row.mr_wst -le $GATE_WORDRUN_WST
        $bi_ok  = $row.bi_wst -le $GATE_TOPBI
        $al_ok  = $row.al_wst -le $GATE_ALTLOOP
        $sb_ok  = ($row.selfbpb -ge $BPB_LO) -and ($row.selfbpb -le $BPB_HI)
        $gate = if ($bpb_ok -and $ni_ok -and $mr_ok -and $bi_ok -and $al_ok -and $sb_ok) { 'PASS' } else { 'fail' }
    }

    $vb_str  = if ($vb -ne $null) { '{0:F4}' -f $vb } else { 'N/A' }
    $db_str  = if ($dbpb -ne $null) { '{0:+0.000;-0.000;0.000}' -f $dbpb } else { '-' }

    Write-Host ('  {0,-10} {1,9} {2,8} {3,14} {4,8} {5,7} {6,7} {7,9} {8,6}' -f `
        $cfg.label, $vb_str, $db_str,
        ('{0:F1}/{1:F1}' -f $row.ni_avg, $row.ni_wst),
        ('{0:F0}' -f $row.mr_wst),
        ('{0:F0}' -f $row.bi_wst),
        ('{0:F0}' -f $row.al_wst),
        ('{0:F2}' -f $row.selfbpb),
        $gate)
}

# -------- Top words per seed (eyeball) ----------------------------------------
Write-Host ''
Write-Host '=== Top words - seeds 1..4 (spot the name tunnel) ===' -ForegroundColor Cyan
foreach ($cfg in $gen_configs) {
    Write-Host ('  -- {0}' -f $cfg.label) -ForegroundColor Magenta
    $s = 0
    foreach ($m in @($agg[$cfg.label].mets)) {
        $s++; if ($s -gt 4) { break }
        Write-Host ('     s{0,-2} {1}' -f $s, $m.top_words)
    }
}

# -------- Interpretation ------------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 43.C3 ===' -ForegroundColor Yellow

function Is-Pass($lbl) {
    if (-not $rows.ContainsKey($lbl)) { return $false }
    $r = $rows[$lbl]
    if ($r.valbpb -eq $null) { return $false }
    return ($r.valbpb -le ($SEE_V2_BPB - $GATE_BPB_IMPROVE)) -and `
           ($r.ni_wst -le $GATE_NAMEISH_WST) -and ($r.mr_wst -le $GATE_WORDRUN_WST) -and `
           ($r.bi_wst -le $GATE_TOPBI) -and ($r.al_wst -le $GATE_ALTLOOP) -and `
           ($r.selfbpb -ge $BPB_LO) -and ($r.selfbpb -le $BPB_HI)
}

$b25_pass = (Is-Pass 'C3.A_b25') -or (Is-Pass 'C3.C_b25')
$b50_pass = (Is-Pass 'C3.B_b50') -or (Is-Pass 'C3.D_b50')
$any_pass = $b25_pass -or $b50_pass

# best BPB among PASSING variants
$best_pass_lbl=''; $best_pass_bpb=$null
foreach ($l in @('C3.A_b25','C3.B_b50','C3.C_b25','C3.D_b50')) {
    if ((Is-Pass $l)) {
        if ($best_pass_bpb -eq $null -or $rows[$l].valbpb -lt $best_pass_bpb) {
            $best_pass_bpb=$rows[$l].valbpb; $best_pass_lbl=$l
        }
    }
}

if (-not $any_pass) {
    Write-Host '  -> NESSUN blend passa: 26 celle grezze non sono la strada.' -ForegroundColor Red
    Write-Host '     Torna a 13 celle (SEE-V2) e passa a 43.D byte-to-lane routing.' -ForegroundColor Red
} else {
    Write-Host ('  -> PROMOSSO: {0}  (valBPB={1:F4}, {2:+0.000;-0.000;0.000} vs SEE-V2)' -f `
        $best_pass_lbl, $best_pass_bpb, ($best_pass_bpb-$SEE_V2_BPB)) -ForegroundColor Green
    if ($b25_pass -and -not $b50_pass) {
        Write-Host '     blend 0.25 passa: il problema era forza plastica eccessiva.' -ForegroundColor Green
        Write-Host '     Omeostasi forte risolve -> promuovi SEE-V2.1.' -ForegroundColor Green
    } elseif ($b50_pass) {
        Write-Host '     blend 0.50 passa: 26 celle utili gia con omeostasi moderata.' -ForegroundColor Green
        Write-Host '     Promuovi SEE-V2.1 (preferisci il blend che minimizza BPB a parita di gate).' -ForegroundColor Green
    }
    Write-Host '     Il silicio scala memoria senza attractor: piu contesto, stabile.' -ForegroundColor Green
}

Write-Host ''
Write-Host '  Guida: un LLM non e solo predire meglio il prossimo byte;'
Write-Host '         e ricordare di piu senza cadere nei propri attractor.'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
