# Phase 44.A - Boundary-Gated L2 Memory Tribunal
#
# Question: does the substrate improve when a SLOW memory (L2) updates only at
# the right moments (boundaries), not every byte? L2 = harness-level gated EMA of
# a fixed random projection of SEE; readout linear over [SEE 192 | L2 64].
#
# Base: C2.A (weights/phase43c2_C2A.bin), frozen Oja, clamp on, readout stupid.
#
# Configs: 44.0 control (no L2) + punct/whitespace (alpha 0.95/0.99) +
#          surprise top10/top20 + entropy-high + combined.
#
# Metrics: teacher-forced val BPB (trainer) + deterministic word-gate
#          (32 samples = 16 seeds x 2 rng) at T=0.65 and T=0.55.
#
# Promotion (per user):
#   branch1: val BPB improves >= 0.005 vs C2.A  AND word-gate not worse than C2.A
#   branch2: val BPB improves >= 0.002 vs C2.A  AND word-gate passes fully
#
# Run:  .\benchmarks\phase38-42\phase44a_boundary.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase44a'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W  = $WDIR + '\phase43c2_C2A.bin'
$WP     = $WDIR + '\phase44'
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
$GATE_NAMEISH_WST = 20.0
$GATE_WORDRUN_WST = 5.0
$GATE_TOPBI       = 8.0
$GATE_ALTLOOP     = 2.0

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44a_boundary.c' @SRC_CORE -o "$BINDIR\phase44a_boundary.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44a_boundary'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Train ---------------------------------------------------------------
Write-Host ''
Write-Host '=== Phase 44.A - Training 9 configs (L2 boundary gates) ===' -ForegroundColor Yellow
Write-Host '  WARNING: 256D features ~30% more memory than 192D. Heavy run.'
Write-Host ''
$TRAIN_OUT = $RDIR + '\phase44a_train.txt'
& "$BINDIR\phase44a_boundary.exe" $TS_DATA $WP $C2A_W 2>&1 | Tee-Object $TRAIN_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
Write-Host 'Training complete.' -ForegroundColor Green

function Get-TrainBPB([string]$sfx) {
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $TRAIN_OUT | Select-Object -Last 1
    if ($line) { return [double](($line.Line -replace '.*BPB=','').Trim()) }
    return $null
}

# label, exe, weights, valbpb, iscontrol/isref
$gen_configs = @(
    @{ label='C2.A';     exe='phase43_generator.exe'; weights=$C2A_W;          valbpb=$C2A_BPB;            ref=$true },
    @{ label='44.0';     exe='phase44_generator.exe'; weights=$WP+'_440.bin';     valbpb=(Get-TrainBPB '_440.bin');     ref=$true },
    @{ label='44.A_a95'; exe='phase44_generator.exe'; weights=$WP+'_44A_a95.bin'; valbpb=(Get-TrainBPB '_44A_a95.bin'); ref=$false },
    @{ label='44.A_a99'; exe='phase44_generator.exe'; weights=$WP+'_44A_a99.bin'; valbpb=(Get-TrainBPB '_44A_a99.bin'); ref=$false },
    @{ label='44.B_a95'; exe='phase44_generator.exe'; weights=$WP+'_44B_a95.bin'; valbpb=(Get-TrainBPB '_44B_a95.bin'); ref=$false },
    @{ label='44.B_a99'; exe='phase44_generator.exe'; weights=$WP+'_44B_a99.bin'; valbpb=(Get-TrainBPB '_44B_a99.bin'); ref=$false },
    @{ label='44.C_t10'; exe='phase44_generator.exe'; weights=$WP+'_44C_t10.bin'; valbpb=(Get-TrainBPB '_44C_t10.bin'); ref=$false },
    @{ label='44.C_t20'; exe='phase44_generator.exe'; weights=$WP+'_44C_t20.bin'; valbpb=(Get-TrainBPB '_44C_t20.bin'); ref=$false },
    @{ label='44.D_hi';  exe='phase44_generator.exe'; weights=$WP+'_44D_hi.bin';  valbpb=(Get-TrainBPB '_44D_hi.bin');  ref=$false },
    @{ label='44.E';     exe='phase44_generator.exe'; weights=$WP+'_44E.bin';      valbpb=(Get-TrainBPB '_44E.bin');     ref=$false }
)

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

# -------- Generate: config x temp x rng x seed --------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

# agg[label][temp] = @{mets=@(); bpbs=@()}
$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{}; foreach ($tp in $TEMPS) { $agg[$cfg.label][$tp] = @{ mets=@(); bpbs=@() } } }

Write-Host ''
Write-Host ('=== Word-gate: temps {0}, {1} seeds x {2} rng = {3}/temp ===' -f ($TEMPS -join ','),$NSEEDS,$RNGS.Count,($NSEEDS*$RNGS.Count)) -ForegroundColor Yellow
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
                Write-Host ('  {0,-9} T{1} rng{2} s{3,-2} ...' -f $cfg.label,$tp,$rng,$si) -NoNewline
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

# -------- Tables per temp -----------------------------------------------------
foreach ($tp in $TEMPS) {
    Write-Host ''
    Write-Host ('======== T={0} : worst-case over 32 samples ========' -f $tp) -ForegroundColor Cyan
    Write-Host ('  {0,-9} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8}' -f 'config','valBPB','dBPB','nameWst','runWst','topBi','altLp','selfBPB')
    Write-Host ('  ' + '-'*74)
    foreach ($cfg in $gen_configs) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { Write-Host ('  {0,-9} [no data]' -f $cfg.label); continue }
        $w = Worst $cfg.label $tp
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        $db = if ($null -ne $cfg.valbpb) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$C2A_BPB) } else { '-' }
        Write-Host ('  {0,-9} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8}' -f `
            $cfg.label,$vb,$db,('{0:F1}' -f $w.ni),('{0:F0}' -f $w.mr),('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F2}' -f $w.self))
    }
}

# -------- Control sanity ------------------------------------------------------
$ctl = Get-TrainBPB '_440.bin'
Write-Host ''
if ($null -ne $ctl) {
    $d = [math]::Abs($ctl - $C2A_BPB)
    $ok = if ($d -le 0.01) {'OK'} else {'WARN'}
    Write-Host ('=== Control 44.0 vs C2.A: {0:F4} vs {1:F4} (|d|={2:F4}) [{3}] ===' -f $ctl,$C2A_BPB,$d,$ok) -ForegroundColor Cyan
}

# -------- Promotion + interpretation ------------------------------------------
function In-Band($s){ return ($s -ge $BPB_LO) -and ($s -le $BPB_HI) }
function Full-Gate($w){ return ($w.ni -le $GATE_NAMEISH_WST) -and ($w.mr -le $GATE_WORDRUN_WST) -and ($w.bi -le $GATE_TOPBI) -and ($w.al -le $GATE_ALTLOOP) -and (In-Band $w.self) }
function Not-Worse($w,$c){ return ($w.ni -le $c.ni) -and ($w.mr -le $c.mr) -and ($w.bi -le $c.bi) -and ($w.al -le $c.al) -and (In-Band $w.self) }

Write-Host ''
Write-Host '=== Verdetto 44.A ===' -ForegroundColor Yellow
$winner=$null; $win_bpb=$null; $win_branch=''; $win_temp=''
foreach ($cfg in $gen_configs) {
    if ($cfg.ref) { continue }
    if ($null -eq $cfg.valbpb) { continue }
    foreach ($tp in $TEMPS) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { continue }
        $w = Worst $cfg.label $tp
        $c2a = Worst 'C2.A' $tp
        $b1 = ($cfg.valbpb -le ($C2A_BPB-0.005)) -and (Not-Worse $w $c2a)
        $b2 = ($cfg.valbpb -le ($C2A_BPB-0.002)) -and (Full-Gate $w)
        if ($b1 -or $b2) {
            if ($null -eq $win_bpb -or $cfg.valbpb -lt $win_bpb) {
                $win_bpb=$cfg.valbpb; $winner=$cfg.label; $win_temp=$tp
                $win_branch = if ($b2) {'branch2 (>=0.002 + full gate)'} else {'branch1 (>=0.005 + not worse)'}
            }
        }
    }
}

if ($null -ne $winner) {
    Write-Host ('  -> PROMOSSO: {0} @ T={1} ({2}), valBPB {3:F4} ({4:+0.000;-0.000;0.000} vs C2.A)' -f `
        $winner,$win_temp,$win_branch,$win_bpb,($win_bpb-$C2A_BPB)) -ForegroundColor Green
    Write-Host '     Una memoria lenta aggiornata nei momenti giusti MIGLIORA il substrate.' -ForegroundColor Green
    Write-Host '     Primo passo verso contesto gerarchico da LLM. Definisci SEE-V4 = C2.A + L2.' -ForegroundColor Green
    if ($winner -like '44.A*' -or $winner -like '44.B*' -or $winner -like '44.E*') {
        Write-Host '     Gate sintattico (punct/whitespace) -> il silicio vuole confini semplici.' -ForegroundColor Green
    } elseif ($winner -like '44.C*' -or $winner -like '44.D*') {
        Write-Host '     Gate dinamico (surprise/entropy) -> il silicio vuole confini emergenti.' -ForegroundColor Green
    }
} else {
    Write-Host '  -> Nessun gate promuove (gate attuali).' -ForegroundColor Yellow
    Write-Host '     Leggi i pattern:' -ForegroundColor Yellow
    Write-Host '     - nessuno migliora BPB: L2 non leggibile dal readout lineare o projection sbagliata' -ForegroundColor Yellow
    Write-Host '     - BPB su ma generation giu: L2 accumula attractor -> serve homeostasis/gated reset su L2' -ForegroundColor Yellow
    Write-Host '     - generation su ma BPB flat: L2 utile per stabilita closed-loop, non per compression' -ForegroundColor Yellow
    Write-Host '     Decidi se iterare projection/alpha o passare a L2 trainable / boundary diversi.' -ForegroundColor Yellow
}

Write-Host ''
Write-Host '  Domanda di fase: il substrate migliora quando una memoria lenta'
Write-Host '  viene aggiornata solo nei momenti giusti?'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
