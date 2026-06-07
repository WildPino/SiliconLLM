# Phase 44.H - Targeted Attractor Forensics (diagnostic, zero training)
#
# 44.G could not decide: the byte-period detector missed the real failure (topBi is
# a WORD-level bigram domination, not a short byte cycle), the loop bucket was nearly
# empty, and flip/Lwin were LOWER in loop than clean. So L2 is neither absolved nor
# condemned. 44.H looks at the FIRST instant the bigram loop is born, with causal
# ablations in generation - no averages over an empty bucket, no training.
#
# Method (per failing config/seed at T=0.55):
#   1. Detect the warning: first time any word-bigram count reaches WARN (loop forming).
#   2. Forensic window around it (rich per-step log: bigram+count, SEE top byte/logit,
#      L2 top byte/logit, argmax-full, ratio/gamma/flip, top1-top2 margin, gate).
#   3. Ablations armed AT the warning (causal "intervene when the loop starts"):
#        none | l2zero (L2 logit->0) | l2freeze (L2 state frozen) | l2reset (L2 zeroed
#        once) | cap1.0 | cap0.5 (tighter cap after warning).
#
# The real question -> verdict:
#   l2zero breaks the loop                 -> A: needs conditional L2 GATING (readout)
#   l2zero doesn't but freeze/reset does   -> B: the L2 MEMORY/state is the attractor
#   none break (or ablation makes it WORSE)-> C: attractor is SEE-/sampling-side; L2 spectator/brake
#
# Configs audited (weights from the 44.F run): D1_cap20inf, F1, F3, F0.
#
# Run:  .\benchmarks\phase38-42\phase44h_forensics.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase44h'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$TEMP    = '0.55'
$RNG     = 12345
$GEN_LEN = 2000
$WARMUP  = 5000
$WARN    = 4        # bigram count that flags "loop forming"
$SCAN_N  = 12       # seeds scanned per config
$WORST_K = 3        # worst-topBi seeds audited per config
$BI_GATE = 8        # topBi pass threshold (loop "broken" if <= this)

$configs = @(
    @{ label='F0';          sfx='_F0.bin' },
    @{ label='D1_cap20inf'; sfx='_D1_cap20inf.bin' },
    @{ label='F1';          sfx='_F1.bin' },
    @{ label='F3';          sfx='_F3.bin' }
)
$modes = @(
    @{ l='none';     a='none';     c=$null },
    @{ l='l2zero';   a='l2zero';   c=$null },
    @{ l='l2freeze'; a='l2freeze'; c=$null },
    @{ l='l2reset';  a='l2reset';  c=$null },
    @{ l='cap1.0';   a='cap';      c='0.010' },
    @{ l='cap0.5';   a='cap';      c='0.005' }
)

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING (generator with --forensic / --ablate) ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

foreach ($c in $configs) {
    $c.file = $WDIR + '\phase44f' + $c.sfx
    if (-not (Test-Path $c.file)) { Write-Error ("Missing 44.F weights: " + $c.file + " (run phase44f_captrain.ps1 first)"); exit 1 }
}

$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$ScanSeeds = @()
for ($i = 0; $i -lt $SCAN_N; $i++) { $ScanSeeds += [int]([math]::Floor($i * ($fsz - $margin) / $SCAN_N)) }

# -------- Run one generation, return topbi/warn/warnstep ----------------------
function Run-Gen($wfile,$off,$ablate,$acap,$forensic) {
    $sf  = $RDIR + '\_stderr.txt'
    $out = $RDIR + '\_stdout.bin'
    $al = ('"{0}" "{1}" --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6} --warn-bigram {7}' -f `
           $TS_DATA,$wfile,$GEN_LEN,$WARMUP,$TEMP,$off,$RNG,$WARN)
    if ($ablate -ne 'none') { $al += " --ablate $ablate"; if ($acap) { $al += " --ablate-cap $acap" } }
    if ($forensic) { $al += (' --forensic "{0}"' -f $forensic) }
    $p = Start-Process -FilePath "$BINDIR\phase44_generator.exe" -ArgumentList $al `
            -RedirectStandardOutput $out -RedirectStandardError $sf -NoNewWindow -PassThru -Wait
    $topbi = $null; $warned = 0; $wstep = -1
    foreach ($line in [System.IO.File]::ReadLines($sf)) {
        if ($line -match 'TOPBI_FINAL:\s*(\d+)\s+WARNED:\s*(\d+)\s+WARN_STEP:\s*(-?\d+)') {
            $topbi = [int]$Matches[1]; $warned = [int]$Matches[2]; $wstep = [int]$Matches[3]
        }
    }
    return [PSCustomObject]@{ topbi=$topbi; warned=$warned; wstep=$wstep; exit=$p.ExitCode }
}

# -------- Scan for worst seeds ------------------------------------------------
Write-Host ''
Write-Host ('=== Scan {0} seeds/config at T={1} for worst topBi ===' -f $SCAN_N,$TEMP) -ForegroundColor Yellow
$worst = @{}
foreach ($c in $configs) {
    $rows = @()
    foreach ($off in $ScanSeeds) {
        $r = Run-Gen $c.file $off 'none' $null $null
        $rows += [PSCustomObject]@{ off=$off; topbi=$r.topbi; warned=$r.warned; wstep=$r.wstep }
        Write-Host ('  {0,-14} off={1,-9} topBi={2,-3} warn@{3}' -f $c.label,$off,$r.topbi,$r.wstep)
    }
    $sel = @($rows | Sort-Object -Property @{Expression={[int]$_.topbi};Descending=$true} | Select-Object -First $WORST_K)
    $worst[$c.label] = $sel
    Write-Host ('  -> {0} worst seeds: {1}' -f $c.label, (($sel | ForEach-Object { '{0}(bi{1})' -f $_.off,$_.topbi }) -join ', ')) -ForegroundColor DarkCyan
}

# -------- Ablation matrix on worst seeds --------------------------------------
Write-Host ''
Write-Host ('=== Ablation matrix on {0} worst seeds/config (warn-bigram={1}) ===' -f $WORST_K,$WARN) -ForegroundColor Yellow
$result = @{}   # label -> mode -> @(topbi per seed)
foreach ($c in $configs) {
    $result[$c.label] = @{}
    foreach ($m in $modes) { $result[$c.label][$m.l] = @() }
    $first = $true
    foreach ($w in $worst[$c.label]) {
        foreach ($m in $modes) {
            $forensic = $null
            if ($first -and $m.l -eq 'none') { $forensic = $RDIR + '\forensic_' + $c.label + '_off' + $w.off + '.tsv' }
            $r = Run-Gen $c.file $w.off $m.a $m.c $forensic
            $result[$c.label][$m.l] += [int]$r.topbi
            Write-Host ('  {0,-14} off={1,-9} {2,-9} topBi={3}' -f $c.label,$w.off,$m.l,$r.topbi)
        }
        $first = $false
    }
}

function MeanI([int[]]$a){ if($a.Count -eq 0){return 0.0}; [math]::Round(($a|Measure-Object -Average).Average,1) }
function MaxI([int[]]$a){ if($a.Count -eq 0){return 0}; ($a|Measure-Object -Maximum).Maximum }

# -------- Table ---------------------------------------------------------------
Write-Host ''
Write-Host '======== topBi by ablation (mean / worst over audited seeds) ========' -ForegroundColor Cyan
Write-Host ('  {0,-14} {1,10} {2,10} {3,10} {4,10} {5,10} {6,10}' -f 'config','none','l2zero','l2freeze','l2reset','cap1.0','cap0.5')
Write-Host ('  ' + '-'*84)
foreach ($c in $configs) {
    $r = $result[$c.label]
    $cells = @()
    foreach ($m in $modes) { $cells += ('{0}/{1}' -f (MeanI $r[$m.l]),(MaxI $r[$m.l])) }
    Write-Host ('  {0,-14} {1,10} {2,10} {3,10} {4,10} {5,10} {6,10}' -f $c.label,$cells[0],$cells[1],$cells[2],$cells[3],$cells[4],$cells[5])
}
Write-Host '  (cells = mean/worst topBi; loop "broken" if <= 8)' -ForegroundColor DarkGray

# -------- Forensic window at loop birth (worst seed, normal run) --------------
function ChByte([int]$b){ if($b -ge 32 -and $b -lt 127){ "'" + [char]$b + "'" } else { '0x{0:x2}' -f $b } }
Write-Host ''
Write-Host '======== Forensic window at loop birth (worst seed, normal run) ========' -ForegroundColor Cyan
foreach ($c in $configs) {
    $w0 = $worst[$c.label][0]
    $ff = $RDIR + '\forensic_' + $c.label + '_off' + $w0.off + '.tsv'
    if (-not (Test-Path $ff)) { continue }
    $lines = @([System.IO.File]::ReadLines($ff) | Where-Object { $_ -and $_[0] -ne '#' -and $_ -notmatch '^step' })
    # find onset = first row with warned=1
    $onset = -1
    for ($k=0; $k -lt $lines.Count; $k++) { $f=$lines[$k] -split "`t"; if ($f.Count -ge 16 -and [int]$f[15] -eq 1) { $onset=$k; break } }
    Write-Host ''
    Write-Host ('  {0}  off={1}  worst topBi={2}  onset row={3}' -f $c.label,$w0.off,$w0.topbi,$onset) -ForegroundColor Yellow
    if ($onset -lt 0) { Write-Host '    (no warning fired on this seed)' -ForegroundColor DarkGray; continue }
    $a = [math]::Max(0,$onset-8); $b = [math]::Min($lines.Count-1,$onset+18)
    Write-Host ('    {0,5} {1,5} {2,-16} {3,4} {4,6} {5,6} {6,4} {7,7} {8,7} {9,4} {10,7} {11,4} {12,4}' -f `
        'step','byte','bigram','bicnt','seeTop','l2Top','argF','ratio','gamma','flip','margin','gate','warn')
    for ($k=$a; $k -le $b; $k++) {
        $f = $lines[$k] -split "`t"; if ($f.Count -lt 16) { continue }
        $mark = if ([int]$f[15] -eq 1 -and ($k -eq 0 -or ([int]($lines[$k-1] -split "`t")[15]) -eq 0)) { '<<' } else { '' }
        Write-Host ('    {0,5} {1,5} {2,-16} {3,4} {4,6} {5,6} {6,4} {7,7} {8,7} {9,4} {10,7} {11,4} {12,4} {13}' -f `
            $f[0],(ChByte([int]$f[1])),$f[2],$f[3],(ChByte([int]$f[5])),(ChByte([int]$f[7])),(ChByte([int]$f[9])),`
            $f[10],$f[11],$f[12],$f[13],$f[14],$f[15],$mark)
    }
}

# -------- Verdict A/B/C per config --------------------------------------------
Write-Host ''
Write-Host '======== Verdetto 44.H: A (gating) / B (memoria) / C (SEE-spectator) ========' -ForegroundColor Yellow
foreach ($c in $configs) {
    $r = $result[$c.label]
    $base = MeanI $r['none']
    $z  = MeanI $r['l2zero']; $fz = MeanI $r['l2freeze']; $rs = MeanI $r['l2reset']
    $c10= MeanI $r['cap1.0']; $c05= MeanI $r['cap0.5']
    function Breaks($v){ ($v -le $BI_GATE) -or ($base -gt 0 -and $v -le $base*0.6) }
    function Worsens($v){ ($base -gt 0 -and $v -ge $base*1.5) }
    $msg=''; $col='Gray'
    if (Breaks $z) {
        $msg = ("A: l2zero ROMPE il loop (topBi {0}->{1}). L'autorita logit istantanea di L2 guida il loop -> gating L2 condizionale." -f $base,$z); $col='Yellow'
    } elseif ((Breaks $fz) -or (Breaks $rs)) {
        $msg = ("B: l2zero non basta ({0}) ma freeze {1}/reset {2} rompe -> e' la MEMORIA/state di L2 l'attrattore, non i logit -> homeostasis/update L2." -f $z,$fz,$rs); $col='Green'
    } elseif ((Breaks $c05) -or (Breaks $c10)) {
        $msg = ("A': solo un cap molto stretto rompe (cap1.0 {0}, cap0.5 {1}); l2zero {2}. L2 va tenuto a guinzaglio CORTO e condizionale." -f $c10,$c05,$z); $col='DarkCyan'
    } elseif ((Worsens $z) -and -not (Breaks $fz) -and -not (Breaks $rs)) {
        $msg = ("C: zeroare L2 PEGGIORA ({0}->{1}) e niente rompe. L2 e spettatore/freno; l'attrattore e SEE-/sampling-side -> rivedere substrate o sampling, non L2." -f $base,$z); $col='Magenta'
    } else {
        $msg = ("inconcludente: none {0} zero {1} freeze {2} reset {3} cap1.0 {4} cap0.5 {5} - leggere i forensic grezzi." -f $base,$z,$fz,$rs,$c10,$c05)
    }
    Write-Host ('  {0,-14}: {1}' -f $c.label,$msg) -ForegroundColor $col
}

Write-Host ''
Write-Host '  Note: F0 (no cap) = referenza. Se anche su F0 nessuna ablation L2 rompe il loop,'
Write-Host '  l attrattore non e nel contributo/stato L2 -> il prossimo passo non e su L2.'
Write-Host ''
Write-Host '  Domanda 44.H: zeroare il contributo L2 rompe il bigram loop (A), serve freeze/reset'
Write-Host '  della memoria L2 (B), o nessuno e l attrattore e SEE-/sampling-side (C)?'
Write-Host ''
Write-Host ('  Forensic TSV grezzi: ' + $RDIR)
