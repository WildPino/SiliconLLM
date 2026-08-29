# Independent machine sampler for P1 Stage 2 (Amendment 1 A1.6).
# The quiescence banner in the D5 harness runs ONCE from main before mode dispatch, so it certifies the
# instant the binary started and nothing about the run. This samples the machine repeatedly DURING the run,
# from a separate process, and records every sample. Stop by creating the stop file.
param([string]$Out, [string]$Stop, [int]$IntervalMs = 1500)
$self = @('nibble_pack_bench','powershell','pwsh','conhost')
"ts,n_proc_gt300MB,max_other_WS_MB,max_other_name,free_phys_MB,cpu_queue" | Out-File -FilePath $Out -Encoding utf8
while (-not (Test-Path $Stop)) {
    $ps = Get-Process | Where-Object { $_.WorkingSet64 -gt 300MB -and $self -notcontains $_.ProcessName }
    $n  = ($ps | Measure-Object).Count
    $top = $ps | Sort-Object -Descending WorkingSet64 | Select-Object -First 1
    $mx = if ($top) { [math]::Round($top.WorkingSet64/1MB,0) } else { 0 }
    $nm = if ($top) { $top.ProcessName } else { 'none' }
    $os = Get-CimInstance Win32_OperatingSystem
    $fp = [math]::Round($os.FreePhysicalMemory/1KB,0)
    $q  = 0
    try { $q = [int](Get-CimInstance Win32_PerfFormattedData_PerfOS_System).ProcessorQueueLength } catch {}
    "$(Get-Date -Format 'HH:mm:ss'),$n,$mx,$nm,$fp,$q" | Out-File -FilePath $Out -Append -Encoding utf8
    Start-Sleep -Milliseconds $IntervalMs
}
