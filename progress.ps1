# Marathi Story Voice - queue progress
#   .\progress.ps1           one snapshot
#   .\progress.ps1 -Watch    refresh every 30s until you press Ctrl+C
param([switch]$Watch, [int]$Every = 30)

$PROJ = "C:\marathi_tts"
$LOG  = "$PROJ\out\queue_run.log"

function Show-Progress {
    if (-not (Test-Path $LOG)) {
        Write-Host "No run log at $LOG - the queue runner has not been started." -ForegroundColor Yellow
        return
    }
    # the runner writes UTF-8; without this the Devanagari comes out as mojibake
    $log = Get-Content $LOG -Encoding utf8 -ErrorAction SilentlyContinue

    $alive = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -like "*run_queue.py*" })
    $ok      = @($log | Select-String -Pattern '^\s+OK\s').Count
    $failed  = @($log | Select-String -Pattern '^\s+FAILED:').Count
    $cur     = $log | Select-String -Pattern '^\[(\d+)/(\d+)\]\s+(.*)$' | Select-Object -Last 1
    $chunk   = $log | Select-String -Pattern 'Chunk (\d+)/(\d+)' | Select-Object -Last 1
    $eta     = $log | Select-String -Pattern '~([\d.]+) min left' | Select-Object -Last 1

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  Marathi Story Voice - queue progress        $(Get-Date -Format 'HH:mm:ss')"
    Write-Host "============================================================"

    if ($alive.Count -gt 0) {
        Write-Host "  status    : RUNNING (pid $($alive[0].ProcessId))" -ForegroundColor Green
    } elseif ($ok + $failed -gt 0 -and $log -match 'done: ') {
        Write-Host "  status    : FINISHED" -ForegroundColor Cyan
    } else {
        Write-Host "  status    : NOT RUNNING - it stopped or was closed" -ForegroundColor Red
    }

    if ($cur) {
        $i, $n, $name = $cur.Matches[0].Groups[1].Value, $cur.Matches[0].Groups[2].Value, $cur.Matches[0].Groups[3].Value
        Write-Host "  story     : $i of $n  -  $name"
    }
    Write-Host "  completed : $ok   failed: $failed"

    if ($chunk) {
        $c = [int]$chunk.Matches[0].Groups[1].Value
        $t = [int]$chunk.Matches[0].Groups[2].Value
        $pct = if ($t) { [math]::Round($c / $t * 100) } else { 0 }
        $bar = ("#" * [math]::Floor($pct / 4)).PadRight(25, '.')
        $etatxt = if ($eta) { "  ~$($eta.Matches[0].Groups[1].Value) min left" } else { "" }
        Write-Host "  chunk     : $c of $t  [$bar] $pct%$etatxt"
    }

    # actual rendered chunks on disk, and how fast they are landing
    $dir = Get-ChildItem "$PROJ\out\parts" -Directory -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($dir) {
        $w = @(Get-ChildItem "$($dir.FullName)\*.wav" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime)
        if ($w.Count -ge 2) {
            $per = ($w[-1].LastWriteTime - $w[0].LastWriteTime).TotalSeconds / ($w.Count - 1)
            Write-Host ("  rate      : {0:N1}s per chunk" -f $per)
        }
        $age = [math]::Round(((Get-Date) - $dir.LastWriteTime).TotalMinutes, 1)
        if ($age -gt 5 -and $alive.Count -gt 0) {
            Write-Host "  WARNING   : nothing written for $age min - it may be stuck" -ForegroundColor Yellow
        }
        Write-Host "  parts     : $($dir.Name)"
    }

    $waiting = @(Get-ChildItem "$PROJ\queue\*.txt" -ErrorAction SilentlyContinue).Count
    Write-Host "  still queued: $waiting file(s) in queue\"

    $recent = Get-ChildItem "$PROJ\out\*.wav" -ErrorAction SilentlyContinue |
              Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-24) } |
              Sort-Object LastWriteTime -Descending | Select-Object -First 5
    if ($recent) {
        Write-Host ""
        Write-Host "  finished audio (newest first):"
        foreach ($r in $recent) {
            $mins = [math]::Round($r.Length / (24000 * 4) / 60, 1)
            Write-Host ("    {0}  {1,5} min  {2}" -f $r.LastWriteTime.ToString('HH:mm'), $mins, $r.Name)
        }
    }
    Write-Host "============================================================"
}

if ($Watch) {
    Write-Host "Watching every $Every s. Ctrl+C to stop." -ForegroundColor DarkGray
    while ($true) { Clear-Host; Show-Progress; Start-Sleep -Seconds $Every }
} else {
    Show-Progress
}
