# Table 1 / 3 재실험 + ablation + 영상 정보 표를 순차 실행한다.
# 사용법 (프로젝트 폴더에서):
#   .\run_remaining.ps1           # 이어서 실행 (이미 있는 결과 파일은 건너뜀)
#   .\run_remaining.ps1 -Clean    # results_reset 을 지우고 처음부터 (이전 환경에서 만든 결과가 섞였을 때)
# venv 를 활성화하지 않아도 venv\Scripts\python.exe 를 직접 사용한다.
# 로그: logs\run_remaining_<시각>.log / 실패한 단계는 건너뛰고 계속 진행하며 마지막에 요약한다.

param([switch]$Clean)

$ErrorActionPreference = "Continue"
# 파이프로 넘겨도 출력이 바로 보이도록 버퍼링 끄기 + 한글 깨짐 방지
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv python 없음: $py"; exit 1 }
Set-Location $PSScriptRoot

# GPU 확인 (CPU 로 도는 것을 막는다)
& $py -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { Write-Host "CUDA 를 사용할 수 없어 중단합니다."; exit 1 }

if ($Clean -and (Test-Path "results_reset")) {
    Remove-Item -Recurse -Force "results_reset"
    Write-Host "results_reset 삭제 후 처음부터 실행"
}

New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\run_remaining_$(Get-Date -Format yyyyMMdd_HHmmss).log"

$steps = @(
    "extract_coords_new.py yolov12s-face",
    "extract_coords_old.py yolov12s-face",
    "extract_coords_new.py yolov11n-face",
    "extract_coords_old.py yolov11n-face",
    "extract_coords_new.py yolov10n-face",
    "extract_coords_old.py yolov10n-face",
    "compare_all_V3.py results_reset/yolov12s-face",
    "compare_all_V3.py results_reset/yolov11n-face",
    "compare_all_V3.py results_reset/yolov10n-face",
    "analyze_ablation.py results_reset/yolov12s-face",
    "analyze_ablation.py results_reset/yolov11n-face",
    "analyze_ablation.py results_reset/yolov10n-face",
    "video_stats.py results_reset/yolov12s-face",
    "measure_identity.py"
)

$failed = @()
foreach ($step in $steps) {
    $start = Get-Date
    "[$start] >>> python $step" | Tee-Object -FilePath $log -Append
    $argv = $step -split " "
    & $py @argv 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $min = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    if ($code -ne 0) {
        $failed += $step
        "[FAIL] $step (exit $code, ${min}분)" | Tee-Object -FilePath $log -Append
    } else {
        "[OK]   $step (${min}분)" | Tee-Object -FilePath $log -Append
    }
}

"`n===== 완료 =====" | Tee-Object -FilePath $log -Append
if ($failed.Count -gt 0) {
    "실패한 단계:" | Tee-Object -FilePath $log -Append
    $failed | ForEach-Object { "  - $_" | Tee-Object -FilePath $log -Append }
} else {
    "모든 단계 성공" | Tee-Object -FilePath $log -Append
}
