<#
Run latest policy (Walker2d by default).

Default: run latest checkpoint (model_*.zip) + matching vecnormalize_*.pkl.
-Best:    run eval\best_model.zip + vecnormalize_final.pkl from same run.
-UseFinalModel: run final_model.zip + vecnormalize_final.pkl.
-RunDir:  specify exact run directory (e.g., "2025-10-28/13-02-09")

Examples:
  .\run_2dn.ps1
  .\run_2dn.ps1 -Best
  .\run_2dn.ps1 -UseFinalModel
  .\run_2dn.ps1 -EnvId Humanoid-v5 -Best
  .\run_2dn.ps1 -RunDir "2025-10-28/13-02-09"
  .\run_2dn.ps1 -RunDir "2025-10-28/13-02-09" -Best
#>

[CmdletBinding()]
param(
  [string]$EnvId = "Walker2d-v5",
  [string]$RunDir = "",
  [switch]$UseFinalModel,
  [switch]$Best,
  [switch]$Deterministic = $true,
  [switch]$Render = $true
)

$ErrorActionPreference = "Stop"

# Prefer venv Python if present
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

function Ensure-File($path, $msg) {
  if (-not $path -or -not (Test-Path $path)) { throw $msg }
  return $path
}

# ---- Determine search path ----
if ($RunDir) {
  # User specified a specific run directory
  $searchPath = Join-Path "outputs" $RunDir
  if (-not (Test-Path $searchPath)) {
    throw "Specified run directory not found: $searchPath"
  }
  Write-Host "Using specified run directory: $searchPath" -ForegroundColor Cyan
} else {
  # Search all outputs
  $searchPath = "outputs"
  Write-Host "Searching in: $searchPath" -ForegroundColor Gray
}

# ---- Select model + vecnorm ----
if ($Best) {
  # Find latest best_model.zip under the search path
  $bestModel = Get-ChildItem $searchPath -Recurse -File -Filter "best_model.zip" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $mdl = Ensure-File $bestModel.FullName "No best_model.zip found under $searchPath."

  # best_model.zip lives in outputs\<run>\eval\  → vecnormalize_final.pkl is in the run root (parent of eval)
  $vec = Join-Path $bestModel.Directory.Parent.FullName "vecnormalize_final.pkl"
  if (-not (Test-Path $vec)) {
    Write-Warning "vecnormalize_final.pkl not found next to best_model; using most recent vecnormalize_final.pkl in search path."
    $vec = Get-ChildItem $searchPath -Recurse -File -Filter "vecnormalize_final.pkl" |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }
  }

}
elseif ($UseFinalModel) {
  # Find latest run that has a final_model.zip
  $run = Get-ChildItem $searchPath -Recurse -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName 'final_model.zip') } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $mdl = Ensure-File (Join-Path $run.FullName 'final_model.zip') "No final_model.zip found under $searchPath."
  $vec = Join-Path $run.FullName 'vecnormalize_final.pkl'

}
else {
  # Latest checkpoint model_*.zip
  $latestModel = Get-ChildItem $searchPath -Recurse -File -Filter "model_*.zip" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $mdl = Ensure-File $latestModel.FullName "No checkpoint model_*.zip found under $searchPath."

  # Try to match vecnormalize_<steps>.pkl in the same folder; fallback: latest vecnormalize_*.pkl
  if ($latestModel.BaseName -match 'model_(\d+)$') {
    $steps = $Matches[1]
    $candidate = Join-Path $latestModel.DirectoryName ("vecnormalize_{0}.pkl" -f $steps)
    if (Test-Path $candidate) {
      $vec = $candidate
    } else {
      Write-Warning "Matching vecnormalize_$steps.pkl not found; using latest vecnormalize_*.pkl."
      $latestVec = Get-ChildItem $searchPath -Recurse -File -Filter "vecnormalize_*.pkl" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
      $vec = $latestVec?.FullName
    }
  }
}

Write-Host "Env:    $EnvId" -ForegroundColor Cyan
Write-Host "Model:  $mdl" -ForegroundColor Green
if ($vec) { Write-Host "VecNorm: $vec" -ForegroundColor Green } else { Write-Warning "No VecNormalize file found. Continuing without it." }

# ---- Build args and launch ----
$argsList = @("evaluate.py", "--env_id", $EnvId, "--model_path", $mdl)
if ($vec) { $argsList += @("--vecnorm_path", $vec) }
if ($Render) { $argsList += "--render" }
if ($Deterministic) { $argsList += "--deterministic" }

Write-Host "`n>>> Launching evaluation..." -ForegroundColor Yellow
Write-Host ("`"" + $py + "`" " + ($argsList -join " "))

& "$py" @argsList
