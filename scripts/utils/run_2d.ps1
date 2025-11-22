<#
Run latest policy (Walker2d by default). Uses checkpoint+matching VecNormalize by default.
-UseFinalModel to run final_model.zip + vecnormalize_final.pkl instead.
#>

param(
  [string]$EnvId = "Walker2d-v5",
  [switch]$UseFinalModel,
  [switch]$Deterministic = $true,
  [switch]$Render = $true
)

# Prefer venv python if present
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

if ($UseFinalModel) {
  $run = Get-ChildItem outputs -Recurse -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName 'final_model.zip') } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $run) { Write-Error "No run with final_model.zip found."; exit 1 }
  $mdl = Join-Path $run.FullName 'final_model.zip'
  $vec = Join-Path $run.FullName 'vecnormalize_final.pkl'
} else {
  # Latest checkpoint model
  $latestModel = Get-ChildItem outputs -Recurse -File -Filter "model_*.zip" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latestModel) { Write-Error "No checkpoint model_*.zip found."; exit 1 }
  $mdl = $latestModel.FullName

  # Try to find the matching vecnormalize_<steps>.pkl
  if ($latestModel.BaseName -match 'model_(\d+)$') {
    $steps = $Matches[1]
    $candidate = Join-Path $latestModel.DirectoryName ("vecnormalize_{0}.pkl" -f $steps)
    if (Test-Path $candidate) {
      $vec = $candidate
    } else {
      # fallback: latest vecnormalize_* anywhere
      $latestVec = Get-ChildItem outputs -Recurse -File -Filter "vecnormalize_*.pkl" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
      $vec = $latestVec?.FullName
    }
  }
}

Write-Host "Model:  $mdl" -ForegroundColor Green
if ($vec) { Write-Host "VecNorm: $vec" -ForegroundColor Green } else { Write-Warning "No VecNormalize file found. Continuing without it." }

# Build args as an array (not a single string!)
# Use the evaluate script from scripts/evaluate/ directory
$evaluateScript = Join-Path $PSScriptRoot "..\evaluate\evaluate_sb3.py"
$argsList = @($evaluateScript, "--env_id", $EnvId, "--model_path", $mdl)
if ($vec) { $argsList += @("--vecnorm_path", $vec) }
if ($Render) { $argsList += "--render" }
if ($Deterministic) { $argsList += "--deterministic" }

Write-Host "`n>>> Launching evaluation..." -ForegroundColor Yellow
Write-Host ($py + " " + ($argsList -join " "))

# Invoke: & <cmd> <args...>
& $py @argsList
