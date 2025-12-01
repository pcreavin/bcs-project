<#
Helper PowerShell script to create a virtual environment and show activation options.

Usage:
  - Run this script from the project root in PowerShell to create the venv and see instructions.
  - If you want to change the execution policy for your user (recommended once), run:
      Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

Notes:
  - This script will NOT persistently activate the venv in your current shell (activation is process-local).
  - After running this script, follow the printed instructions to activate the venv in your current terminal.
#>

Param()

Write-Host "Checking for existing virtual environment ('venv')..."
if (-not (Test-Path -Path "venv")) {
    Write-Host "Virtual environment not found. Creating venv..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create venv. Ensure 'python' is on PATH and is the desired version."
        exit 1
    }
    Write-Host "Created virtual environment in ./venv"
} else {
    Write-Host "Virtual environment already exists at ./venv"
}

Write-Host "\nAttempting to set execution policy for CurrentUser to RemoteSigned (will succeed if allowed)..."
try {
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force -ErrorAction Stop
    Write-Host "Set execution policy for CurrentUser to RemoteSigned"
} catch {
    Write-Warning "Could not set execution policy from this session (this can happen if scripts are blocked)."
    Write-Host "You can either run the one-time bypass to activate, or run the following command in an elevated PowerShell if you want to change the policy for your user:" -ForegroundColor Yellow
    Write-Host "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force" -ForegroundColor Yellow
}

Write-Host "\nNext steps (choose one):" -ForegroundColor Green
Write-Host "1) Temporary bypass (do this in the same PowerShell session to immediately activate):" -ForegroundColor Cyan
Write-Host "   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force; .\venv\Scripts\Activate.ps1" -ForegroundColor White

Write-Host "2) Persistent for your user (run once):" -ForegroundColor Cyan
Write-Host "   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force" -ForegroundColor White
Write-Host "   Then: .\venv\Scripts\Activate.ps1" -ForegroundColor White

Write-Host "3) Use Command Prompt instead (no PowerShell policy changes):" -ForegroundColor Cyan
Write-Host "   Open Command Prompt and run: venv\Scripts\activate" -ForegroundColor White

Write-Host "\nIf you want, you can also run the helper batch file for CMD: scripts\\activate_cmd.bat" -ForegroundColor Green

Write-Host "\nFinally, install project dependencies once venv is active:" -ForegroundColor Green
Write-Host "   pip install -r requirements.txt" -ForegroundColor White

Write-Host "\nDone. If activation fails, paste the error message in an issue or ask for help." -ForegroundColor Green
