# ============================================================
# MS for Autonomy - Experiment Launcher (PowerShell)
# Run this script or double-click to launch the experiment menu
# ============================================================

# Change to script directory
Set-Location $PSScriptRoot

# Use isaaclab311 Python directly (bypasses conda activation issues)
& "C:\miniconda3\envs\isaaclab311\python.exe" launch.py @args

# Keep window open if launched by double-click
if ($Host.Name -eq "ConsoleHost") {
    Write-Host ""
    Write-Host "Press any key to close..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
