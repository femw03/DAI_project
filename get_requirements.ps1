
# PowerShell script (extract_requirements.ps1)

# Check if virtual environment is active
if (-not $env:VIRTUAL_ENV) {
    Write-Output "No virtual environment is currently active."
    Write-Output "Please activate your virtual environment and run this script again."
    exit 1
}

# Extract requirements
pip freeze | Out-File -FilePath requirements.txt -Encoding utf8

Write-Output "Requirements have been extracted to requirements.txt"