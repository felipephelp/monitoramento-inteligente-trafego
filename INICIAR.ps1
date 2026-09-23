$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$labPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) {
    throw 'Ambiente não encontrado. Consulte README.md para instalar.'
}
& $labPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
