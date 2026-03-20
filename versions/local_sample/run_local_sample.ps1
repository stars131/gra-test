[CmdletBinding()]
param(
    [string]$LogsPath = ""
)

$ErrorActionPreference = "Stop"

$VersionRoot = Split-Path -Parent $PSScriptRoot
Set-Location $VersionRoot
Set-Location ..

$config = "src/config/config_local_sample.yaml"
$dataDir = "data/raw/CIC-IDS-2017-local-sample"

python download_datasets.py --dataset local_sample

if ([string]::IsNullOrWhiteSpace($LogsPath)) {
    python main.py --data_dir $dataDir --mode full --config $config
}
else {
    python main.py --data_dir $dataDir --mode full --config $config --logs_path $LogsPath
}
