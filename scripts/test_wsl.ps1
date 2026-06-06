param(
    [Parameter(Mandatory = $false)]
    [string]$Artifact = "ccproxy.wsl"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($PSVersionTable.PSEdition -ne "Core") {
    throw "The WSL harness requires PowerShell Core."
}

if (-not $IsWindows) {
    throw "The WSL harness must run on real Windows."
}

if (-not (Get-Module -ListAvailable -Name Pester)) {
    throw "Pester is required. Install it with: Install-Module Pester -Scope CurrentUser"
}

$resolvedArtifact = Resolve-Path $Artifact
$env:CCPROXY_WSL_ARTIFACT = $resolvedArtifact.Path

Invoke-Pester -Path (Join-Path $PSScriptRoot "..\tests\wsl") -Output Detailed
