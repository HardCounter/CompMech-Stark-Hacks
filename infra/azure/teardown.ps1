<#
.SYNOPSIS
    Deallocate or fully delete the Azure GPU VM.

.DESCRIPTION
    Reads infra/azure/.env and, by default, only deallocates the VM (stops
    billing but keeps disks). Use -Purge to delete the whole resource group.

.EXAMPLE
    PS> ./infra/azure/teardown.ps1           # deallocate only
    PS> ./infra/azure/teardown.ps1 -Purge    # delete the resource group
#>

[CmdletBinding()]
param(
    [switch]$Purge
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile   = Join-Path $ScriptDir '.env'

if (-not (Test-Path $EnvFile)) {
    Write-Error "Missing $EnvFile."
}

$cfg = @{}
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#')) {
        $idx = $line.IndexOf('=')
        if ($idx -gt 0) {
            $cfg[$line.Substring(0, $idx).Trim()] = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
        }
    }
}

$RG = $cfg['AZ_RG']
$VM = $cfg['AZ_VM_NAME']

if ($Purge) {
    Write-Host "==> PURGE: deleting entire resource group $RG (this removes ALL resources in it)..."
    $confirm = Read-Host "Type the RG name '$RG' to confirm"
    if ($confirm -ne $RG) {
        Write-Error "Confirmation mismatch; aborted."
    }
    az group delete --name $RG --yes --no-wait
    Write-Host "    deletion kicked off in background."
} else {
    Write-Host "==> Deallocating VM $VM in $RG ..."
    az vm deallocate --resource-group $RG --name $VM
    Write-Host "    VM deallocated. Disks + public IP remain (still billed for storage)."
    Write-Host "    Restart:   az vm start -g $RG -n $VM"
    Write-Host "    Purge all: ./infra/azure/teardown.ps1 -Purge"
}
