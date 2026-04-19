<#
.SYNOPSIS
    Provision a single-GPU Azure VM for gripper_cv training.

.DESCRIPTION
    Reads infra/azure/.env, creates (or reuses) a resource group, generates an
    SSH keypair if needed, and creates a VM using the Microsoft HPC Ubuntu
    22.04 image (CUDA + NVIDIA drivers preinstalled). Locks SSH to the current
    public IP and enables auto-shutdown.

.NOTES
    Run from the repo root:  ./infra/azure/provision_vm.ps1

.EXAMPLE
    PS> ./infra/azure/provision_vm.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipKeygen,
    [switch]$ForceRecreate
)

$ErrorActionPreference = 'Stop'

# ---- Locate env file --------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile   = Join-Path $ScriptDir '.env'
$KeyDir    = Join-Path $ScriptDir 'keys'
$KeyPath   = Join-Path $KeyDir 'id_rsa'
$VmInfoOut = Join-Path $ScriptDir '.vm_info'

if (-not (Test-Path $EnvFile)) {
    Write-Error "Missing $EnvFile. Copy .env.example to .env and fill it in."
}

# ---- Parse .env -------------------------------------------------------------
$cfg = @{}
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#')) {
        $idx = $line.IndexOf('=')
        if ($idx -gt 0) {
            $key = $line.Substring(0, $idx).Trim()
            $val = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
            $cfg[$key] = $val
        }
    }
}

$required = @('AZ_RG', 'AZ_REGION', 'AZ_VM_NAME')
foreach ($k in $required) {
    if (-not $cfg.ContainsKey($k) -or [string]::IsNullOrWhiteSpace($cfg[$k])) {
        Write-Error "Missing required config: $k (set it in $EnvFile)"
    }
}

$RG         = $cfg['AZ_RG']
$REGION     = $cfg['AZ_REGION']
$VM         = $cfg['AZ_VM_NAME']
$SIZE       = if ($cfg['AZ_VM_SIZE'])  { $cfg['AZ_VM_SIZE'] }  else { 'Standard_NC4as_T4_v3' }
$IMAGE      = if ($cfg['AZ_IMAGE'])    { $cfg['AZ_IMAGE'] }    else { 'microsoft-dsvm:ubuntu-hpc:2204:latest' }
$ADMIN      = if ($cfg['AZ_ADMIN_USER']) { $cfg['AZ_ADMIN_USER'] } else { 'azureuser' }
$AUTOSTOP   = $cfg['AZ_AUTOSHUTDOWN_TIME']
$USE_SPOT   = $cfg['AZ_USE_SPOT'] -eq '1'
$SPOT_MAX   = if ($cfg['AZ_SPOT_MAX_PRICE']) { $cfg['AZ_SPOT_MAX_PRICE'] } else { '-1' }

# ---- Az CLI sanity ----------------------------------------------------------
Write-Host "==> Checking Azure CLI login..."
$acct = az account show --output json 2>$null | ConvertFrom-Json
if (-not $acct) {
    Write-Error "Not logged in. Run 'az login' first."
}
Write-Host "    subscription: $($acct.name) ($($acct.id))"

# ---- SSH key ---------------------------------------------------------------
if (-not (Test-Path $KeyDir)) { New-Item -ItemType Directory -Path $KeyDir | Out-Null }

if (-not $SkipKeygen -and -not (Test-Path $KeyPath)) {
    Write-Host "==> Generating SSH keypair at $KeyPath ..."
    ssh-keygen -t rsa -b 4096 -N '""' -f $KeyPath -q | Out-Null
}
$SshPub = Get-Content "$KeyPath.pub" -Raw

# ---- Detect current public IP for NSG lockdown ------------------------------
Write-Host "==> Detecting your public IP for SSH allow-list..."
try {
    $MyIp = (Invoke-WebRequest -UseBasicParsing -Uri 'https://api.ipify.org').Content.Trim()
    Write-Host "    your IP: $MyIp"
} catch {
    Write-Warning "Could not detect public IP; SSH will be open to 0.0.0.0/0. Tighten later."
    $MyIp = '*'
}

# ---- Resource group ---------------------------------------------------------
$rgExists = az group exists --name $RG
if ($rgExists -eq 'false') {
    Write-Host "==> Creating resource group $RG in $REGION..."
    az group create --name $RG --location $REGION --output none
} else {
    Write-Host "==> Reusing existing resource group $RG"
}

# ---- Teardown if forced -----------------------------------------------------
if ($ForceRecreate) {
    Write-Host "==> --ForceRecreate: deleting existing VM if present..."
    az vm delete --resource-group $RG --name $VM --yes --output none 2>$null
}

# ---- Check if VM already exists --------------------------------------------
$existing = az vm show --resource-group $RG --name $VM --output json 2>$null | ConvertFrom-Json
if ($existing) {
    Write-Host "==> VM $VM already exists in $RG (skipping create). Use -ForceRecreate to redo."
} else {
    Write-Host "==> Creating VM $VM ($SIZE) in $RG/$REGION..."
    $createArgs = @(
        'vm', 'create',
        '--resource-group', $RG,
        '--name', $VM,
        '--image', $IMAGE,
        '--size', $SIZE,
        '--admin-username', $ADMIN,
        '--ssh-key-values', "$KeyPath.pub",
        '--public-ip-sku', 'Standard',
        '--os-disk-size-gb', '128',
        '--storage-sku', 'Premium_LRS',
        '--output', 'json'
    )
    if ($USE_SPOT) {
        Write-Host "    (spot instance enabled, max price: $SPOT_MAX)"
        $createArgs += @('--priority', 'Spot', '--eviction-policy', 'Deallocate', '--max-price', $SPOT_MAX)
    }
    $createOut = az @createArgs | ConvertFrom-Json
    if (-not $createOut) { Write-Error "VM creation failed." }
}

# ---- Retrieve IP -----------------------------------------------------------
$ip = az vm list-ip-addresses --resource-group $RG --name $VM --output json | ConvertFrom-Json
$PublicIp = $ip[0].virtualMachine.network.publicIpAddresses[0].ipAddress
if (-not $PublicIp) { Write-Error "Could not resolve VM public IP." }

# ---- NSG rule: SSH locked to your IP ---------------------------------------
Write-Host "==> Locking SSH allow-list to $MyIp ..."
$nsg = "${VM}NSG"
az network nsg rule update `
    --resource-group $RG `
    --nsg-name $nsg `
    --name default-allow-ssh `
    --source-address-prefixes $MyIp `
    --output none 2>$null | Out-Null

# ---- Auto-shutdown ---------------------------------------------------------
if ($AUTOSTOP) {
    Write-Host "==> Enabling auto-shutdown at $AUTOSTOP ..."
    $vmId = az vm show --resource-group $RG --name $VM --query id -o tsv
    az vm auto-shutdown --resource-group $RG --name $VM --time $AUTOSTOP --output none
}

# ---- Write vm info ---------------------------------------------------------
$info = [ordered]@{
    resource_group = $RG
    region         = $REGION
    vm_name        = $VM
    vm_size        = $SIZE
    image          = $IMAGE
    admin_user     = $ADMIN
    public_ip      = $PublicIp
    ssh_command    = "ssh -i $KeyPath $ADMIN@$PublicIp"
    scp_prefix     = "scp -i $KeyPath"
    autoshutdown   = $AUTOSTOP
    provisioned_at = (Get-Date).ToString('o')
}
($info | ConvertTo-Json) | Set-Content -Path $VmInfoOut

Write-Host ''
Write-Host '========================================================'
Write-Host "VM ready."
Write-Host "  public IP  : $PublicIp"
Write-Host "  SSH        : ssh -i $KeyPath $ADMIN@$PublicIp"
Write-Host "  upload     : scp -i $KeyPath infra/azure/bootstrap_vm.sh ${ADMIN}@${PublicIp}:~"
Write-Host "  then on VM : bash ~/bootstrap_vm.sh"
Write-Host "  deallocate : az vm deallocate -g $RG -n $VM"
Write-Host "  teardown   : ./infra/azure/teardown.ps1"
Write-Host '========================================================'
