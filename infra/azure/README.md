# Azure GPU training — hackathon runbook

End-to-end recipe to train / fine-tune **GQ-CNN 2.0** on a single-GPU Azure
VM. Target: spin a VM up, SSH in, train, pull the ONNX back to the Pi, shut
the VM down. Pay-as-you-go, no Azure ML workspace needed.

## 0. Prerequisites

- Azure subscription with **pay-as-you-go** (or MSDN / free credits)
- Quota for at least one of these SKUs in your target region:
  - `Standard_NC4as_T4_v3` (1× T4 16 GB) — cheapest, **preferred**
  - `Standard_NV6ads_A10_v5` (1× A10 24 GB) — fallback if T4 quota = 0
  - `Standard_NC6s_v3`       (1× V100 16 GB) — only if the above are unavailable
- Azure CLI 2.50+ installed locally (`winget install -e --id Microsoft.AzureCLI` on Windows)
- OpenSSH installed locally (`where.exe ssh` should return a path on Windows 10+)

Confirm quota before you burn time on provisioning:

```powershell
az vm list-usage --location eastus --output table | Select-String "NC|NV"
```

If the number under `CurrentValue/Limit` for `Standard NCASv3_T4 Family` is `0/0`,
request quota from the Azure Portal:
`Subscriptions -> Usage + quotas -> Filter: "NCAS T4 v3" -> Request Increase -> 4 vCPU`.
Approvals typically take <1 hour for small hackathon quotas.

## 1. Log in locally

```powershell
az login
az account set --subscription "<subscription-name-or-id>"
az account show --output table   # sanity check
```

## 2. Configure the provisioning script

Copy the env template and fill in your values:

```powershell
Copy-Item infra/azure/.env.example infra/azure/.env
notepad infra/azure/.env
```

Minimum required: `AZ_RG`, `AZ_REGION`, `AZ_VM_NAME`. The script reuses the
resource group if it already exists. SSH keys are auto-generated into
`infra/azure/keys/` if you don't have one.

## 3. Provision the VM

```powershell
./infra/azure/provision_vm.ps1
```

The script:
1. Creates (or reuses) the resource group.
2. Generates an SSH keypair in `infra/azure/keys/` if missing.
3. Creates a VM with the Microsoft HPC Ubuntu 22.04 image
   (`microsoft-dsvm:ubuntu-hpc:2204:latest`), which ships with CUDA +
   NVIDIA drivers pre-installed.
4. Opens port 22 to your current public IP only (auto-detected).
5. Enables auto-shutdown at 23:00 local time so a forgotten VM doesn't bleed
   credits overnight.
6. Writes the resulting public IP + ready-to-copy SSH command to
   `infra/azure/.vm_info`.

Total provisioning time: ~3–6 minutes.

## 4. SSH in + run the bootstrap

From the output of step 3:

```powershell
ssh -i infra/azure/keys/id_rsa azureuser@<public-ip>
```

Once logged in, run the bootstrap to install the project:

```bash
curl -fsSL https://raw.githubusercontent.com/<your-fork>/CompMech-Stark-Hacks/main/infra/azure/bootstrap_vm.sh | bash
# or scp the script over if the repo is private:
# (on your laptop)  scp -i infra/azure/keys/id_rsa infra/azure/bootstrap_vm.sh azureuser@<ip>:~
# (on the VM)       bash ~/bootstrap_vm.sh
```

The bootstrap:
- Installs system packages (`git`, `python3.11`, `python3.11-venv`).
- Clones the repo to `~/CompMech-Stark-Hacks`.
- Creates `~/CompMech-Stark-Hacks/.venv` and installs the project in editable
  mode with `torch` (CUDA wheel).
- Verifies `torch.cuda.is_available()`.

## 5. Connectivity smoke test (no training yet)

The dataset step (`--data-dir`) is deliberately deferred. For now we just
confirm the VM is wired correctly:

```bash
cd ~/CompMech-Stark-Hacks
source .venv/bin/activate
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

# ONNX export works on the VM too — good sanity check of the environment
python scripts/export_gqcnn_onnx.py --strategy pytorch \
    --output models/gqcnn_2.0.onnx
```

## 6. Pulling trained artefacts back

Once Phase B training actually runs:

```powershell
scp -i infra/azure/keys/id_rsa \
    azureuser@<ip>:~/CompMech-Stark-Hacks/models/gqcnn_2.0_finetuned.onnx \
    ./models/
```

## 7. Shut down when not training

**Paying-by-the-minute VMs need to be deallocated**, not just stopped:

```powershell
az vm deallocate --resource-group $env:AZ_RG --name $env:AZ_VM_NAME
```

To fully tear down (deletes the VM, disks, NIC, public IP, NSG):

```powershell
./infra/azure/teardown.ps1
```

## Cost quick-reference (pay-as-you-go, eastus, Nov 2025 list prices)

| SKU                      | $/hr   | $/day if left on |
|--------------------------|--------|------------------|
| `Standard_NC4as_T4_v3`   | ~0.53  | ~12.72           |
| `Standard_NV6ads_A10_v5` | ~0.91  | ~21.84           |
| `Standard_NC6s_v3`       | ~3.06  | ~73.44           |

Using **Azure Spot** cuts these by ~60–80%, at the cost of possible eviction
mid-job. The provisioning script has `AZ_USE_SPOT=1` to enable it.
