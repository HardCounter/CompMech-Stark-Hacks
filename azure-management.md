
# 1 Sanity check the CLI is on PATH
az --version

# 2 Log in (opens a browser)
az login
az account set --subscription "<your-subscription-name-or-id>"
az account show --output table

# 3 BEFORE you burn time provisioning: confirm GPU quota in your region.
#    Look for "Standard NCASv3_T4 Family" -> CurrentValue/Limit should have room for 4 vCPU.
az vm list-usage --location eastus --output table | Select-String "NC|NV"

# 4 Configure + provision
Copy-Item infra/azure/.env.example infra/azure/.env
notepad infra/azure/.env                    # fill in RG, region, VM name
./infra/azure/provision_vm.ps1

# 5 Use the SSH command printed at the end, then on the VM:
#    (laptop)   scp -i infra/azure/keys/id_rsa infra/azure/bootstrap_vm.sh azureuser@<ip>:~
#    (VM)       bash ~/bootstrap_vm.sh

# 6 When done for the day:
az vm deallocate -g <rg> -n <vm>          # stop billing (keeps disks)
# or ./infra/azure/teardown.ps1 -Purge    # nuke everything