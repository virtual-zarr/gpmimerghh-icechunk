
# WIP Design + Infra to create an Icechunk Store for GPM IMERG HH 07


## Local development

```sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Launch instance

```sh
./launch-instance.sh
```

Once instance is launched, add to ~/.ssh/config

```sh
Host my-ec2
  HostName <new_host_ip>
  User ubuntu
  IdentityFile <your_pem_file>.pem
```

Install Remote-SSH VS Code plugin.

Run Remote-SSH: Connect to Host...

rsync files (run locally):

```sh
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  -e "ssh -i <your_pem_file>.pem" \
  ~/aimee-os/Projects/gpm_imerg_hh_icechunk ubuntu@<new_host_ip>:~/
```