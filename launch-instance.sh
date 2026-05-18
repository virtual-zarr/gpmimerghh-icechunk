#!/bin/bash

# Configuration — edit these
AMI_ID="ami-075686beab831bb7f"        # Ubuntu 24.04 LTS (us-west-2); change for your region
INSTANCE_TYPE="t3.micro"
KEY_NAME=
SECURITY_GROUP_ID=
REGION="us-west-2"

# User data script — runs on first boot
USER_DATA=$(cat <<'EOF'
#!/bin/bash
set -e

# Update system
apt-get update -y
apt-get upgrade -y

# Install Python and dependencies
apt-get install -y python3 python3-pip python3-venv curl

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Make uv available system-wide
cp /root/.local/bin/uv /usr/local/bin/uv
cp /root/.local/bin/uvx /usr/local/bin/uvx

# Also install for ubuntu user
sudo -u ubuntu bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/ubuntu/.bashrc

echo "Setup complete" >> /var/log/user-data-complete.log
EOF
)

# Launch the instance
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --user-data "$USER_DATA" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=python-uv-instance}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Launched instance: $INSTANCE_ID"
echo "Waiting for it to be running..."

# Wait until running
aws ec2 wait instance-running \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo ""
echo "Instance is running!"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP:   $PUBLIC_IP"
echo ""
echo "Connect with:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Note: wait ~60s for user data setup to finish before connecting"