#!/bin/bash
nebius config set parent-id project-e00x5b2cd62wekz2bcn77

export SUBNET_ID=$(nebius vpc subnet list \
  --format json \
  | jq -r ".items[0].metadata.id")


echo "Using subnet ID: $SUBNET_ID"

# Check if disk already exists
EXISTING_DISK=$(nebius compute disk list --format json | jq -r '.items[] | select(.metadata.name=="drug-discovery-vm-disk-1") | .metadata.id')

if [ -n "$EXISTING_DISK" ]; then
  echo "Using existing disk with ID: $EXISTING_DISK"
  export INF_VM_BOOT_DISK_ID=$EXISTING_DISK
else
  echo "Creating new disk..."
  export INF_VM_BOOT_DISK_ID=$(nebius compute disk create \
    --name drug-discovery-vm-disk-1 \
    --size-gibibytes 1027 \
    --type network_ssd \
    --source-image-family-image-family ubuntu22.04-cuda12 \
    --block-size-bytes 4096 \
    --format json | jq -r ".metadata.id")
fi

export NETWORK_INTERFACE_NAME=nova-miner-api-network-interface
export USER_DATA=$(cat <<EOF | jq -Rs '.'
#cloud-config
users:
  - name: user
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat ~/.ssh/id_ed25519.pub)
EOF
)

# Check if instance already exists
EXISTING_INSTANCE=$(nebius compute instance list --format json | jq -r '.items[] | select(.metadata.name=="drug-discovery-instance") | .metadata.id')

if [ -n "$EXISTING_INSTANCE" ]; then
  echo "Using existing instance with ID: $EXISTING_INSTANCE"
  export INF_VM_ID=$EXISTING_INSTANCE
else
  echo "Creating new instance..."
  export INF_VM_ID=$(nebius compute instance create \
    --format json \
    - <<EOF | jq -r ".metadata.id"
{
  "metadata": {
    "name": "drug-discovery-instance"
  },
  "spec": {
    "stopped": false,
    "cloud_init_user_data": $USER_DATA,
    "resources": {
      "platform": "gpu-h100-sxm",
      "preset": "1gpu-16vcpu-200gb"
    },
    "boot_disk": {
      "attach_mode": "READ_WRITE",
      "existing_disk": {
        "id": "$INF_VM_BOOT_DISK_ID"
      }
    },
    "network_interfaces": [
      {
        "name": "$NETWORK_INTERFACE_NAME",
        "subnet_id": "$SUBNET_ID",
        "ip_address": {},
        "public_ip_address": {}
      }
    ]
  }
}
EOF
  )
fi

# Wait for the instance to be ready
echo "Waiting for instance to be ready..."
sleep 30

# Get the instance details and extract the public IP address
export INF_PUBLIC_IP_ADDRESS=$(nebius compute instance get \
  --id $INF_VM_ID \
  --format json \
  | jq -r '.status.network_interfaces.[0].public_ip_address.address | split("/")[0]')

echo "Instance is ready with IP: $INF_PUBLIC_IP_ADDRESS"
echo "Connecting via SSH..."

# Add a small delay to ensure SSH is ready
sleep 1
ssh user@$INF_PUBLIC_IP_ADDRESS

