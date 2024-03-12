#!/bin/bash

# API Key and SSH details
LAMBDA_API_KEY="secret_api-key2_e4b54166fae54bee895300564adaf1e4.zS0nxFFon2OAqgUbAU43rAEYEY9QP0FL"
SSH_PRIVATE_KEY="/path/to/your/private/key"
SSH_KEY_NAME="x399" # Name of the SSH key in the Lambda Labs Cloud console
LOCAL_FILES_PATH="./*"
REMOTE_PATH="~/"
FILE_TO_EXECUTE="run_script.sh"

# Define the instance request payload
cat > request.json << EOF
{
  "region_name": "us-east-1",
  "instance_type_name": "gpu_1x_a10",
  "ssh_key_names": [
    "$SSH_KEY_NAME"
  ],
  "file_system_names": [],
  "quantity": 1
}
EOF

#curl -H "Authorization: Bearer $LAMBDA_API_KEY" https://cloud.lambdalabs.com/api/v1/instance-types

# Launch the instance
echo "Launching instance..."
curl -u $LAMBDA_API_KEY: https://cloud.lambdalabs.com/api/v1/instance-operations/launch -d @request.json -H "Content-Type: application/json"

LAUNCH_RESPONSE=$(curl -H "Authorization: Bearer $LAMBDA_API_KEY" https://cloud.lambdalabs.com/api/v1/instances)
echo "Launching instance - response: $LAUNCH_RESPONSE"


INSTANCE_ID=$(echo $LAUNCH_RESPONSE | jq -r '.data[0].id')
echo "Instance launched with ID: $INSTANCE_ID"

# Check instance status in a loop
STATUS="unknown"
while [ "$STATUS" != "active" ]; do
  echo "Checking instance status..."
  INSTANCE_DETAILS=$(curl -u $LAMBDA_API_KEY: https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID)
  STATUS=$(echo $INSTANCE_DETAILS | jq -r '.data.status')
  if [ "$STATUS" != "active" ]; then
    echo "Instance status: $STATUS. Waiting..."
    sleep 10
  else
    echo "Instance is active."
  fi
done

# Now that instance is active, retrieve IP
INSTANCE_IP=$(echo $INSTANCE_DETAILS | jq -r '.data.ip')
echo "Instance IP: $INSTANCE_IP"

# Copy files from local to the remote instance
scp -i $SSH_PRIVATE_KEY -o StrictHostKeyChecking=no $LOCAL_FILES_PATH ubuntu@$INSTANCE_IP:$REMOTE_PATH

# Run a specific file on the remote instance
ssh -i $SSH_PRIVATE_KEY -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "bash -c 'cd $REMOTE_PATH && ./$FILE_TO_EXECUTE'"

# Reminder to terminate the instance
#echo "Script executed. Please manually terminate the instance using the Lambda Labs Cloud console or API."
