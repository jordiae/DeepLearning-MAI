#!/bin/bash

DATA_CLUSTER=dt01.bsc.es

read -p 'Username: ' USER

# Deploy code and data
RSYNC="rsync -a -e \"ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no\""
DEPLOY_ADDRESS=${USER}@${DATA_CLUSTER}
DEPLOY_DIR=/home/nct01/$USER/

echo "$RSYNC pretrained_models experiments src ${DEPLOY_ADDRESS}:${DEPLOY_DIR}"
bash -c "$RSYNC pretrained_models experiments src ${DEPLOY_ADDRESS}:${DEPLOY_DIR}"

if [ $? -ne 0 ]
then
  echo "rsync error to ${DEPLOY_ADDRESS}"
  exit 1
fi