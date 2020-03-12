#!/bin/bash
if [ $# -lt 2 ]
then
  echo "Usage: ./execute_experiment.sh experiment_folder debug|main"
  exit 1
fi

# Variables
EXPERIMENT_FOLDER=$1
MODE=$2
DATA_CLUSTER=dt01.bsc.es
LOGIN_CLUSTER=plogin1.bsc.es

read -p 'Username: ' USER

# Deploy code and data
RSYNC="rsync -a -e \"ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no\""
DEPLOY_ADDRESS=${USER}@${DATA_CLUSTER}
DEPLOY_DIR=/home/nct01/$USER/

echo "$RSYNC data experiments src venv ${DEPLOY_ADDRESS}:${DEPLOY_DIR}"
bash -c "$RSYNC data experiments src venv ${DEPLOY_ADDRESS}:${DEPLOY_DIR}"

if [ $? -ne 0 ]
then
  echo "rsync error to ${DEPLOY_ADDRESS}"
  exit 1
fi

# Run experiment
SSH="ssh -l root -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"
EXECUTION_ADDRESS=${USER}@${LOGIN_CLUSTER}
EXECUTION_DIR=/home/nct01/$USER/$EXPERIMENT_FOLDER

if [ $MODE == "debug" ]
then
  $SSH $EXECUTION_ADDRESS "cd $EXECUTION_DIR/; export PYTHONPATH=\"${PYTHONPATH}:/home/nct01/$USER/src/\"; ./debug_launcher.sh"
elif [ $MODE == "main" ]
then
  $SSH $EXECUTION_ADDRESS "cd $EXECUTION_DIR/; export PYTHONPATH=\"${PYTHONPATH}:/home/nct01/$USER/src/\"; ./main_launcher.sh"
else
  echo "execution mode not valid"
  exit 1
fi

exit ${?}
