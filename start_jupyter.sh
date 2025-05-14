#!/bin/bash

# Define the command to run Jupyter Lab
command="pipenv run jupyter lab --no-browser --ip=0.0.0.0"

# Define the output log file
output_file="jupyter_out.log"

# Run the command in the background using nohup
nohup $command > $output_file 2>&1 &

