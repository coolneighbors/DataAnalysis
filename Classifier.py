import argparse

import panoptes_aggregation
import subprocess
# https://aggregation-caesar.zooniverse.org/README.html

class Classifier:
    def __init__(self, classifications_csv_filename, workflow_csv_filename, config_directory="Config", extraction_directory="Extractions", reductions_directory="Reductions"):
        self.classifications_csv_filename = classifications_csv_filename
        self.workflow_csv_filename = workflow_csv_filename
        self.config_directory = config_directory
        self.extraction_directory = extraction_directory
        self.reductions_directory = reductions_directory

    def config(self, workflow_id, **kwargs):

        # Define the command you want to run
        command = f"panoptes_aggregation config {self.workflow_csv_filename} {workflow_id}"

        # Construct the command string with optional arguments
        command_str = command

        single_dash_keys = ["d", "v", "vv", "k", "h"]
        double_dash_keys = ["version", "help", "min_version", "max_version", "keywords", "dir", "verbose"]
        allowed_keys = single_dash_keys + double_dash_keys

        # Check that the kwargs are valid and add them to the command string
        for key, value in kwargs.items():
            if(key not in allowed_keys):
                raise ValueError(f"Invalid argument: {key}")
            if(key in single_dash_keys):
                if(value is not None):
                    command_str += f" -{key} {value}"
                else:
                    command_str += f" -{key}"
            elif(key in double_dash_keys):
                if(value is not None):
                    command_str += f" --{key} {value}"
                else:
                    command_str += f" --{key}"

        # Run the command and capture the output
        output = subprocess.check_output(command_str, shell=True)

        # Decode the output assuming it's in UTF-8 encoding
        decoded_output = output.decode("utf-8")

        # Print the output
        print(decoded_output)
