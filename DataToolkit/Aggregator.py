import os
import panoptes_aggregation
import subprocess

# https://aggregation-caesar.zooniverse.org/README.html

class Aggregator:
    def __init__(self, classifications_csv_filename, workflow_csv_filename, config_directory="Config", extractions_directory="Extractions", reductions_directory="Reductions"):
        """
        Initializes a Classifier object. This object is used to generate config files,
        extracted files, and reduction files of a workflow using the Panoptes aggregation CLI.

        Parameters
        ----------
            classifications_csv_filename : str
                The name of the classifications csv file.
            workflow_csv_filename : str
                The name of the workflow csv file.
            config_directory : str, optional
                The name of the directory where the config files will be stored.
                The default is "Config".
            extractions_directory : str, optional
                The name of the directory where the extracted files will be stored.
                The default is "Extractions".
            reductions_directory : str, optional
                The name of the directory where the reduction files will be stored.
                The default is "Reductions".

        Notes
        -----
        The config files are used to generate the extracted files, which are used to generate the reduction files.
        """

        # Initialize variables

        # If the classifications csv filename is a relative path, make it an absolute path based on the current working directory
        if(not os.path.isabs(classifications_csv_filename)):
            classifications_csv_filename = os.path.join(os.getcwd(), classifications_csv_filename)

        # If the workflow csv filename is a relative path, make it an absolute path based on the current working directory
        if(not os.path.isabs(workflow_csv_filename)):
            workflow_csv_filename = os.path.join(os.getcwd(), workflow_csv_filename)

        self.classifications_csv_filename = classifications_csv_filename
        self.workflow_csv_filename = workflow_csv_filename
        
        # If the config directory is a relative path, make it an absolute path based on the current working directory
        if(not os.path.isabs(config_directory)):
            config_directory = os.path.join(os.getcwd(), config_directory)
            
        # If the extraction directory is a relative path, make it an absolute path based on the current working directory
        if(not os.path.isabs(extractions_directory)):
            extractions_directory = os.path.join(os.getcwd(), extractions_directory)
        
        # If the reductions directory is a relative path, make it an absolute path based on the current working directory
        if(not os.path.isabs(reductions_directory)):
            reductions_directory = os.path.join(os.getcwd(), reductions_directory)
        
        self.config_directory = config_directory
        self.extractions_directory = extractions_directory
        self.reductions_directory = reductions_directory
        self.extractor_config_file = None
        self.reducer_config_file = None
        self.task_config_file = None
        self.workflow_id = None
        self.extracted_file = None
        self.reduced_file = None

    def aggregateWorkflow(self, workflow_id, **kwargs):
        """
        Classifies a workflow using the Panoptes aggregation client by generating config files,
        extracted files, and reduction files.

        Parameters
        ----------
            workflow_id : int
                The ID of the workflow to be aggregated.
            **kwargs : optional
                Optional arguments to be passed to the config command.
        """

        # Generate config files, extracted files, and reduction files
        self.config(workflow_id, **kwargs)
        self.extract()
        self.reduce()

        # Print the results
        print(f"Classifications complete.")
        print(f"Extracted file: {self.extracted_file}")
        print(f"Reduced file: {self.reduced_file}")

    def config(self, workflow_id, **kwargs):
        """
        Generates config files for a workflow using the Panoptes aggregation CLI.

        Parameters
        ----------
            workflow_id : int
                The ID of the workflow to be aggregated.
            **kwargs : optional
                Optional arguments to be passed to the config command.
        """

        # Store the workflow ID
        self.workflow_id = workflow_id

        if(self.workflow_id is None):
            raise ValueError("Workflow ID has not been set.")

        if(not os.path.exists(self.workflow_csv_filename)):
            raise FileNotFoundError(f"Workflow file {self.workflow_csv_filename} does not exist.")

        # Create the config directory if it does not exist
        if(not os.path.exists(self.config_directory)):
            os.mkdir(self.config_directory)

        # Define the command you want to run
        # TODO: Fix the command so that it knows what current working directory it should be running in.

        command = f"panoptes_aggregation config {self.workflow_csv_filename} {workflow_id} -d {self.config_directory}"

        # Construct the command string with optional arguments
        command_str = command

        # Define the allowed keys for the optional arguments of the config command
        single_dash_keys = ["v", "vv", "k", "h"]
        double_dash_keys = ["version", "help", "min_version", "max_version", "keywords", "verbose"]
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
        output = subprocess.check_output(command_str, cwd=os.getcwd())

        # Decode the output assuming it's in UTF-8 encoding
        decoded_output = output.decode("utf-8")

        # Print the output
        print(decoded_output)

        # Extract the files from the output
        split_output = decoded_output.split("\n")[1:-1]

        # Initialize the files list
        files = []
        for line in split_output:
            possible_file = line.removesuffix("\r")
            # If the possible file exists
            if(os.path.exists(possible_file)):
                files.append(possible_file)

        # Store the config files in the appropriate variables
        for file in files:
            if("Extractor" in file):
                self.extractor_config_file = file
            elif("Reducer" in file):
                self.reducer_config_file = file
            elif("Task" in file):
                self.task_config_file = file

    def extract(self, **kwargs):
        """
        Generates an extracted file using the Panoptes aggregation CLI.

        Parameters
        ----------
            **kwargs : optional
                Optional arguments to be passed to the extract command.
        """

        # Check that the extractor config file is defined
        if(self.extractor_config_file is None):
            raise ValueError("Extractor file is not defined. Please run config() first.")

        if(not os.path.exists(self.classifications_csv_filename)):
            raise ValueError(f"Classifications file {self.classifications_csv_filename} does not exist.")

        if(not os.path.exists(self.extractor_config_file)):
            raise ValueError(f"Extractor config file {self.extractor_config_file} does not exist.")

        # Create the extraction directory if it does not exist
        if(not os.path.exists(self.extractions_directory)):
            os.mkdir(self.extractions_directory)

        # Define the command you want to run
        command = f"panoptes_aggregation extract {self.classifications_csv_filename} {self.extractor_config_file} -d {self.extractions_directory}"

        # Construct the command string with optional arguments
        command_str = command

        # Set the default output file name to be a modified version of the config file name
        if(kwargs.get("o") is None and kwargs.get("output") is None):
            kwargs["o"] = self.extractor_config_file.split("_config_")[1]
            kwargs["output"] = self.extractor_config_file.split("_config_")[1]

        # Define the allowed keys for the optional arguments of the extract command
        single_dash_keys = ["o", "O", "c", "vv", "h"]
        double_dash_keys = ["output", "help", "order", "cpu_count", "verbose"]
        allowed_keys = single_dash_keys + double_dash_keys

        # Check that the kwargs are valid and add them to the command string
        for key, value in kwargs.items():
            if (key not in allowed_keys):
                raise ValueError(f"Invalid argument: {key}")
            if (key in single_dash_keys):
                if (value is not None):
                    command_str += f" -{key} {value}"
                else:
                    command_str += f" -{key}"
            elif (key in double_dash_keys):
                if (value is not None):
                    command_str += f" --{key} {value}"
                else:
                    command_str += f" --{key}"

        # Run the command and capture the output
        output = subprocess.check_output(command_str, cwd=os.getcwd())

        # Decode the output assuming it's in UTF-8 encoding
        decoded_output = output.decode("utf-8")

        # Print the output
        print(decoded_output)

        # Save the filename of the extracted file
        if(kwargs.get("o") is not None):
            self.extracted_file = os.path.join(self.extractions_directory, "question_extractor_" + kwargs.get("o").removesuffix(".yaml") + ".csv")
        else:
            self.extracted_file = os.path.join(self.extractions_directory, "question_extractor_" + kwargs.get("output").removesuffix(".yaml") + ".csv")

    def reduce(self, **kwargs):
        """
        Generates a reduced file using the Panoptes aggregation CLI.

        Parameters
        ----------
            **kwargs : optional
                Optional arguments to be passed to the reduce command.
        """

        # Check that the reducer config file is defined
        if(self.reducer_config_file is None):
            raise ValueError("Extractor file is not defined. Please run config() first.")

        # Check that the extracted file is defined
        if(self.extracted_file is None):
            raise ValueError("Extracted file is not defined. Please run extract() first.")

        if(not os.path.exists(self.extracted_file)):
            raise ValueError(f"Extracted file {self.extracted_file} does not exist.")

        if(not os.path.exists(self.reducer_config_file)):
            raise ValueError(f"Reducer config file {self.reducer_config_file} does not exist.")

        # Create the reductions directory if it does not exist
        if(not os.path.exists(self.reductions_directory)):
            os.mkdir(self.reductions_directory)

        # Define the command you want to run
        command = f"panoptes_aggregation reduce {self.extracted_file} {self.reducer_config_file} -d {self.reductions_directory} -s"

        # Construct the command string with optional arguments
        command_str = command

        # Set the default output file name to be a modified version of the extracted file name
        if(kwargs.get("o") is None and kwargs.get("output") is None):
            head, tail = os.path.split(self.extracted_file)
            extracted_filename = tail.removeprefix("question_extractor_")
            kwargs["o"] = extracted_filename
            kwargs["output"] = extracted_filename

        # Define the allowed keys for the optional arguments of the reduce command
        single_dash_keys = ["o", "O", "F", "c", "h"]
        double_dash_keys = ["output", "help", "order", "cpu_count"]
        allowed_keys = single_dash_keys + double_dash_keys

        # Check that the kwargs are valid and add them to the command string
        for key, value in kwargs.items():
            if (key not in allowed_keys):
                raise ValueError(f"Invalid argument: {key}")
            if (key in single_dash_keys):
                if (value is not None):
                    command_str += f" -{key} {value}"
                else:
                    command_str += f" -{key}"
            elif (key in double_dash_keys):
                if (value is not None):
                    command_str += f" --{key} {value}"
                else:
                    command_str += f" --{key}"

        # Run the command and capture the output
        output = subprocess.check_output(command_str, cwd=os.getcwd())

        # Decode the output assuming it's in UTF-8 encoding
        decoded_output = output.decode("utf-8")

        # Print the output
        print(decoded_output)

        # Save the filename of the reduced file
        if (kwargs.get("o") is not None):
            self.reduced_file = os.path.join(self.reductions_directory, "question_reducer_" + kwargs.get("o").removesuffix(".csv") + ".csv")
        else:
            self.reduced_file = os.path.join(self.reductions_directory, "question_reducer_" + kwargs.get("output").removesuffix(".csv") + ".csv")