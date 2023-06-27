import csv
import pickle

from unWISE_verse import Data
from unWISE_verse.Spout import Spout


class Discriminator:
    def __init__(self, metadata_list):
        """
        Initializes a Discriminator object. This object is used to filter metadata objects based on a functional condition.

        Parameters
        ----------
            metadata_list : list
                A list of Metadata objects.
        """

        # Initialize the metadata list.
        self.metadata_list = metadata_list

    def findValidMetadata(self, functional_condition, *field_names):
        """
        Finds all metadata objects that satisfy the functional condition.

        Parameters
        ----------
            functional_condition : function
                A function that takes in the values of the fields that you want to filter by and returns a boolean value.
            field_names : tuple, strings, optional
                The names of the fields that you want to filter by.

        Returns
        -------
            valid_metadata_list : list
                A list of Metadata objects that satisfy the functional condition.
        Notes
        -----
            The functional condition must be a function that takes in the values of the fields that you want to filter by
            and returns a boolean value. The functional condition must return a boolean value. If the functional condition
            returns None, an exception will be raised.
        """

        # Initialize the list of valid metadata objects.
        valid_metadata_list = []

        # Iterate through the metadata objects in the metadata list.
        for metadata in self.metadata_list:
            # Check if the metadata object satisfies the functional condition.
            if self.isValid(metadata, functional_condition, *field_names):
                # If it does, add it to the list of valid metadata objects.
                valid_metadata_list.append(metadata)

        # Return the list of valid metadata objects.
        return valid_metadata_list

    def isValid(self, metadata, functional_condition, *field_names):
        """
        Determines whether or not a metadata object satisfies the functional condition.

        Parameters
        ----------
            metadata : Metadata
                The metadata object that you want to check.
            functional_condition : function
                A function that takes in the values of the fields that you want to filter by and returns a boolean value.
            field_names : tuple, strings, optional
                The names of the fields that you want to filter by.

        Returns
        -------
            result : bool
                True if the metadata object satisfies the functional condition. False otherwise.
        Notes
        -----
        Keep in mind that when writing the functional_condition, the input values will all be strings.
        You will need to convert them to the appropriate data type within the functional condition
        (since this information is not known beforehand in the Metadata object) if it is desirable to do so.

        The inputs to the functional condition must be in the same order as the field names provided otherwise
        you may get unexpected results.
        """

        # Check if the metadata object has all of the fields that you want to filter by and return False if it does not.
        for field_name in field_names:
            if not metadata.hasField(field_name):
                return False

        # Initialize the list of field values which will be associated with the field names.
        field_name_value_list = []

        # Iterate through the field names.
        for field_name in field_names:
            # Get the value of the field.
            field_name_value = metadata.getFieldValue(field_name)

            # Add the field value to the list of field values.
            field_name_value_list.append(field_name_value)

        # Convert the list of field values to a tuple.
        field_name_value_tuple = tuple(field_name_value_list)

        # Apply the functional condition to the field values and return the result.
        result = functional_condition(*field_name_value_tuple)

        # Check if the result is None and raise an exception if it is.
        if(result is None):
            raise Exception("The functional condition returned None. This is not allowed.")

        # Return the result.
        return result

    @staticmethod
    def saveResult(list, filename):
        """
        Saves a list of objects to a file.

        Parameters
        ----------
            list : list
                The list of objects that you want to save.
            filename : str
                The name of the file that you want to save the list to.
        """

        # Open the file and save the list to it.
        with open(filename, 'wb') as file:
            pickle.dump(list, file)

    @staticmethod
    def loadResult(filename):
        """
        Loads a list of objects from a file.

        Parameters
        ----------
            filename : str
                The name of the file that you want to load the list from.

        Returns
        -------
            list : list
                The list of objects that you loaded from the file.
        """

        # Open the file and load the list from it.
        with open(filename, 'rb') as file:
            list = pickle.load(file)
            return list


class SubjectDiscriminator(Discriminator):
    def __init__(self, subject_set):
        """
        Initializes a SubjectDiscriminator object. This object is used to filter subject objects (from a subject set)
        based on a functional condition.

        Parameters
        ----------
            subject_set : SubjectSet
                The subject set that you want to filter subjects from.

        Notes
        -----
            The subject set must be a subject set that you have already created in the Zooniverse project.
        """

        # Initialize the subject list.
        self.subject_list = []

        # Iterate through the subjects in the subject set and add them to the subject list.
        for index, subject in enumerate(subject_set.subjects):
            # Print the number of completed subjects every 1000 subjects.
            if (index % 1000 == 0 and index != 0):
                # Print the number of completed subjects
                print(f"Added {index} subjects to list.")

            # Add the subject to the subject list.
            self.subject_list.append(subject)

        # Initialize the metadata list.
        self.metadata_list = []

        # Print the length of the retrieved subject list.
        print(f"Length of Subject List: {len(self.subject_list)}")

        # Iterate through the subjects in the subject list and add their metadata to the metadata list.
        for index, subject in enumerate(self.subject_list):
            # Get the metadata of the subject.
            subject_metadata = self.getSubjectMetadata(subject)

            # Add the index of the subject to the metadata.
            subject_metadata.addField("index", index)

            # Add the metadata to the metadata list.
            self.metadata_list.append(subject_metadata)

        super().__init__(self.metadata_list)

    def getSubjectMetadata(self, subject):
        """
        Gets the metadata of a subject.

        Parameters
        ----------
            subject : Subject
                The subject that you want to get the metadata of.

        Returns
        -------
            metadata : Metadata
                The metadata of the subject.
        """

        # Create a metadata object from the subject's metadata dictionary.
        metadata = Data.Metadata.createFromDictionary(subject.metadata)
        return metadata

    def findValidSubjects(self, functional_condition, *field_names, subject_as_input=False, display_printouts=True):
        """
        Finds all subjects that satisfy the functional condition.

        Parameters
        ----------
            functional_condition : function
                A function that takes in the values of the fields that you want to filter by and returns a boolean value.
            field_names : tuple, strings, optional
                The names of the fields that you want to filter by.
            subject_as_input : bool, optional
                True if you want to pass in the subject object as an input to the functional condition. False otherwise.
            display_printouts : bool, optional
                True if you want to display printouts. False otherwise.

        Returns
        -------
            valid_subject_list : list
                A list of subjects that satisfy the functional condition.
        """

        # Initialize the list of valid subjects.
        valid_subject_list = []

        # Check if the subject is going to be the input to the functional condition.
        if(subject_as_input):
            # Check if the user has passed in any field names.
            if(len(field_names) != 0):
                # Display a warning message if the user has passed in field names but has also requested the subject as input.
                print("WARNING: You have requested a subject as input, but you have also passed in metadata field names. The metadata field names will be ignored.")

            # Iterate through the subjects in the subject list.
            for index, subject in enumerate(self.subject_list):
                # Get the result of the functional condition with the subject as input.
                result = functional_condition(subject)

                # Check if the result is None and raise an exception if it is.
                if(result is None):
                    raise Exception("The functional condition returned None. This is not allowed.")

                # Check if the result is True and add the subject to the list of valid subjects if it is.
                if(result):
                    # Add the subject to the list of valid subjects.
                    valid_subject_list.append(subject)

                # Display a printout every 1000 subjects.
                if (display_printouts and index % 1000 == 0 and index != 0):
                    print(f"Checked {index} subjects.")
        else:
            # Iterate through the metadata in the metadata list.
            for index, metadata in enumerate(self.metadata_list):
                # Check if the metadata is valid.
                if self.isValid(metadata, functional_condition, *field_names):
                    # Add the subject to the list of valid subjects.
                    valid_subject_list.append(self.subject_list[metadata.getFieldValue("index")])

                # Display a printout every 1000 subjects.
                if(display_printouts and index % 1000 == 0 and index != 0):
                    print(f"Checked {index} subjects.")

        # Return the list of valid subjects.
        return valid_subject_list

    @staticmethod
    def saveResult(subject_list, filename):
        """
        Saves a list of subject ids to a file.

        Parameters
        ----------
            subject_list : list
                The list of subjects that you want to save.
            filename : str
                The name of the file that you want to save the list to.
        """

        # Initialize the list of subject ids.
        subject_id_list = []

        # Iterate through the subjects in the subject list
        for subject in subject_list:
            # Add the subject id to the list of subject ids.
            subject_id_list.append(subject.id)

        # Create the file.
        with open(filename, 'wb') as file:
            # Save the list of subject ids to the file.
            pickle.dump(subject_id_list, file)

    @staticmethod
    def loadResult(filename):
        """
        Loads a list of subject ids from a file and returns the associated list of subject objects.

        Parameters
        ----------
            filename : str
                The name of the file that you want to load the list from.

        Returns
        -------
            subject_list : list
                The list of subjects that you loaded from the file.
        """

        # Open the file.
        with open(filename, 'rb') as file:
            # Load the list of subject ids from the file.
            list = pickle.load(file)

            # Initialize the list of subjects.
            subject_list = []

            # Iterate through the subject ids in the list.
            for index, subject_id in enumerate(list):
                # For every 10% of the list, print out the number of subjects that have been loaded.
                if(index % min(int(len(list)/10), 1000) == 0 and index != 0):
                    print(f"Loaded {index} subjects.")

                # Get the subject object from the subject id and add it to the list of subjects.
                subject_list.append(Spout.get_subject(subject_id))

            # Return the list of subjects.
            return subject_list

    @staticmethod
    def convertToCSV(subject_list, filename):
        """
        Converts a list of subjects to a CSV file.

        Parameters
        ----------
            subject_list : list
                The list of subjects that you want to convert to a CSV file.
            filename : str
                The name of the CSV file that you want to save the list to.

        Notes
        -----
            The CSV file will also contain all of the fields in the subject objects' metadata dictionary.
        """

        # Create the CSV file.
        with open(filename, mode="w", newline='') as csv_file:
            # Get the field names associated with the subject objects and their metadata.
            fieldnames = subject_list[0].__dict__.keys() | subject_list[0].metadata.keys()

            # Create the CSV writer with the field names.
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write the header row to the CSV file.
            csv_writer.writeheader()

            # Iterate through the subjects in the subject list.
            for subject in subject_list:
                # Create a row dictionary that contains the subject object's fields and metadata.
                row_dict = subject.__dict__ | subject.metadata

                # Write the row dictionary to the CSV file.
                csv_writer.writerow(row_dict)


class CSVDiscriminator(Discriminator):
    def __init__(self, csv_path):
        """
        Creates a CSV discriminator. This discriminator will load a CSV file and use it to filter the rows.

        Parameters
        ----------
            csv_path : str
                The path to the CSV file that you want to load.
        """

        # Initialize the csv path and metadata list.
        self.csv_path = csv_path
        self.metadata_list = []

        # Open the CSV file.
        with open(csv_path, "r", newline='') as csv_file:
            # Create a CSV reader.
            csv_reader = csv.DictReader(csv_file)

            # Iterate through the rows in the CSV file.
            for index, row in enumerate(csv_reader):
                # Create a metadata object from the row.
                metadata = Data.Metadata.createFromDictionary(row)

                # Add the metadata object to the metadata list.
                self.metadata_list.append(metadata)

        super().__init__(self.metadata_list)


class SubjectCSVDiscriminator(CSVDiscriminator):
    def __init__(self, subject_set, csv_path):
        """
        Creates a subject CSV discriminator. This discriminator will load a CSV file and use it to filter the rows,
        where each row corresponds to a subject.

        Parameters
        ----------
            subject_set : panoptes_client.SubjectSet
                The subject set that you want to filter by.
            csv_path : str
                The path to the CSV file that you want to load, where each row corresponds to a subject.
        """

        # Initialize the subject set.
        self.subject_set = subject_set

        # Reduce the CSV file to only contain rows that correspond to subjects in the subject set.
        reduced_csv_path = self.reduceCSV(csv_path)

        # Expand the CSV file to contain all of the metadata for each subject.
        expanded_csv_path = self.expandCSV(reduced_csv_path)

        super().__init__(expanded_csv_path)

    def reduceCSV(self, csv_path):
        """
        Reduces a CSV file to only contain rows that correspond to subjects in the subject set.

        Parameters
        ----------
            csv_path : str
                The path to the CSV file that you want to reduce.

        Returns
        -------
            reduced_csv_path : str
                The path to the reduced CSV file.
        """

        # Create the reduced CSV filename
        reduced_csv_path = csv_path.replace(".csv", "_reduced.csv")

        # Open the CSV file.
        with open(csv_path, "r", newline='') as csv_file:
            # Create a CSV reader for the CSV file.
            csv_reader = csv.DictReader(csv_file)

            # Create the reduced CSV file.
            with open(reduced_csv_path, "w", newline='') as reduced_csv_file:
                # Create a CSV writer for the reduced CSV file.
                reduced_csv_writer = csv.DictWriter(reduced_csv_file, fieldnames=csv_reader.fieldnames)

                # Write the header row to the reduced CSV file.
                reduced_csv_writer.writeheader()

                # Iterate through the rows in the CSV file.
                for row in csv_reader:
                    # Check if the subject corresponds the subject set.
                    if row["subject_set_id"] == self.subject_set.id:
                        # Write the row to the reduced CSV file.
                        reduced_csv_writer.writerow(row)

        # Return the reduced CSV path.
        return reduced_csv_path

    def expandCSV(self, csv_path):
        """
        Expands a CSV file to include all of the fields in the subject objects' metadata dictionary.

        Parameters
        ----------
            csv_path : str
                The path to the CSV file that you want to expand.

        Returns
        -------
            expanded_csv_path : str
                The path to the expanded CSV file.
        """

        # Create the expanded CSV filename.
        expanded_csv_path = csv_path.replace(".csv", "_expanded.csv")

        # Create the expanded CSV file.
        with open(expanded_csv_path, mode="w", newline='') as expanded_csv_file:
            # Initialize the expanded CSV reader, header keys, and metadata keys.
            expanded_csv_reader = None
            header_keys = []
            metadata_keys = []

            # Open the CSV file.
            with open(csv_path, mode="r", newline='') as csv_file:
                # Create a CSV reader for the CSV file.
                csv_reader = csv.DictReader(csv_file)

                # Iterate through the rows in the CSV file.
                for index, row in enumerate(csv_reader):
                    # Check if this is the first row.
                    if (index == 0):
                        # Get the header keys
                        header_keys = row.keys()

                        # Get the metadata keys
                        metadata_keys = eval(row["metadata"]).keys()

                        # Combine the header keys and metadata keys.
                        header_keys = header_keys | metadata_keys

                        # Create the expanded CSV reader.
                        expanded_csv_reader = csv.DictWriter(expanded_csv_file, fieldnames=header_keys)

                        # Write the header row to the expanded CSV file.
                        expanded_csv_reader.writeheader()

                    # Create a new row dictionary.
                    new_row = {}

                    # Get the metadata row dictionary.
                    metadata_row_dict = eval(row["metadata"])

                    # Iterate through the header keys.
                    for key in header_keys:

                        # Check if the key is in the row or the metadata.
                        if (key in row.keys()):
                            # Add the key to the new row.
                            new_row[key] = row[key]
                        elif (key in metadata_keys):
                            # Add the key to the new row.
                            new_row[key] = metadata_row_dict[key]

                    # Write the new row to the expanded CSV file.
                    expanded_csv_reader.writerow(new_row)

        # Return the expanded CSV path.
        return expanded_csv_path

    def findValidSubjects(self, functional_condition, *field_names, display_printouts=True):
        """
        Finds all of the subjects in the subject set that satisfy the functional condition.

        Parameters
        ----------
            functional_condition : function
                The function that you want to use to filter the subjects.
            field_names : str
                The names of the fields that you want to use to filter the subjects.
            display_printouts : bool
                Whether or not you want to display printouts.

        Returns
        -------
            valid_subject_list : list
                A list of the subjects that satisfy the functional condition.
        """

        # Initialize the valid subject list.
        valid_subject_list = []

        # Iterate through the metadata list.
        for index, metadata in enumerate(self.metadata_list):
            # Check if the metadata is valid.
            if self.isValid(metadata, functional_condition, *field_names):
                # Get the subject.
                subject = Spout.get_subject(metadata.getFieldValue("subject_id"), self.subject_set.id)

                # Add the subject to the valid subject list.
                valid_subject_list.append(subject)

            # Display a printout every 1000 subjects.
            if(display_printouts and index % 1000 == 0 and index != 0):
                print(f"Checked {index} subjects.")

        # Return the valid subject list.
        return valid_subject_list

    @staticmethod
    def saveResult(subject_list, filename):
        """
        Saves the result of a subject list to a CSV file.

        Parameters
        ----------
            subject_list : list
                The list of subjects that you want to save.
            filename : str
                The name of the file that you want to save the subjects to.
        """

        # Save the subject list to a CSV file as a list of subject IDs.
        SubjectDiscriminator.saveResult(subject_list, filename)

    @staticmethod
    def loadResult(filename):
        """
        Loads the result of a subject list from a CSV file.

        Parameters
        ----------
            filename : str
                The name of the file that you want to load the subjects from.

        Returns
        -------
            subject_list : list
                The list of subjects that you want to load.
        """

        # Load the subject list from a CSV file from a list of subject IDs.
        return SubjectDiscriminator.loadResult(filename)

    @staticmethod
    def convertToCSV(subject_list, filename):
        """
        Converts a subject list to a CSV file.

        Parameters
        ----------
            subject_list : list
                The list of subjects that you want to convert.
            filename : str
                The name of the file that you want to convert the subjects to.

        Returns
        -------
            subject_list : list
                The list of subjects that you want to convert.
        """

        # Convert a subject list to a CSV file.
        SubjectDiscriminator.convertToCSV(subject_list, filename)