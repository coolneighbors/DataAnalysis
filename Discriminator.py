import csv

from panoptes_client.set_member_subject import SetMemberSubject
import unWISE_verse
from unWISE_verse import Data
import pickle

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

        valid_metadata_list = []
        for metadata in self.metadata_list:
            if self.isValid(metadata, functional_condition, *field_names):
                valid_metadata_list.append(metadata)
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
        """

        for field_name in field_names:
            if not metadata.hasField(field_name):
                return False

        field_name_value_list = []
        for field_name in field_names:
            field_name_value = metadata.getFieldValue(field_name)
            field_name_value_list.append(field_name_value)
        field_name_value_tuple = tuple(field_name_value_list)
        # Keep in mind that when writing the functional_condition, the input values will all be strings.
        # You will need to convert them to the appropriate data type within the functional condition
        # if desirable to do so.
        result = functional_condition(*field_name_value_tuple)
        if(result is None):
            raise Exception("The functional condition returned None. This is not allowed.")
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

        self.subject_list = []
        for index, subject in enumerate(subject_set.subjects):
            if (index % 1000 == 0 and index != 0):
                print(f"Added {index} subjects to list.")
            self.subject_list.append(subject)

        self.metadata_list = []
        print(f"Length of Subject List: {len(self.subject_list)}")
        for index, subject in enumerate(self.subject_list):
            subject_metadata = self.getSubjectMetadata(subject)
            subject_metadata.addField("index", index)
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

        valid_subject_list = []
        if(subject_as_input):
            if(len(field_names) != 0):
                print("WARNING: You have requested a subject as input, but you have also passed in metadata field names. The metadata field names will be ignored.")
            for index, subject in enumerate(self.subject_list):
                result = functional_condition(subject)
                if(result is None):
                    raise Exception("The functional condition returned None. This is not allowed.")
                if(result):
                    valid_subject_list.append(subject)
                    if display_printouts:
                        print(f"Subject with TARGET ID {subject.metadata['TARGET ID']} is valid.")
                if (display_printouts and index % 1000 == 0 and index != 0):
                    print(f"Checked {index} subjects.")
        else:
            for index, metadata in enumerate(self.metadata_list):
                if self.isValid(metadata, functional_condition, *field_names):
                    valid_subject_list.append(self.subject_list[metadata.getFieldValue("index")])
                    if display_printouts:
                        print(f"Subject with TARGET ID {metadata.getFieldValue('TARGET ID')} is valid.")
                if(display_printouts and index % 1000 == 0 and index != 0):
                    print(f"Checked {index} subjects.")
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

        subject_id_list = []
        for subject in subject_list:
            subject_id_list.append(subject.id)
        with open(filename, 'wb') as file:
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

        with open(filename, 'rb') as file:
            list = pickle.load(file)
            subject_list = []
            for index, subject_id in enumerate(list):
                if(index % min(int(len(list)/10), 1000) == 0 and index != 0):
                    print(f"Loaded {index} subjects.")
                subject_list.append(Spout.get_subject(subject_id))
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

        with open(filename, mode="w", newline='') as csv_file:
            fieldnames = subject_list[0].__dict__.keys() | subject_list[0].metadata.keys()
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for subject in subject_list:
                row_dict = subject.__dict__ | subject.metadata
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

        self.csv_path = csv_path
        self.metadata_list = []
        with open(csv_path, "r", newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for index, row in enumerate(csv_reader):
                metadata = Data.Metadata.createFromDictionary(row)
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

        self.subject_set = subject_set
        reduced_csv_path = self.reduceCSV(csv_path)
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

        reduced_csv_path = csv_path.replace(".csv", "_reduced.csv")
        with open(csv_path, "r", newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            with open(reduced_csv_path, "w", newline='') as reduced_csv_file:
                reduced_csv_writer = csv.DictWriter(reduced_csv_file, fieldnames=csv_reader.fieldnames)
                reduced_csv_writer.writeheader()
                for row in csv_reader:
                    if row["subject_set_id"] == self.subject_set.id:
                        reduced_csv_writer.writerow(row)
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

        expanded_csv_path = csv_path.replace(".csv", "_expanded.csv")
        with open(expanded_csv_path, mode="w", newline='') as expanded_csv_file:
            expanded_csv_reader = None
            header_keys = []
            metadata_keys = []
            with open(csv_path, mode="r", newline='') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for index, row in enumerate(csv_reader):
                    if (index == 0):
                        header_keys = row.keys()
                        metadata_keys = eval(row["metadata"]).keys()
                        header_keys = header_keys | metadata_keys
                        expanded_csv_reader = csv.DictWriter(expanded_csv_file, fieldnames=header_keys)
                        expanded_csv_reader.writeheader()

                    new_row = {}
                    metadata_row_dict = eval(row["metadata"])
                    for key in header_keys:
                        if (key in row.keys()):
                            new_row[key] = row[key]
                        elif (key in metadata_keys):
                            new_row[key] = metadata_row_dict[key]
                    expanded_csv_reader.writerow(new_row)
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

        valid_subject_list = []
        for index, metadata in enumerate(self.metadata_list):
            if self.isValid(metadata, functional_condition, *field_names):
                subject = Spout.get_subject(metadata.getFieldValue("subject_id"), self.subject_set.id)
                valid_subject_list.append(subject)
                if display_printouts:
                    print(f"Subject with TARGET ID {metadata.getFieldValue('TARGET ID')} is valid.")
            if(display_printouts and index % 1000 == 0 and index != 0):
                print(f"Checked {index} subjects.")
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

        SubjectDiscriminator.convertToCSV(subject_list, filename)