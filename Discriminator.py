import csv

from panoptes_client.set_member_subject import SetMemberSubject
import unWISE_verse
from unWISE_verse import Data
import pickle

class Discriminator:
    def __init__(self, metadata_list):
        self.metadata_list = metadata_list

    def findValidMetadata(self, functional_condition, *field_names):
        valid_metadata_list = []
        for metadata in self.metadata_list:
            if self.isValid(metadata, functional_condition, *field_names):
                valid_metadata_list.append(metadata)
        return valid_metadata_list

    def isValid(self, metadata, functional_condition, *field_names):
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
        with open(filename, 'wb') as file:
            pickle.dump(list, file)

    @staticmethod
    def loadResult(filename):
        with open(filename, 'rb') as file:
            list = pickle.load(file)
            return list

class SubjectDiscriminator(Discriminator):
    def __init__(self, subject_set):
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
        metadata = Data.Metadata.createFromDictionary(subject.metadata)
        return metadata

    def findValidSubjects(self, functional_condition, *field_names, subject_as_input=False, display_printouts=True):
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
    def getSubjectFromSubjectSet(subject_set_id, subject_id):
        if(subject_set_id is None):
            for sms in SetMemberSubject.where(subject_id=subject_id):
                return sms.links.subject
        else:
            for sms in SetMemberSubject.where(subject_set_id=subject_set_id, subject_id=subject_id):
                return sms.links.subject

    @staticmethod
    def saveResult(subject_list, filename):
        subject_id_list = []
        for subject in subject_list:
            subject_id_list.append(subject.id)
        with open(filename, 'wb') as file:
            pickle.dump(subject_id_list, file)

    @staticmethod
    def loadResult(filename):
        with open(filename, 'rb') as file:
            list = pickle.load(file)
            subject_list = []
            for index, subject_id in enumerate(list):
                if(index % min(int(len(list)/10), 1000) == 0 and index != 0):
                    print(f"Loaded {index} subjects.")
                subject_list.append(SubjectDiscriminator.getSubjectFromSubjectSet(None, subject_id))
            return subject_list

    @staticmethod
    def convertToCSV(subject_list, filename):
        with open(filename, mode="w", newline='') as csv_file:
            fieldnames = subject_list[0].__dict__.keys() | subject_list[0].metadata.keys()
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for subject in subject_list:
                row_dict = subject.__dict__ | subject.metadata
                csv_writer.writerow(row_dict)

class CSVDiscriminator(Discriminator):
    def __init__(self, csv_path):
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
        self.subject_set = subject_set
        reduced_csv_path = self.reduceCSV(csv_path)
        expanded_csv_path = self.expandCSV(reduced_csv_path)
        super().__init__(expanded_csv_path)

    def reduceCSV(self, csv_path):
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
        valid_subject_list = []
        for index, metadata in enumerate(self.metadata_list):
            if self.isValid(metadata, functional_condition, *field_names):
                subject = SubjectDiscriminator.getSubjectFromSubjectSet(self.subject_set.id, metadata.getFieldValue("subject_id"))
                valid_subject_list.append(subject)
                if display_printouts:
                    print(f"Subject with TARGET ID {metadata.getFieldValue('TARGET ID')} is valid.")
            if(display_printouts and index % 1000 == 0 and index != 0):
                print(f"Checked {index} subjects.")
        return valid_subject_list

    @staticmethod
    def saveResult(subject_list, filename):
        SubjectDiscriminator.saveResult(subject_list, filename)

    @staticmethod
    def loadResult(filename):
        return SubjectDiscriminator.loadResult(filename)

    @staticmethod
    def convertToCSV(subject_list, filename):
        SubjectDiscriminator.convertToCSV(subject_list, filename)