

def metadataDiscrimination():

    from DataToolkit.Discriminator import Discriminator
    from unWISE_verse.Data import Metadata

    print("Metadata Discrimination Example")
    print("------------------------------")


    print("The default character for private metadata is: {} \n".format(Metadata.privatization_symbol))

    # Create some metadata
    metadata_1 = Metadata(field_names=["Name", "Amount", "Cost", "Calories", "#Ranking"],
                          metadata_values=["Banana", 1, 0.25, 105, 1])
    metadata_2 = Metadata(field_names=["Name", "Amount", "Cost", "Calories", "#Ranking"],
                          metadata_values=["Apple", 1, 0.30, 95, 2])
    metadata_3 = Metadata(field_names=["Name", "Amount", "Cost", "Calories", "#Ranking"],
                          metadata_values=["Orange", 1, 0.35, 45, 3])
    metadata_4 = Metadata(field_names=["Name", "Amount", "Cost", "Calories", "#Ranking"],
                          metadata_values=["Pear", 1, 0.40, 105, 4])

    # Create a list of metadata
    metadata_list = [metadata_1, metadata_2, metadata_3, metadata_4]

    # Create a discriminator with the metadata list
    discriminator = Discriminator(metadata_list)

    # Create a functional condition, which is just some function that returns a boolean value
    # based on the field values of each metadata object. Must return either True or False, cannot return None.
    functional_condition = lambda cost: cost < 0.35

    # Find the valid metadata based on the functional condition
    # The further arguments are the field names of the metadata objects that the functional condition is based on.
    # "Cost", in this case, but it could be any of the field names or multiple field names (e.g. "Cost", "Calories")
    valid_metadata = discriminator.findValidMetadata(functional_condition, "Cost")

    # Print the valid metadata
    print("The valid metadata is:")
    for metadata in valid_metadata:
        print(metadata)
    print("\n")

    # Save the metadata objects to a pickle file
    discriminator.saveResult(valid_metadata, "metadata_discriminator_example.pkl")

    # Load the metadata objects from the pickle file
    loaded_metadata_list = discriminator.loadResult("metadata_discriminator_example.pkl")
    print(f"The loaded metadata is:")
    for metadata in loaded_metadata_list:
        print(metadata)
    print("\n")

    # Save the metadata objects to a csv file
    discriminator.saveResultToCSV(valid_metadata, "metadata_discriminator_example.csv")

def subjectDiscrimination():

    from unWISE_verse.Spout import Spout
    from unWISE_verse.Login import Login
    from DataToolkit.Discriminator import SubjectDiscriminator

    print("Subject Discrimination Example")
    print("------------------------------")

    # To interact with the Zooniverse API, you need to create a Spout object or call Spout.loginToZooniverse.
    # Once you have logged in once you will not need to log in again, unless you want to log in as a different user,
    # or you delete the login file. The login file is stored locally in the current working directory.

    # If you have not logged in previously, Spout.requestLogin() will prompt you to provide your login details in the console.
    login = Spout.requestLogin()

    # Alternatively, you can create a Login object yourself and pass it to Spout.loginToZooniverse(login)
    Spout.loginToZooniverse(login)

    # Set the project ID of the project you want to get subjects from
    project_id = 18925
    # Set the subject set ID of the subject set you want to get subjects from (optional)
    subject_set_id = 113708
    # Set whether you want to get only orphaned subjects (subjects that have no association with any existing subject sets) (optional)
    only_orphans = False

    # Prompts telling you to fill in the project ID and subject set ID if you have not already done so
    if(project_id is None):
        print("Please fill in the project id in the subjectDiscrimination() example.")
        return None

    if(subject_set_id is None):
        print("Please fill in the subject set id in the subjectDiscrimination() example, unless you want to get all subjects from the project.")
        return None

    if(only_orphans):
        print("Only getting orphaned subjects from the project.")


    # Get the subjects from the project
    subject_list = Spout.get_subjects_from_project(project_id, subject_set_id=subject_set_id, only_orphans=False)

    # Print the number of subjects in the project
    if(len(subject_list) == 0):
        print("There are no subjects in this project.")
        return None

    if(subject_set_id is None):
        print("There are {} subjects in this project.".format(len(subject_list)))
    else:
        print("There are {} subjects in this subject set.".format(len(subject_list)))

    # Create a discriminator with the subject list
    subject_discriminator = SubjectDiscriminator(subject_list)

    # Create a functional condition, which is just some function that returns a boolean value
    # based on the metadata values of each subject object. Must return either True or False, cannot return None.
    functional_condition = lambda ra, dec: float(ra) > 0 and float(dec) > 0

    # Find the valid subjects based on the functional condition
    # The further arguments are the metadata field names of the subject objects that the functional condition is based on.
    # "ra" and "dec", in this case, but it could be any of the metadata field names or multiple metadata field names.
    valid_subjects = subject_discriminator.findValidSubjects(functional_condition, "RA", "DEC")

    # Print the valid subjects
    max_count = 10
    print(f"There are {len(valid_subjects)} valid subjects. The first {min(max_count, len(valid_subjects))} are:")
    for subject in valid_subjects[:max_count]:
        print(subject)

    if(len(valid_subjects) > max_count):
        print("...\n")
    else:
        print("\n")

    # Save the valid subjects to a pickle file
    subject_discriminator.saveResult(valid_subjects, "subject_discriminator_example.pkl")

    # Load the valid subjects from the pickle file
    loaded_subject_list = subject_discriminator.loadResult("subject_discriminator_example.pkl")

    print(f"The first {min(max_count, len(loaded_subject_list))} loaded subjects are:")
    for subject in loaded_subject_list[:max_count]:
        print(subject)

    if (len(valid_subjects) > max_count):
        print("...\n")
    else:
        print("\n")

    # Save the valid subjects to a csv file
    subject_discriminator.saveResultToCSV(valid_subjects, "subject_discriminator_example.csv")

def csvDiscriminator():

        from DataToolkit.Discriminator import CSVDiscriminator

        print("CSV Discrimination Example")
        print("--------------------------")

        # Create a csv discriminator with a csv file
        csv_discriminator = CSVDiscriminator("csv_discriminator_example.csv")

        # Create a functional condition, which is just some function that returns a boolean value
        # based on the values of each row. Must return either True or False, cannot return None.
        functional_condition = lambda calories: float(calories) > 300

        # Find the valid rows based on the functional condition
        # The further arguments are the header names of the columns that the functional condition is based on.
        # "Calories", in this case, but it could be any of the header names or multiple header names.
        valid_metadata = csv_discriminator.findValidMetadata(functional_condition, "Calories")

        # Print the valid metadata
        print("The valid metadata are:")
        for row in valid_metadata:
            print(row)
        print("\n")

        # Save the metadata objects to a pickle file
        csv_discriminator.saveResult(valid_metadata, "csv_discriminator_example.pkl")

        # Load the metadata objects from the pickle file
        loaded_metadata_list = csv_discriminator.loadResult("csv_discriminator_example.pkl")

        print(f"The loaded metadata is:")
        for metadata in loaded_metadata_list:
            print(metadata)
        print("\n")

        # Save the metadata objects to a csv file
        csv_discriminator.saveResultToCSV(valid_metadata, "csv_discriminator_example_result.csv")

def subjectCSVDiscriminator():

        from unWISE_verse.Spout import Spout
        from unWISE_verse.Login import Login
        from DataToolkit.Discriminator import SubjectCSVDiscriminator

        print("Subject CSV Discrimination Example")
        print("----------------------------------")

        # Set the subject set ID of the subject set you want to get subjects from (optional)
        subject_set_id = 113708

        # Prompts telling you to fill in the subject set ID if you have not already done so.
        if(subject_set_id is None):
            print("Please fill in the subject set id in the subjectCSVDiscrimination() example, unless you want to get all subjects from the csv.")
            return None

        subject_csv = None

        if(subject_csv is None):
            print("Please fill in the subject csv file in the subjectCSVDiscrimination() example. This file can be downloaded from the Zooniverse project builder under the 'Data Exports' tab.")
            return None

        # SubjectCSVDiscriminator will automatically try to log in to Zooniverse if you have not already done so.
        # It will prompt you to enter your username and password as well as to provide a project ID if you have not already done so.
        subject_csv_discriminator = SubjectCSVDiscriminator(subject_csv, subject_set_identifer=subject_set_id)

        # Create a functional condition, which is just some function that returns a boolean value
        # based on the values of each row. Must return either True or False, cannot return None.
        functional_condition = lambda ra, dec: float(ra) > 0 and float(dec) > 0

        # Find the valid subjects based on the functional condition
        # The further arguments are the metadata field names of the subject objects that the functional condition is based on.
        # "ra" and "dec", in this case, but it could be any of the metadata field names or multiple metadata field names.
        valid_subjects = subject_csv_discriminator.findValidSubjects(functional_condition, "RA", "DEC")

        # Print the valid subjects
        max_count = 10
        print(f"There are {len(valid_subjects)} valid subjects. The first {min(max_count, len(valid_subjects))} are:")
        for subject in valid_subjects[:max_count]:
            print(subject)

        if(len(valid_subjects) > max_count):
            print("...\n")
        else:
            print("\n")

        # Save the valid subjects to a pickle file
        subject_csv_discriminator.saveResult(valid_subjects, "subject_discriminator_example.pkl")

        # Load the valid subjects from the pickle file
        loaded_subject_list = subject_csv_discriminator.loadResult("subject_discriminator_example.pkl")

        print(f"The first {min(max_count, len(loaded_subject_list))} loaded subjects are:")
        for subject in loaded_subject_list[:max_count]:
            print(subject)

        if (len(valid_subjects) > max_count):
            print("...\n")
        else:
            print("\n")

        # Save the valid subjects to a csv file
        subject_csv_discriminator.saveResultToCSV(valid_subjects, "subject_csv_discriminator_example_result.csv")

if (__name__ == "__main__"):
    metadataDiscrimination()
    subjectDiscrimination()
    csvDiscriminator()
    subjectCSVDiscriminator()
