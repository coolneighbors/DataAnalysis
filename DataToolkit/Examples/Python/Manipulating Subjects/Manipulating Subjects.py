import warnings

import unWISE_verse.Spout
from unWISE_verse.Spout import Spout

def loggingInToZooniverse():
    # Get user's Zooniverse login information: username and password.
    # If you do not have a Zooniverse account, you can create one here: https://www.zooniverse.org/

    # This command will prompt the user to input their Zooniverse username and password via the console/terminal it is run within.
    login = Spout.requestLogin(filename="login.pickle", save=True)

    # When this is done once, if save is set to True, your login information will be locally saved to the filename.
    # Which is login.pickle by default.

    # Use the login information to log in to Zooniverse. All future requests to Zooniverse via Spout will be made using this login.
    Spout.loginToZooniverse(login)

def gettingSubjects():
    # The Spout object itself was developed to be a functional component of the unWISE-verse user interface. However,
    # static methods have been added to the Spout class to allow for the use of Spout outside of the unWISE-verse user interface.
    # This allows for access to already uploaded subjects and subject sets and the ability to modify them.

    # Provide the project id and subject set id you wish to get subjects from.
    # By default, the project ID and subject set ID are saved to the filenames "project_id.pickle" and "subject_set_id.pickle" and save is set to True.
    project_id, subject_set_id = Spout.requestZooniverseIDs(filenames=["project_id.pickle", "subject_set_id.pickle"], save=True)

    print("Warning: This will take a long time to run if there are a lot of subjects in the project.")

    # Get all subjects in the project.
    project_subjects = Spout.get_subjects_from_project(project_id, only_orphans=False)
    print(f"Number of subjects in the project: {len(project_subjects)}")

    # Get all orphaned subjects in the project.
    orphaned_project_subjects = Spout.get_subjects_from_project(project_id, only_orphans=True)
    print(f"Number of orphaned subjects in the project: {len(orphaned_project_subjects)}")

    # Get all subjects in the subject set.
    subject_set_subjects = Spout.get_subjects_from_project(project_id, subject_set_id, only_orphans=False)
    print(f"Number of subjects in the subject set: {len(subject_set_subjects)}")

    # Get a single subject by its ID.
    subject_id = None
    if(subject_id is not None):
        single_subject = Spout.get_subject(subject_id)
        print(f"Single subject: {single_subject}")

    # With the list of subjects, or individual subjects, you can do whatever you want with them. For example, you can find a subset of
    # subjects that meet a certain criteria using the SubjectDiscriminator class (see DataToolkit\Discriminator.py).

    # Additionally, you can get a user object from their ID or username:
    user_id = None
    if(user_id is not None):
        user = Spout.get_user(user_id)
        print(f"User: {user}")

    return subject_set_subjects

def modifyingSubjects(subjects):

    # To avoid accidentally modifying subjects, these functions are commented out.
    # If you would like to use them, please uncomment them. Be sure to verify you are modifying the correct subjects
    # before uncommenting them.

    # Remove subjects from a subject set:
    """
    Spout.remove_subjects(project_id, subject_set_id, subjects)
    """

    # Delete subjects from Zooniverse:
    """
    Spout.delete_subjects(subjects)
    """

    # Modify subject metadata field names:
    """
    Spout.modify_subject_metadata_field_name(subjects, "Ecliptic Coordinates", "#Ecliptic Coordinates")
    """

    # Modify subject metadata field values:
    """
    Spout.modify_subject_metadata_field_value(subjects, "FOV", "~120.0 x ~120.0 arcseconds")
    """

    # Check if a subject has images:
    """
    for subject in subjects:
        print(f"Subject {subject.id} has images: {Spout.subject_has_images(subject)}")
    """

    # Check if a subject has metadata:
    """
    for subject in subjects:
        print(f"Subject {subject.id} has metadata: {Spout.subject_has_metadata(subject)}")
    """


if (__name__ == "__main__"):
    loggingInToZooniverse()
    subjects = gettingSubjects()
    modifyingSubjects(subjects)



