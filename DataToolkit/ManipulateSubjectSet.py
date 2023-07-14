import csv
import getpass
import math
import sys
from time import sleep

import panoptes_client
from panoptes_client import Subject
from unWISE_verse.Login import Login
from unWISE_verse.Spout import Spout

from Aggregator import Aggregator
from ConditionalFunctions import *
from Decorators import timer
from unWISE_verse.MetadataPointers import generate_VizieR_url
from Discriminator import Discriminator, SubjectDiscriminator, CSVDiscriminator, SubjectCSVDiscriminator

# Login
login = Spout.requestLogin()

# IDs
project_id, subject_set_id = Spout.requestZooniverseIDs()

# Spout
display_printouts = False
spout = Spout(project_identifier=project_id, login=login, display_printouts=display_printouts)


"""
#The following lines are examples of how to modify subject metadata.

# This line will change the field name "TARGET ID" to "#TARGET ID" for all subjects in the list, valid_subject_list.
#spout.modify_subject_metadata_field_name(valid_subject_list, "Ecliptic Coordinates", "#Ecliptic Coordinates")

# This line will change the field value for the metadata "FOV" to "~120.0 x ~120.0 arcseconds" for all subjects in the list, valid_subject_list.
#spout.modify_subject_metadata_field_value(valid_subject_list, "FOV", "~120.0 x ~120.0 arcseconds")

# Please note that the following line will delete the subjects. This is irreversible.
# Please verify that your functional condition is absolutely correct before uncommenting the following line.
# This line will delete all subjects in the list, valid_subject_list, from the subject set.
#spout.delete_subjects(valid_subject_list)

#valid_subject_list = discriminator.findValidSubjects(all_subjects, display_printouts=True)
#print(f"Valid Subject List: {valid_subject_list}")
#print(f"Number of valid subjects: {len(valid_subject_list)}")
"""


