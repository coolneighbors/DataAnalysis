import csv

from matplotlib import pyplot as plt
from unWISE_verse.Login import Login
from unWISE_verse.Spout import Spout

import Plotter
from Discriminator import SubjectDiscriminator, CSVDiscriminator, SubjectCSVDiscriminator

# Login
login = Login(username='austinh2001', password='qhmq3FAuAwkNVQT')

# IDs
project_id = 18801
subject_set_id = 113407

# Spout
display_printouts = True
spout = Spout(project_identifier=project_id, login=login, display_printouts=display_printouts)

subject_set = spout.get_subject_set(subject_set_id)

print(f"Subject Set: {subject_set}")
discriminator = SubjectDiscriminator(subject_set)
print("Discriminator created.")

def in_targets_file(TARGET_ID):
    TARGET_ID = int(TARGET_ID)
    target_filename = "bad_candidates-to_remove.csv"
    with open(target_filename, newline='') as targets_file:
        reader = csv.DictReader(targets_file)
        for row in reader:
            if(TARGET_ID == int(row["TARGET ID"])):
                return True
    return False

def not_in_targets_file(TARGET_ID):
    TARGET_ID = int(TARGET_ID)
    target_filename = "bad_candidates-to_remove.csv"
    with open(target_filename, newline='') as targets_file:
        reader = csv.DictReader(targets_file)
        for row in reader:
            if(TARGET_ID == int(row["TARGET ID"])):
                return False
    return True

def all_subjects():
    return True

valid_subject_list = discriminator.findValidSubjects(all_subjects, display_printouts=True)

from Plotter import Plotter, CSVPlotter, SubjectPlotter, SubjectCSVPlotter

plot_kwargs = {"alpha": 0.5, "s": 1, "color": 'green', "label": "Good Candidates"}
plotter = CSVPlotter("coolneighbors-to_remove.csv")
plotter.plot(plot_kwargs, show=False)

plot_kwargs = {"alpha": 0.5, "s": 1, "color": 'red', "label": "Bad Candidates"}
plotter = CSVPlotter("bad_candidates-to_remove-full.csv")
plotter.plot(plot_kwargs, show=False)

plot_kwargs = {"alpha": 0.5, "s": 1, "color": 'blue', "label": "Remaining Candidates"}
plotter = SubjectPlotter(valid_subject_list)
plotter.plot(plot_kwargs, show=False)

plt.legend(loc='upper right', fontsize='x-large')
plt.title("Cool Neighbors Candidates")
plt.show()