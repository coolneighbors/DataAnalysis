import csv

from matplotlib import pyplot as plt
from unWISE_verse.Login import Login
from unWISE_verse.Spout import Spout

from ConditionalFunctions import *

import Plotter
from Discriminator import SubjectDiscriminator, CSVDiscriminator, SubjectCSVDiscriminator

# Login
login = Spout.requestLogin()

# IDs
project_id, subject_set_id = Spout.requestZooniverseIDs()


# Spout
display_printouts = True
spout = Spout(project_identifier=project_id, login=login, display_printouts=display_printouts)

subject_set = spout.get_subject_set(subject_set_id)

print(f"Subject Set: {subject_set}")
discriminator = SubjectDiscriminator(subject_set)
print("Discriminator created.")


valid_subject_list = discriminator.findValidSubjects(all_subjects, display_printouts=True)

from Plotter import Plotter, CSVPlotter, SubjectPlotter, SubjectCSVPlotter

plot_kwargs = {"alpha": 0.5, "s": 10, "color": 'blue', "label": f"Subject Set {subject_set_id} Subjects"}
plotter = SubjectPlotter(valid_subject_list)
plotter.plot(plot_kwargs, show=False)
plt.legend(loc='upper right', fontsize='x-large')
plt.show()