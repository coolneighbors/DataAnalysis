from time import sleep

import Checker
from Analyzer import Analyzer
from Classifier import Classifier

#classifier = Classifier("backyard-worlds-cool-neighbors-classifications.csv", "backyard-worlds-cool-neighbors-workflows.csv")
#classifier.classifyWorkflow(workflow_id=24299, v=1.6)
from Discriminator import Discriminator, CSVDiscriminator

extracted_file = "Extractions/question_extractor_workflow_24299_V1.6.csv"
reduced_file = "Reductions/question_reducer_workflow_24299_V1.6.csv"
subject_file = "backyard-worlds-cool-neighbors-subjects.csv"
total_number_of_subjects = 27801
analyzer = Analyzer(extracted_file, reduced_file, subject_file)
subject_ids = analyzer.getSubjectIDs()
unique_users = analyzer.getUniqueUsers()
accepted_subjects = []
acceptance_ratio = 0.7


for subject_id in subject_ids:
    subject_metadata = analyzer.getSubjectMetadata(subject_id)
    acceptable_boolean, subject_classifications_dict = analyzer.isAcceptableCandidate(subject_id, acceptance_ratio=acceptance_ratio)
    if(acceptable_boolean):
        print("Subject " + str(subject_id) + f" is an acceptable candidate: {subject_classifications_dict}")
        accepted_subjects.append(subject_id)

print(f"Number of accepted subjects: {len(accepted_subjects)}")

not_in_simbad_subjects = []
not_in_gaia_subjects = []

not_in_either_subjects = []



for index, subject_id in enumerate(accepted_subjects):
    print("Checking subject " + str(subject_id) + f" ({index + 1} out of {len(accepted_subjects)})")
    simbad_result = analyzer.getSIMBADQuery(subject_id)
    if(simbad_result is None):
        print("SIMBAD query failed for subject " + str(subject_id))
        continue
    in_simbad = len(simbad_result) > 0

    gaia_result = analyzer.getGaiaQuery(subject_id)
    if(gaia_result is None):
        print("Gaia query failed for subject " + str(subject_id))
        continue
    in_gaia = len(gaia_result) > 0

    if(not in_simbad and not in_gaia):
        not_in_simbad_subjects.append(subject_id)
        not_in_gaia_subjects.append(subject_id)
        not_in_either_subjects.append(subject_id)
    elif(not in_simbad):
        not_in_simbad_subjects.append(subject_id)
    elif(not in_gaia):
        not_in_gaia_subjects.append(subject_id)

not_in_simbad_subject_dataframes = []
not_in_gaia_subject_dataframes = []
not_in_either_subject_dataframes = []

for subject_id in not_in_simbad_subjects:
    not_in_simbad_subject_dataframes.append(analyzer.getSubjectDataframe(subject_id))

for subject_id in not_in_gaia_subjects:
    not_in_gaia_subject_dataframes.append(analyzer.getSubjectDataframe(subject_id))

for subject_id in not_in_either_subjects:
    not_in_either_subject_dataframes.append(analyzer.getSubjectDataframe(subject_id))

not_in_simbad_subjects_dataframe = Analyzer.combineSubjectDataframes(not_in_simbad_subject_dataframes)
not_in_gaia_subjects_dataframe = Analyzer.combineSubjectDataframes(not_in_gaia_subject_dataframes)
not_in_either_subjects_dataframe = Analyzer.combineSubjectDataframes(not_in_either_subject_dataframes)

Analyzer.saveSubjectDataframe(not_in_simbad_subjects_dataframe, "not_in_simbad_subjects.csv")
Analyzer.saveSubjectDataframe(not_in_gaia_subjects_dataframe, "not_in_gaia_subjects.csv")
Analyzer.saveSubjectDataframe(not_in_either_subjects_dataframe, "not_in_either_subjects.csv")




