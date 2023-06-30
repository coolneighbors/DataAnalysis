import os.path
from time import sleep

import Checker
from Analyzer import Analyzer
from Classifier import Classifier

def runClassifier():
    classifier = Classifier("backyard-worlds-cool-neighbors-classifications.csv", "backyard-worlds-cool-neighbors-workflows.csv")
    classifier.classifyWorkflow(workflow_id=24299, v=1.6)

def findAcceptableCandidates(acceptance_ratio):
    subject_ids = analyzer.getSubjectIDs()
    accepted_subjects = []

    for subject_id in subject_ids:
        acceptable_boolean, subject_classifications_dict = analyzer.isAcceptableCandidate(subject_id, acceptance_ratio=acceptance_ratio)
        if (acceptable_boolean):
            print("Subject " + str(subject_id) + f" is an acceptable candidate: {subject_classifications_dict}")
            accepted_subjects.append(subject_id)

    print(f"Number of accepted subjects: {len(accepted_subjects)}")
    return accepted_subjects

def checkAcceptableCandidates(accepted_subjects):
    not_in_simbad_subjects = []
    not_in_gaia_subjects = []
    not_in_either_subjects = []

    separation = 32
    for index, subject_id in enumerate(accepted_subjects):
        print("Checking subject " + str(subject_id) + f" ({index + 1} out of {len(accepted_subjects)})")
        database_check_dict, database_query_dict = analyzer.checkSubjectFieldOfView(subject_id, gaia_separation=separation)
        no_database = not any(database_check_dict.values())
        if (no_database):
            print(f"Subject {subject_id} is not in either database.")
            not_in_either_subjects.append(subject_id)
            not_in_simbad_subjects.append(subject_id)
            not_in_gaia_subjects.append(subject_id)
        else:
            for database_name, in_database in database_check_dict.items():
                if (not in_database):
                    if (database_name == "SIMBAD"):
                        print(f"Subject {subject_id} is not in SIMBAD.")
                        not_in_simbad_subjects.append(subject_id)
                    elif (database_name == "Gaia"):
                        print(f"Subject {subject_id} is not in Gaia.")
                        not_in_gaia_subjects.append(subject_id)
                else:
                    if(database_name == "SIMBAD"):
                        print(f"Subject {subject_id} is in SIMBAD.")
                    elif(database_name == "Gaia"):
                        print(f"Subject {subject_id} is in Gaia.")

    not_in_simbad_subject_dataframes = analyzer.getSubjectDataframe(not_in_simbad_subjects)
    not_in_gaia_subject_dataframes = analyzer.getSubjectDataframe(not_in_gaia_subjects)
    not_in_either_subject_dataframes = analyzer.getSubjectDataframe(not_in_either_subjects)

    not_in_simbad_subjects_dataframe = Analyzer.combineSubjectDataframes(not_in_simbad_subject_dataframes)
    not_in_gaia_subjects_dataframe = Analyzer.combineSubjectDataframes(not_in_gaia_subject_dataframes)
    not_in_either_subjects_dataframe = Analyzer.combineSubjectDataframes(not_in_either_subject_dataframes)

    Analyzer.saveSubjectDataframe(not_in_simbad_subjects_dataframe, "not_in_simbad_subjects.csv")
    Analyzer.saveSubjectDataframe(not_in_gaia_subjects_dataframe, "not_in_gaia_subjects.csv")
    Analyzer.saveSubjectDataframe(not_in_either_subjects_dataframe, "not_in_either_subjects.csv")

    generated_files = ["not_in_simbad_subjects.csv", "not_in_gaia_subjects.csv", "not_in_either_subjects.csv"]
    return generated_files

def runAcceptableCandidateCheck():
    if (os.path.exists("acceptable_candidates.csv")):
        print("Found acceptable candidates file.")
        acceptable_candidates = analyzer.loadSubjectDataframe("acceptable_candidates.csv")
    else:
        print("No acceptable candidates file found. Generating new one.")
        acceptable_candidates = findAcceptableCandidates(acceptance_ratio=0.5)
        acceptable_candidates_dataframe = analyzer.combineSubjectDataframes(
            analyzer.getSubjectDataframe(acceptable_candidates))
        Analyzer.saveSubjectDataframe(acceptable_candidates_dataframe, "acceptable_candidates.csv")

    generated_files = checkAcceptableCandidates(acceptable_candidates)
    print("Generated files: " + str(generated_files))

def plotClassificationDistribution(title="Classification Distribution: Day x"):
    total_subject_count = 27801
    analyzer.plotClassificationDistribution(total_subject_count=total_subject_count, title=title)

def plotTopUsers(count_threshold=None, percentile=None, **kwargs):
    analyzer.plotTopUsersClassificationCounts(count_threshold=count_threshold, percentile=percentile, **kwargs)

#runClassifier()
extracted_file = "Extractions/question_extractor_workflow_24299_V1.6.csv"
reduced_file = "Reductions/question_reducer_workflow_24299_V1.6.csv"
subject_file = "backyard-worlds-cool-neighbors-subjects.csv"
analyzer = Analyzer(extracted_file, reduced_file, subject_file)

if (__name__ == "__main__"):
    plotTopUsers(percentile=98, title=f"Users in the Top 2% of Classifications: Day 4")
