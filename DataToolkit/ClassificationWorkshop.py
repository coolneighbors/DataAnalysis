import datetime
import functools
import os.path

import matplotlib.pyplot as plt

from DataToolkit.Analyzer import Analyzer, Classifier
from DataToolkit.Aggregator import Aggregator
from DataToolkit.Decorators import ignore_warnings, timer
import astropy.units as u

def runAggregator(workflow_id=24299, version=1.6, classifications_csv="backyard-worlds-cool-neighbors-classifications.csv", workflows_csv="backyard-worlds-cool-neighbors-workflows.csv",config_directory="Config", extractions_directory="Extractions", reductions_directory="Reductions"):
    # Check the reductions folder to see if the file already exists
    aggregator = Aggregator(classifications_csv, workflows_csv, config_directory=config_directory, extractions_directory=extractions_directory, reductions_directory=reductions_directory)
    if(os.path.exists("{}/question_extractor_workflow_{}_V{}.csv".format(aggregator.extractions_directory, workflow_id, version)) and os.path.exists("{}/question_reducer_workflow_{}_V{}.csv".format(aggregator.reductions_directory, workflow_id, version))):
        print("Aggregated files already exist, skipping aggregation")
    else:
        aggregator.aggregateWorkflow(workflow_id=workflow_id, v=version)

def plotClassificationDistribution():
    total_subject_count = 27800
    analyzer.plotClassificationDistribution(total_subject_count=total_subject_count)

def plotTopUsers(count_threshold=None, percentile=None, **kwargs):
    analyzer.plotTotalClassificationsByTopUsers(count_threshold=count_threshold, percentile=percentile, **kwargs)

runAggregator()
classification_file = "backyard-worlds-cool-neighbors-classifications.csv"
extracted_file = "Extractions/question_extractor_workflow_24299_V1.6.csv"
reduced_file = "Reductions/question_reducer_workflow_24299_V1.6.csv"
subject_file = "backyard-worlds-cool-neighbors-subjects.csv"
#analyzer = Analyzer(extracted_file, reduced_file, subject_file)
analyzer = Analyzer.load()

subject_ids = analyzer.getSubjectIDs()
test_usernames = ["Rattus", "pathfinder7567", "jcstew"]
# TODO: Investigate why box search is not always providing same results as cone search
if (__name__ == "__main__"):
    analyzer.classifier.plotAccuracyVsClassificationTotals(include_logged_out_users=True, default_insufficient_classifications=True, log_plot=True, classification_minimum=0, verified_classifications_minimum=10, accuracy_threshold=0)









