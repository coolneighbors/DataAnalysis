import datetime
import functools

import matplotlib.pyplot as plt

from Analyzer import Analyzer, Classifier
from Aggregator import Aggregator
from Decorators import ignore_warnings, timer
import astropy.units as u

def runAggregator():
    aggregator = Aggregator("backyard-worlds-cool-neighbors-classifications.csv", "backyard-worlds-cool-neighbors-workflows.csv")
    aggregator.aggregateWorkflow(workflow_id=24299, v=1.6)

def plotClassificationDistribution():
    total_subject_count = 27800
    analyzer.plotClassificationDistribution(total_subject_count=total_subject_count)

def plotTopUsers(count_threshold=None, percentile=None, **kwargs):
    analyzer.plotTotalClassificationsByTopUsers(count_threshold=count_threshold, percentile=percentile, **kwargs)

#runAggregator()
classification_file = "backyard-worlds-cool-neighbors-classifications.csv"
extracted_file = "Extractions/question_extractor_workflow_24299_V1.6.csv"
reduced_file = "Reductions/question_reducer_workflow_24299_V1.6.csv"
subject_file = "backyard-worlds-cool-neighbors-subjects.csv"
#analyzer = Analyzer(extracted_file, reduced_file, subject_file)
analyzer = Analyzer.load()

subject_ids = analyzer.getSubjectIDs()
test_usernames = ["Rattus", "pathfinder7567", "jcstew"]
intern = "acbravo"
worrisome_user = "eantonio2023"
# TODO: Investigate why box search is not always providing same results as cone search
if (__name__ == "__main__"):
    #analyzer.getSimbadQueryForSubject(89273238, plot=True)
    #analyzer.getSimbadQueryForSubject(89273238, search_type="Cone", radius=60 * u.arcsec, plot=True)









