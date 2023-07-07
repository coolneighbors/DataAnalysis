import os.path
import typing
from io import TextIOWrapper
from time import sleep

from Analyzer import Analyzer
from Classifier import Classifier
import astropy.units as u

def runClassifier():
    classifier = Classifier("backyard-worlds-cool-neighbors-classifications.csv", "backyard-worlds-cool-neighbors-workflows.csv")
    classifier.classifyWorkflow(workflow_id=24299, v=1.6)

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
    subjects = analyzer.plotConditionalQueries("testing.csv", database_name="Simbad")



