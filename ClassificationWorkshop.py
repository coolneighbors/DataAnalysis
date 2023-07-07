import datetime
import functools
import os.path
import typing
from io import TextIOWrapper
from time import sleep

import numpy as np
import pandas as pd

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

# Create a timing decorator
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        time_in_ms = (end_time - start_time).total_seconds() * 1000
        print(f"Function {func.__name__} took {time_in_ms} ms to run.")
        return result
    return wrapper

test_subjects = [89459775, 89459776, 89459778, 89459780]
if (__name__ == "__main__"):
    analyzer.plotSubjectClassifications(test_subjects)



