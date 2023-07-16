from DataToolkit.Aggregator import Aggregator
import os


def runAggregator(workflow_id=24299, version=1.6, classifications_csv="backyard-worlds-cool-neighbors-classifications.csv", workflows_csv="backyard-worlds-cool-neighbors-workflows.csv", config_directory="Config", extractions_directory="Extractions", reductions_directory="Reductions"):
    aggregator = Aggregator(classifications_csv, workflows_csv, config_directory=config_directory, extractions_directory=extractions_directory, reductions_directory=reductions_directory)

    # Check if the extractions and reductions files have already been processed for this workflow_id and version
    if (os.path.exists("{}/question_extractor_workflow_{}_V{}.csv".format(aggregator.extractions_directory, workflow_id, version)) and os.path.exists("{}/question_reducer_workflow_{}_V{}.csv".format(aggregator.reductions_directory, workflow_id, version))):
        print("Aggregated files already exist, skipping aggregation.")
    else:
        # Aggregate the workflow classifications
        aggregator.aggregateWorkflow(workflow_id=workflow_id, v=version)


if (__name__ == "__main__"):
    print("Starting aggregation")
    runAggregator()
    print("Finished aggregation")