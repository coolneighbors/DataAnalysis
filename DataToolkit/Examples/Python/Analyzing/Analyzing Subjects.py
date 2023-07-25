import os

from unWISE_verse.Spout import Spout

from DataToolkit.Aggregator import Aggregator
from DataToolkit.Analyzer import Analyzer
from DataToolkit.Decorators import ignore_warnings


def runAggregator():

    # For more details about Aggregating, see the Aggregating example.

    # This is the workflow ID and version for the Backyard Worlds: Cool Neighbors project's Launch-0 workflow.
    workflow_id = 24299
    version = 1.6

    # Default names for the CSV files that are exported via Zooniverse's data exports tab.
    classifications_csv = "backyard-worlds-cool-neighbors-classifications.csv"
    workflows_csv = "backyard-worlds-cool-neighbors-workflows.csv"
    config_directory = "Config"
    extractions_directory = "Extractions"
    reductions_directory = "Reductions"

    # Check whether the aggregated files already exist for this workflow and version
    aggregator = Aggregator(classifications_csv_filename=classifications_csv, workflow_csv_filename=workflows_csv, config_directory=config_directory, extractions_directory=extractions_directory, reductions_directory=reductions_directory)

    if(os.path.exists("{}/question_extractor_workflow_{}_V{}.csv".format(aggregator.extractions_directory, workflow_id, version)) and os.path.exists("{}/question_reducer_workflow_{}_V{}.csv".format(aggregator.reductions_directory, workflow_id, version))):
        print("Aggregated files already exist, skipping aggregation.")
    else:
        aggregator.aggregateWorkflow(workflow_id=workflow_id, v=version)

def createAnalyzer():
    # Provide the filepaths of the aggregated files
    extracted_file = "Extractions/question_extractor_workflow_24299_V1.6.csv"
    reduced_file = "Reductions/question_reducer_workflow_24299_V1.6.csv"

    # Subject file is optional but highly recommended, as it allows for you to work with subjects offline
    # and is generally faster than the online version.
    subject_file = "backyard-worlds-cool-neighbors-subjects.csv"

    if(subject_file is not None):
        # If an offline analyzer has already been created and saved, you can load it instead of creating it again. You cannot
        # load an online analyzer.
        if (os.path.exists("analyzer.pickle")):
            analyzer = Analyzer.load()
        else:
            analyzer = Analyzer(extracted_file, reduced_file, subject_file)
    else:
        analyzer = Analyzer(extracted_file, reduced_file)

    return analyzer

def gettingInformation(analyzer):
    print("\nUsing Analyzer...\n")
    # The analyzer object contains a lot of functionality. It is recommended to look at the Analyzer class in DataToolkit\Analyzer.py
    # to see everything it can do. Here are some examples of what you can do with the analyzer:

    # Get the total number of classifications in the aggregated files.
    print(f"Number of classifications: {analyzer.getTotalClassifications()}\n")

    # Get the total number of classifications for a subjects which have at least n classifications.
    n = 5
    print(f"Number of classifications for subjects with at least {n} classifications: {analyzer.getSubsetOfTotalClassifications(minimum_subject_classification_count=5)}\n")

    # Get the valid subject ids from the workflow classifications.
    subject_ids = analyzer.getSubjectIDs()
    print(f"Valid Subjects:", *subject_ids[0:10], "...\n")

    # Plot the classification distribution for all subjects.
    total_subject_count = 27800
    print("Plotting classification distribution for all valid subjects...\n")
    analyzer.plotClassificationDistribution(total_subject_count=total_subject_count)

    # Compute time statistics for the classifications.
    print("Computing time statistics for classifications...")
    users_average_time, users_std_time, users_median_time = analyzer.computeTimeStatisticsForClassifications()
    print(f"Average time: {round(users_average_time,2)} seconds\nStandard deviation: {round(users_std_time,2)} seconds\nMedian: {round(users_median_time,2)} seconds\n")

    # Get the usernames or user ids of the users who have classified.
    usernames = analyzer.getUniqueUserIdentifiers(user_identifier="username")
    print(f"Usernames:", *usernames[0:10], "...\n")

    # include_logged_out_users must be false since logged-out users do not have user ids.
    user_ids = analyzer.getUniqueUserIdentifiers(include_logged_out_users=False, user_identifier="user id")
    print(f"User ids:", *user_ids[0:10], "...\n")

    # Get classifications done by a specific user.
    user_classifications = analyzer.getClassificationsByUser(usernames[0])
    print(f"Classifications by user {usernames[0]}: \n{user_classifications}\n")

    # Get the total number of classifications done by a specific user.
    user_classification_count = analyzer.getTotalClassificationsByUser(usernames[0])
    print(f"Total classifications by user {usernames[0]}: {user_classification_count}\n")

    # Get the total number of classifications done by the top users (two modes: percentile or classification threshold).
    top_users_classification_count = analyzer.getTotalClassificationsByTopUsers(classification_threshold=None, percentile=98)
    print(f"Total classifications by top users: {top_users_classification_count}\n")

    # Plot the classifications done by the top users (two modes: percentile or classification threshold).
    print("Plotting classifications done by top users...\n")
    analyzer.plotTotalClassificationsByTopUsers(classification_threshold=None, percentile=98)

    # Get the classifications done for a specific subject.
    subject_classifications = analyzer.getClassificationsForSubject(subject_ids[0])
    print(f"Classifications for subject {subject_ids[0]}: \n{subject_classifications}\n")

    # Plot the classifications done for a specific subject.
    print(f"Plotting classifications for subject {subject_ids[0]}...\n")
    analyzer.plotClassificationsForSubject(subject_ids[0])

    # Compute the time statistics for a specific user.
    user_average_time, user_std_time, user_median_time = analyzer.computeTimeStatisticsForUser(usernames[0])
    print(f"Average time for user {usernames[0]}: {round(user_average_time,2)} seconds\nStandard deviation: {round(user_std_time,2)} seconds\nMedian: {round(user_median_time,2)} seconds\n")

    # Plot the time histogram for a specific user.
    print(f"Plotting time histogram for user {usernames[0]}...\n")
    analyzer.plotTimeHistogramForUserClassifications(usernames[0])

    # Plot the time histogram for all classifications/users. Since there are users who have too
    # few consecutive classifications, which brings up a warning, I am ignoring warnings for this function.
    print("Plotting time histogram for all classifications...\n")
    ignore_warnings(analyzer.plotTimeHistogramForAllClassifications)()

    # Plot classification timeline
    print("Plotting classification timeline...\n")
    analyzer.plotClassificationTimeline()

    # Login to Zooniverse with Spout to access the next two functions.
    # You will need to log in to Spout to use these functions or use online mode.
    login = Spout.requestLogin()
    Spout.loginToZooniverse(login)

    # Get the user object for a specific user. Not disabled for offline mode, but you will need to log in to Spout
    # to get the user object.
    user_object = analyzer.getUser(usernames[0])
    print(f"User object for user {usernames[0]}: {user_object}\n")

    # Get the subject object for a specific subject. Not disabled for offline mode, but you will need to log in to Spout
    # to get the subject object.
    subject_object = analyzer.getSubject(subject_ids[0])
    print(f"Subject object for subject {subject_ids[0]}: {subject_object}\n")

    # Get the top usernames (two modes: percentile or classification threshold).
    top_usernames = analyzer.getTopUsernames(classification_threshold=None, percentile=98)
    print(f"Top usernames: {top_usernames}\n")

    # Get the number of users in the top users (two modes: percentile or classification threshold).
    top_user_count = analyzer.getTopUsernamesCount(classification_threshold=None, percentile=98)
    print(f"Number of users in top users: {top_user_count}\n")

    # Gets the user objects of the top users (two modes: percentile or classification threshold).
    top_user_objects = analyzer.getTopUsers(classification_threshold=None, percentile=98)
    print(f"Top user objects: {top_user_objects}\n")

    # Get the Pandas dataframe associated with the subject from the subject file,
    # which contains all the data and metadata information about the subject, the extracted file which contains the
    # classification information, and the reduced file which contains the aggregated classification information.

    # Get the subject dataframe for a specific subject.
    subject_dataframe_from_subject_file = analyzer.getSubjectDataframe(subject_ids[0], dataframe_type="default")
    print(f"Subject dataframe for subject {subject_ids[0]}: \n{subject_dataframe_from_subject_file}\n")

    # Get the subject dataframe for a specific subject from the extracted file.
    subject_dataframe_from_extracted_file = analyzer.getSubjectDataframe(subject_ids[0], dataframe_type="extracted")
    print(f"Extracted Subject dataframe for subject {subject_ids[0]}: \n{subject_dataframe_from_extracted_file}\n")

    # Get the subject dataframe for a specific subject from the reduced file.
    subject_dataframe_from_reduced_file = analyzer.getSubjectDataframe(subject_ids[0], dataframe_type="reduced")
    print(f"Reduced Subject dataframe for subject {subject_ids[0]}: \n{subject_dataframe_from_reduced_file}\n")

    # Combine subject dataframes.
    subject_dataframe_0 = analyzer.getSubjectDataframe(subject_ids[0], dataframe_type="default")
    subject_dataframe_1 = analyzer.getSubjectDataframe(subject_ids[1], dataframe_type="default")
    combined_subject_dataframe = analyzer.combineSubjectDataframes([subject_dataframe_0, subject_dataframe_1])
    print(f"Combined subject dataframe: \n{combined_subject_dataframe}\n")

    # Save the subject dataframe to a CSV file.
    analyzer.saveSubjectDataframeToFile(subject_dataframe_from_subject_file, "subject_dataframe.csv")

    # Load the subject dataframe from a CSV file.
    subject_dataframe_from_file = analyzer.loadSubjectDataframeFromFile("subject_dataframe.csv")
    print(f"Subject dataframe from file: \n{subject_dataframe_from_file}\n")

    # Verify that a subject exists.
    subject_exists = analyzer.subjectExists(subject_ids[0])
    print(f"Subject {subject_ids[0]} exists: {subject_exists}\n")

    # Get the subject metadata for a specific subject.
    subject_metadata = analyzer.getSubjectMetadata(subject_ids[0])
    print(f"Subject metadata for subject {subject_ids[0]}: \n{subject_metadata}\n")

    # Get a particular subject metadata field for a specific subject.
    subject_metadata_field = analyzer.getSubjectMetadataField(subject_ids[0], "ID")
    print(f"Subject metadata field for subject {subject_ids[0]}: \n{subject_metadata_field}\n")

    # Show the subject in wise-view.
    print(f"Showing subject {subject_ids[0]} in wise-view...\n")
    analyzer.showSubjectInWiseView(subject_ids[0], open_in_browser=True)

    # Get the SIMBAD link for a specific subject.
    simbad_link = analyzer.getSimbadLinkForSubject(subject_ids[0])
    print(f"SIMBAD link for subject {subject_ids[0]}: {simbad_link}\n")

def runningQueries(analyzer):
    # Get the subject IDs for the queries.
    subject_ids = analyzer.getSubjectIDs()

    # Get the SIMBAD query for a specific subject.
    simbad_query = analyzer.getSimbadQueryForSubject(subject_ids[0], plot=True)
    print(f"SIMBAD query for subject {subject_ids[0]}: \n{simbad_query}\n")

    # Get the conditional SIMBAD query for a specific subject.
    conditional_simbad_query = analyzer.getConditionalSimbadQueryForSubject(subject_ids[0], plot=True)
    print(f"Conditional SIMBAD query for subject {subject_ids[0]}: \n{conditional_simbad_query}\n")

    # Check if there exists a source in SIMBAD for a specific subject's FOV.
    source_exists_in_simbad = analyzer.sourceExistsInSimbadForSubject(subject_ids[0])
    print(f"Source exists in SIMBAD for subject {subject_ids[0]}: {source_exists_in_simbad}\n")

    # Get the Gaia query for a specific subject.
    gaia_query = analyzer.getGaiaQueryForSubject(subject_ids[0], plot=True)
    print(f"Gaia query for subject {subject_ids[0]}: \n{gaia_query}\n")

    # Get the conditional Gaia query for a specific subject.
    conditional_gaia_query = analyzer.getConditionalGaiaQueryForSubject(subject_ids[0], plot=True)
    print(f"Conditional Gaia query for subject {subject_ids[0]}: \n{conditional_gaia_query}\n")

    # Check if there exists a source in Gaia for a specific subject's FOV.
    source_exists_in_gaia = analyzer.sourceExistsInGaiaForSubject(subject_ids[0])
    print(f"Source exists in Gaia for subject {subject_ids[0]}: {source_exists_in_gaia}\n")

def findingCandidates(analyzer):
    # Get the subject IDs for the queries.
    subject_ids = analyzer.getSubjectIDs()

    # Get the subject type for a specific subject.
    subject_type = analyzer.getSubjectType(subject_ids[0])
    print(f"Subject type for subject {subject_ids[0]}: {subject_type}\n")

    # Check if a specific subject is an acceptable candidate.
    is_acceptable_candidate, subject_classifications = analyzer.checkIfCandidateIsAcceptable(subject_ids[0], 0.5, acceptance_threshold=1, weighted=False)
    print(f"Subject {subject_ids[0]} an acceptable candidate: {is_acceptable_candidate}")
    print(f"Subject classifications for subject {subject_ids[0]}: {subject_classifications}\n")

    # Find the acceptable candidates.
    # Saves the acceptable candidates to a csv file.
    acceptable_candidates = analyzer.findAcceptableCandidates(acceptance_ratio=0.5, save=True, weighted=False)
    print(f"Acceptable candidates: {acceptable_candidates}\n")

    # Sort and exclude the acceptable candidates by database.
    print("Warning: This may take a while...\n")
    generated_files = analyzer.sortAcceptableCandidatesByDatabase(acceptable_candidates)
    print(f"Generated files: {generated_files}\n")

    # To perform both the acceptable candidate finding and sorting in one step, run the following:
    print("Warning: This may take a while...\n")
    analyzer.performCandidatesSort(acceptance_ratio=0.5)
    print("Acceptable candidates found and sorted!\n")

def usingClassifier(analyzer):

    # Get the classifier object from within the analyzer object.
    classifier = analyzer.classifier

    # Get the usernames from the analyzer.
    usernames = analyzer.getUniqueUserIdentifiers(user_identifier="username", include_logged_out_users=True)

    # Get the user accuracy for a specific user.
    user_accuracy = classifier.getUserAccuracy(usernames[0], default_insufficient_classifications=True)
    print(f"User accuracy for user {usernames[0]}: {user_accuracy}\n")

    # Get the user verified classifications for a specific user.
    verified_classifications_by_user = classifier.getUserVerifiedClassifications(usernames[0])
    print(f"Verified classifications by user {usernames[0]}: {verified_classifications_by_user}\n")

    # Get the user information for a specific user.
    user_information = classifier.getUserInformation(usernames[0], default_insufficient_classifications=True)
    print(f"User information for user {usernames[0]}: {user_information}\n")

    # Get the user accuracy for all users.
    user_accuracies = classifier.getAllUserAccuracies(include_logged_out_users=True, default_insufficient_classifications=True)
    print("User accuracies:", *user_accuracies[0:10], "...\n")

    # Get all user information.
    user_information = classifier.getAllUserInformation(include_logged_out_users=True, default_insufficient_classifications=True)
    print(f"User information: too much to display...\n")

    # Get the most accurate users.
    most_accurate_users = classifier.getMostAccurateUsernames(include_logged_out_users=True, default_insufficient_classifications=True, classification_threshold=0, verified_classifications_threshold=10, accuracy_threshold=0.0)
    print("Most accurate users:", *most_accurate_users[0:10], "...\n")

    # Plot user performance.
    classifier.plotUserPerformance(usernames[0])

    # Plot all users' performance as a histogram.
    classifier.plotAllUsersPerformanceHistogram(include_logged_out_users=True, default_insufficient_classifications=True)

    # Plot top users' performance as a histogram.
    classifier.plotTopUsersPerformanceHistogram(classification_threshold=None, percentile=98, default_insufficient_classifications=True)

    # Plot top users' performances
    classifier.plotTopUsersPerformances(classification_threshold=None, percentile=98, default_insufficient_classifications=True)

    # Plot most accurate users' performances
    classifier.plotMostAccurateUsers(include_logged_out_users=True, default_insufficient_classifications=True, classification_threshold=0, verified_classifications_threshold=100, accuracy_threshold=0.0)

    # Plot accuracy vs. number of classifications
    classifier.plotAccuracyVsClassificationTotals(include_logged_out_users=True, default_insufficient_classifications=True, log_plot=True, classification_threshold=0, verified_classifications_threshold=100, accuracy_threshold=0.0)

if (__name__ == "__main__"):
    runAggregator()
    analyzer = createAnalyzer()
    gettingInformation(analyzer)
    #runningQueries(analyzer)
    #findingCandidates(analyzer)
    #usingClassifier(analyzer)







