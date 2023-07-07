import datetime
import functools
import inspect
import math
import os
import pickle
import typing
import warnings
from copy import copy
from io import TextIOWrapper
from time import sleep

import astropy
import numpy as np
import pandas
import pandas as pd
import webbrowser
import matplotlib.pyplot as plt
import functools
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from typing import List, Dict, Tuple, Union, Optional, TextIO, Any, Callable, Iterable, Set
from unWISE_verse.Spout import Spout
from Searcher import SimbadSearcher, GaiaSearcher
from Plotter import SubjectCSVPlotter
# TODO: Add a weighting system for users' classifications based on their accuracy on the known subjects
# TODO: Redo documentation for Analyzer class and add type hints, result hints, and docstrings for all methods
# TODO: Check hashed IP's to see if they're the same user as a logged-in user for classification counts, etc.
# TODO: Investigate Gini_coefficient for classification distribution analysis (As suggested by Marc)
from numpy import ndarray

file_typing = Union[str, TextIOWrapper, TextIO]

# Input typing for @uses_subject_ids decorator
subjects_typing = Union[str, int, TextIOWrapper, TextIO, pd.DataFrame, Iterable[Union[str, int]]]

# Processes subject_input into a list of integer subject_ids
def uses_subject_ids(func):
    @functools.wraps(func)
    def conversion_wrapper(*args, **kwargs):
        subject_ids = []
        is_static_method = isinstance(func, staticmethod)
        is_class_method = isinstance(func, classmethod)

        if (is_static_method or is_class_method):
            self = None
            subject_input = args[0]
            args = args[1:]
        else:
            self = args[0]
            subject_input = args[1]
            args = args[2:]

        if (subject_input is None):
            return None
        elif ((isinstance(subject_input, str) and os.path.exists(subject_input)) or isinstance(subject_input, TextIOWrapper) or isinstance(subject_input, TextIO)):
            subject_ids = pd.read_csv(Analyzer.verifyFile(subject_input, ".csv"), usecols=["subject_id"])["subject_id"].tolist()
        elif(isinstance(subject_input, str) or isinstance(subject_input, int)):
            try:
                subject_ids = int(subject_input)
            except ValueError:
                raise ValueError("Invalid subject ID: " + str(subject_input))
        elif (isinstance(subject_input, pd.DataFrame)):
            subject_ids = subject_input["subject_id"].tolist()
        elif (isinstance(subject_input, Iterable)):
            for subject_id in subject_input:
                try:
                    subject_ids.append(int(subject_id))
                except ValueError:
                    raise ValueError("Invalid subject ID: " + str(subject_id))
        else:
            raise TypeError("Invalid subjects type: " + str(type(subject_input)))

        if(self is None):
            return func.__func__(subject_ids, *args, **kwargs)
        else:
            return func(self, subject_ids, *args, **kwargs)
    return conversion_wrapper

# Converts a function that takes in a single value and returns a single value into a function that takes in an iterable and returns a list of values
def multioutput(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        is_static_method = isinstance(func, staticmethod)
        is_class_method = isinstance(func, classmethod)
        if(is_static_method or is_class_method):
            if(len(args) == 0):
                raise TypeError("No arguments provided to {}.".format(func.__name__))
            if(isinstance(args[0], Iterable) and not isinstance(args[0], str) and not isinstance(args[0], TextIOWrapper) and not isinstance(args[0], TextIO)):
                iterable_type = type(args[0])
                iterable_values = args[0]
                args = args[1:]
                return iterable_type([func.__func__(iterable_value, *args, **kwargs) for iterable_value in iterable_values])
            else:
                return func.__func__(*args, **kwargs)
        else:
            self = args[0]
            args = args[1:]
            if(len(args) == 0):
                raise TypeError("No arguments provided to {}.".format(func.__name__))
            if(isinstance(args[0], Iterable) and not isinstance(args[0], str) and not isinstance(args[0], TextIOWrapper) and not isinstance(args[0], TextIO)):
                iterable_type = type(args[0])
                iterable_values = args[0]
                args = args[1:]
                return iterable_type([func(self, iterable_value, *args, **kwargs) for iterable_value in iterable_values])
            else:
                return func(self, *args, **kwargs)
    return wrapper

class Analyzer:
    def __init__(self, extracted_file: file_typing, reduced_file: file_typing, subjects_file: Optional[file_typing] = None, save_login: bool = True) -> None:
        # Verify that the extracted_file is a valid file
        self.extracted_file, self.reduced_file = Analyzer.verifyFile((extracted_file, reduced_file), required_file_type=".csv")

        # Read the extracted and reduced files into Pandas DataFrames
        self.extracted_dataframe = pd.read_csv(self.extracted_file)
        self.reduced_dataframe = pd.read_csv(self.reduced_file)

        # Verify that the subjects_file is a valid file, if provided.
        if (subjects_file is not None):
            self.subjects_file = subjects_file
            self.subjects_dataframe = pd.read_csv(self.subjects_file)
        else:
            # If no subjects_file is provided, then the user must be logged in to access the subjects.
            self.subjects_file = None
            self.subjects_dataframe = None
            self.subject_set_id = None
            self.login(save_login)

    @multioutput
    @staticmethod
    def verifyFile(file: file_typing, required_file_type: Optional[str] = None) -> str:
        """
        Verifies that the file is a valid file.

        Parameters
        ----------
        file : Union[str, TextIOWrapper]
            The file to verify.
        required_file_type : Optional[str]
            The required file type of the file. If None, then the file type is not checked.

        Returns
        -------
        file_path : str
            The path to the file.

        Raises
        ------
        TypeError
            If the file is not a valid file input.
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not the required file type.

        """

        file_path = None

        if (isinstance(file, str)):
            file_path = file
        elif (isinstance(file, TextIOWrapper)):
            file_path = file.name
            file.close()
        else:
            raise TypeError(f"File '{file}' is not a valid file input.")

        if (not os.path.exists(file_path)):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        if(required_file_type is not None):
            if(not file_path.endswith(required_file_type)):
                raise ValueError(f"File '{file_path}' is not a {required_file_type} file.")

        return file_path

    def login(self, save: bool = True) -> None:
        """
        Logs the user into the Zooniverse project and saves the login information to a file.

        Parameters
        ----------
        save : bool
            Whether to save the login information to a file.
        """

        # Get login
        login = Spout.requestLogin(save=save)

        # Get Zooniverse IDs
        project_id, self.subject_set_id = Spout.requestZooniverseIDs(save=save)

        # Create Spout object to access the Zooniverse project
        Spout(project_identifier=project_id, login=login, display_printouts=True)

    def getUniqueUsers(self, include_logged_out=True):
        if(include_logged_out):
            return self.extracted_dataframe["user_name"].unique()
        else:
            unique_users = self.extracted_dataframe["user_name"].unique()
            logged_in_unique_users = [user for user in unique_users if "not-logged-in" not in user]
            return logged_in_unique_users

    def getTotalClassifications(self):
        # Return the number of classifications
        return len(self.extracted_dataframe)

    @multioutput
    def getUserClassifications(self, user_identifier):
        # Check if the user_id is a string or an integer
        if(isinstance(user_identifier, str)):
            # If it's a string, then it's a username
            username = user_identifier
            return self.extracted_dataframe[self.extracted_dataframe["user_name"] == username]
        else:
            # If it's an integer, then it's a Zooniverse ID
            return self.extracted_dataframe[self.extracted_dataframe["user_id"] == user_identifier]

    @multioutput
    def getUserClassificationsCount(self, user_id):
        # Return the number of classifications made by that user
        return len(self.getUserClassifications(user_id))

    def getTopUsers(self, count_threshold=None, percentile=None):

        if (count_threshold is None and percentile is None):
            raise ValueError("You must provide either a count_threshold or a percentile.")
        elif(count_threshold is not None and percentile is not None):
            raise ValueError("You cannot provide both a count_threshold and a percentile.")

        unique_users = self.getUniqueUsers()
        user_classifications_dict = {}

        for unique_user in unique_users:
            user_classifications_dict[unique_user] = self.getUserClassificationsCount(unique_user)

        # Sort the dictionary by the number of classifications
        sorted_user_classifications_dict = {k: v for k, v in sorted(user_classifications_dict.items(), key=lambda item: item[1])}

        # Reverse the dictionary
        sorted_user_classifications_dict = dict(reversed(list(sorted_user_classifications_dict.items())))

        top_users_dict = {}

        def userMeetsRequirements(user, count_threshold, percentile):
            if(percentile is not None):
                if(sorted_user_classifications_dict[user] >= np.percentile(list(sorted_user_classifications_dict.values()), percentile)):
                    return True
                else:
                    return False
            else:
                if(sorted_user_classifications_dict[user] >= count_threshold):
                    return True
                else:
                    return False

        for user in sorted_user_classifications_dict:
            if(userMeetsRequirements(user, count_threshold, percentile)):
                top_users_dict[user] = sorted_user_classifications_dict[user]
        return top_users_dict

    def getTopUsersCount(self, count_threshold=None, percentile=None):
        top_users_dict = self.getTopUsers(count_threshold=count_threshold, percentile=percentile)
        return len(top_users_dict)

    def getTopUsersClassificationsCount(self, count_threshold=None, percentile=None):
        top_users_dict = self.getTopUsers(count_threshold=count_threshold, percentile=percentile)
        return sum(top_users_dict.values())

    def plotTopUsersClassificationCounts(self, count_threshold=None, percentile=None, **kwargs):
        top_users_dict = self.getTopUsers(count_threshold=count_threshold, percentile=percentile)
        # Plot the number of classifications made by each user but stop names from overlapping
        usernames = list(top_users_dict.keys())
        user_counts = list(top_users_dict.values())
        # Generate x-coordinates for the bars
        x = np.arange(len(top_users_dict))

        # Create the bar plot
        fig, ax = plt.subplots()

        if ("title" in kwargs):
            plt.title(kwargs["title"])
            del kwargs["title"]
        else:
            if (percentile is not None):
                plt.title(f"Users in the Top {100 - percentile}% of Classifications")
            elif (count_threshold is not None):
                plt.title(f"Users with More Than {count_threshold} Classifications")

        if ("ylabel" in kwargs):
            plt.ylabel(kwargs["ylabel"])
            del kwargs["ylabel"]
        else:
            plt.ylabel("Number of Classifications", fontsize=15)

        bars = ax.bar(x, user_counts, **kwargs)

        for i, bar in enumerate(bars):
            # Display the username below the bar, but angled 45 degrees
            offset = 0.2
            ax.text(bar.get_x() + bar.get_width() / 2 + offset, -5, usernames[i], ha='right', va='top', rotation=45)

            # Display the user's count
            ax.text(bar.get_x() + bar.get_width() / 2, user_counts[i] + 10, str(user_counts[i]), horizontalalignment='center', verticalalignment='bottom', fontsize=10)

        # Customize the plot
        ax.set_xticks(x)
        ax.set_xticklabels([])

        plt.show()

    def getSubjectIDs(self):
        # Return the list of subject IDs
        return self.reduced_dataframe["subject_id"].values

    @uses_subject_ids
    @multioutput
    def getSubjectDataframe(self, subject_input: subjects_typing, dataframe_type: str = "default") -> pd.DataFrame:

        def getSubjectDataframeFromID(subject_id: int, dataframe_type: str = "default") -> pd.DataFrame:

            if(not isinstance(subject_id, int)):
                raise ValueError("The subject_id must be an integer.")

            if (dataframe_type == "default"):
                # If default, then return the metadata of the subject as a dataframe
                subject_metadata = self.getSubjectMetadata(subject_id)

                if(subject_metadata is None):
                    warnings.warn(f"Subject {subject_id} does not exist, returning empty Dataframe.")
                    return pd.DataFrame()

                # Add the standard subject_id column
                subject_metadata["subject_id"] = subject_id

                # Add the standard metadata column
                subject_metadata["metadata"] = str(subject_metadata)

                # Convert the metadata to a dataframe
                subject_metadata_dataframe = pd.DataFrame.from_dict(subject_metadata, orient="index").transpose()
                return subject_metadata_dataframe

            elif (dataframe_type == "reduced"):
                # If reduced, then return the reduced dataframe for that subject
                return self.reduced_dataframe[self.reduced_dataframe["subject_id"] == subject_id]
            elif (dataframe_type == "extracted"):
                # If not reduced, then return the extracted dataframe for that subject
                return self.extracted_dataframe[self.extracted_dataframe["subject_id"] == subject_id]

        subject_dataframe = getSubjectDataframeFromID(subject_input, dataframe_type=dataframe_type)

        # ignore warnings about setting values on a copy of a slice from a DataFrame
        pd.options.mode.chained_assignment = None
        subject_dataframe.drop_duplicates(subset=["subject_id"], inplace=True)
        return subject_dataframe

    def combineSubjectDataframes(self, subject_dataframes: Iterable[pd.DataFrame]) -> pd.DataFrame:
        # Combine a list of subject dataframes into a single dataframe
        if(not isinstance(subject_dataframes, Iterable)):
            if(isinstance(subject_dataframes, pd.DataFrame)):
                return subject_dataframes
            else:
                raise ValueError("The subject_dataframes must be a list of dataframes.")

        subject_dataframe = pd.concat(subject_dataframes, ignore_index=True)
        subject_dataframe.drop_duplicates(subset=["subject_id"], inplace=True)
        subject_dataframe.reset_index(drop=True, inplace=True)
        return subject_dataframe

    @multioutput
    @uses_subject_ids
    def getSubjectClassifications(self, subject_input):
        # Get the subject's dataframe
        subject_dataframes = self.getSubjectDataframe(subject_input, dataframe_type="reduced")
        classification_dicts = []

        if(not isinstance(subject_dataframes, list)):
            subject_dataframes = [subject_dataframes]

        for subject_dataframe in subject_dataframes:

            try:
                # Try to get the number of "yes" classifications
                yes_count = int(subject_dataframe["data.yes"].values)
            except ValueError:
                # If there are no "yes" classifications, then set the count to 0
                yes_count = 0

            try:
                # Try to get the number of "no" classifications
                no_count = int(subject_dataframe["data.no"].values[0])
            except ValueError:
                # If there are no "no" classifications, then set the count to 0
                no_count = 0

            # Return the dictionary of the number of "yes" and "no" classifications
            classification_dicts.append({"yes": yes_count, "no": no_count})

        if(len(classification_dicts) == 1):
            return classification_dicts[0]
        else:
            return classification_dicts

    def getClassificationCountDict(self, total_subject_count=None):
        subject_ids = self.getSubjectIDs()
        number_of_classifications_dict = {}
        for subject_id in subject_ids:
            classification_dict = self.getSubjectClassifications(subject_id)
            total_classifications = classification_dict['yes'] + classification_dict['no']
            if (total_classifications in number_of_classifications_dict):
                number_of_classifications_dict[total_classifications] += 1
            else:
                number_of_classifications_dict[total_classifications] = 1

        if(total_subject_count is not None):
            number_of_classifications_dict[0] = total_subject_count - len(subject_ids)
        number_of_classifications_dict = dict(sorted(number_of_classifications_dict.items()))
        return number_of_classifications_dict

    def getSubsetClassificationCount(self, minimum_count=0, total_subject_count=None):
        classification_count_dict = self.getClassificationCountDict(total_subject_count)
        return sum(value for key, value in classification_count_dict.items() if key >= minimum_count)

    def plotClassificationDistribution(self, total_subject_count=None, **kwargs):
        # if title is in kwargs, then remove it and put its value in the title
        if("title" in kwargs):
            plt.title(kwargs["title"])
            del kwargs["title"]
        else:
            plt.title("Classification Distribution")

        if("xlabel" in kwargs):
            plt.xlabel(kwargs["xlabel"])
            del kwargs["xlabel"]
        else:
            plt.xlabel("Number of Classifications")
        if("ylabel" in kwargs):
            plt.ylabel(kwargs["ylabel"])
            del kwargs["ylabel"]
        else:
            plt.ylabel("Number of Subjects")

        number_of_classifications_dict = self.getClassificationCountDict(total_subject_count)

        plt.bar(number_of_classifications_dict.keys(), number_of_classifications_dict.values(), **kwargs)

        for key in number_of_classifications_dict:
            plt.text(key, number_of_classifications_dict[key], number_of_classifications_dict[key], ha='center', va='bottom')

        plt.xticks(range(len(number_of_classifications_dict)))
        plt.show()

    def plotSubjectClassifications(self, subject_id):

        # Get the number of "yes" and "no" classifications for that subject as a dictionary
        classification_dict = self.getSubjectClassifications(subject_id)

        # Get the number of "yes" and "no" classifications from the dictionary
        yes_count = classification_dict["yes"]
        no_count = classification_dict["no"]

        # Compute the total number of classifications
        total_count = yes_count + no_count

        # Compute the percentage of "yes" and "no" classifications
        yes_percent = yes_count / total_count
        no_percent = no_count / total_count

        # Plot the pie chart
        plt.pie([yes_percent, no_percent], labels=["Yes", "No"], autopct='%1.1f%%')
        plt.axis('equal')
        plt.title("Subject ID: " + str(subject_id) + " Classifications")
        plt.legend([f"{yes_count} classifications", f"{no_count} classifications"])
        plt.show()

    # TODO: Rewrite these classification time methods to use a single method which takes in a list of usernames and returns a list of classification times/stats
    def computeClassificationTimeStatistics(self):

        # Get the unique usernames
        usernames = self.getUniqueUsers()

        # Initialize the list of classification times
        users_classification_times = []

        # Iterate over all unique usernames
        for username in usernames:

            # Get the user's classifications
            user_classifications = self.getUserClassifications(username)

            # Convert the created_at column to datetime objects
            user_times = pd.to_datetime(user_classifications["created_at"])

            # Initialize the previous index
            previous_index = None

            # Iterate over all indices in the user's classifications
            for index in user_times.index:
                # If there is a previous index, then compute the time difference
                if(previous_index is not None):
                    # Compute the time difference between the current and previous classification
                    time_difference = user_times[index] - user_times[previous_index]

                    # Set the upper time limit to 5 minutes
                    upper_time_limit = 60*5

                    # If the time difference is less than the upper time limit, then add it to the list of classification times
                    if(time_difference.seconds < upper_time_limit):
                        users_classification_times.append(time_difference.seconds)
                # Set the previous index to the current index
                previous_index = index

        # Compute the average time between classifications for all users
        users_average_time = sum(users_classification_times) / len(users_classification_times)

        users_std_time = np.std(users_classification_times)

        median_time = np.median(users_classification_times)

        # Return the average time between classifications for all users
        return users_average_time, users_std_time, median_time

    @multioutput
    def computeUserClassificationTimeStatistics(self, username):
        # Get the user's classifications
        user_classifications = self.getUserClassifications(username)

        user_classification_times = []

        # Convert the created_at column to datetime objects
        user_times = pd.to_datetime(user_classifications["created_at"])

        # Initialize the previous index
        previous_index = None

        # Iterate over all indices in the user's classifications
        for index in user_times.index:
            # If there is a previous index, then compute the time difference
            if (previous_index is not None):
                # Compute the time difference between the current and previous classification
                time_difference = user_times[index] - user_times[previous_index]

                # Set the upper time limit to 5 minutes
                upper_time_limit = 60 * 5

                # If the time difference is less than the upper time limit, then add it to the list of classification times
                if (time_difference.seconds < upper_time_limit):
                    user_classification_times.append(time_difference.seconds)
            # Set the previous index to the current index
            previous_index = index

        # Compute the average time between classifications for the user


        if(len(user_classification_times) == 0):
            raise ValueError("User " + username + " has no classifications.")
        user_average_time = sum(user_classification_times) / len(user_classification_times)

        user_std_time = np.std(user_classification_times)

        median_time = np.median(user_classification_times)

        # Return the average time between classifications for all users
        return user_average_time, user_std_time, median_time

    def plotClassificationTimeHistogram(self, bins=100):

        # Get the unique usernames
        usernames = self.getUniqueUsers()

        # Initialize the list of classification times
        users_classification_times = []

        # Iterate over all unique usernames
        for username in usernames:
            # Get the user's classifications
            user_classifications = self.getUserClassifications(username)

            # Convert the created_at column to datetime objects
            user_times = pd.to_datetime(user_classifications["created_at"])

            # Initialize the previous index
            previous_index = None

            # Iterate over all indices in the user's classifications
            for index in user_times.index:
                # If there is a previous index, then compute the time difference
                if(previous_index is not None):
                    # Compute the time difference between the current and previous classification
                    time_difference = user_times[index] - user_times[previous_index]

                    # Set the lower time limit to 0 seconds
                    lower_time_limit = 0

                    # Set the upper time limit to 5 minutes
                    upper_time_limit = 60*5

                    # If the time difference is less than the upper time limit, then add it to the list of classification times
                    if(time_difference.seconds > lower_time_limit and time_difference.seconds < upper_time_limit):
                        users_classification_times.append(time_difference.seconds)

                # Set the previous index to the current index
                previous_index = index

        # Plot the histogram
        plt.hist(users_classification_times, bins=bins)

        users_average_time = sum(users_classification_times) / len(users_classification_times)
        plt.axvline(users_average_time, color='red', linestyle='solid', linewidth=1, label="Average Time: " + str(round(users_average_time, 2)) + " seconds")

        user_std_time = np.std(users_classification_times)
        plt.axvline(users_average_time + user_std_time, color='red', linestyle='dashed', linewidth=1, label="Standard Deviation: " + str(round(user_std_time, 2)) + " seconds")
        plt.axvline(users_average_time - user_std_time, color='red', linestyle='dashed', linewidth=1)

        median_time = np.median(users_classification_times)
        plt.axvline(median_time, color='orange', linestyle='solid', linewidth=1, label="Median: " + str(median_time) + " seconds")

        plt.title("Classification Time Histogram")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

    def plotUserClassificationTimeHistogram(self, username, bins=100):

        # Initialize the list of classification times
        user_classification_times = []

        # Iterate over all unique usernames
        # Get the user's classifications
        user_classifications = self.getUserClassifications(username)

        # Convert the created_at column to datetime objects
        user_times = pd.to_datetime(user_classifications["created_at"])

        # Initialize the previous index
        previous_index = None

        # Iterate over all indices in the user's classifications
        for index in user_times.index:
            # If there is a previous index, then compute the time difference
            if (previous_index is not None):
                # Compute the time difference between the current and previous classification
                time_difference = user_times[index] - user_times[previous_index]

                # Set the lower time limit to 0 seconds
                lower_time_limit = 0

                # Set the upper time limit to 5 minutes
                upper_time_limit = 60 * 5

                # If the time difference is less than the upper time limit, then add it to the list of classification times
                if (time_difference.seconds > lower_time_limit and time_difference.seconds < upper_time_limit):
                    user_classification_times.append(time_difference.seconds)

            # Set the previous index to the current index
            previous_index = index

        # Plot the histogram
        plt.hist(user_classification_times, bins=bins)

        users_average_time = sum(user_classification_times) / len(user_classification_times)
        plt.axvline(users_average_time, color='red', linestyle='solid', linewidth=1, label="Average Time: " + str(round(users_average_time, 2)) + " seconds")

        user_std_time = np.std(user_classification_times)
        plt.axvline(users_average_time + user_std_time, color='red', linestyle='dashed', linewidth=1, label="Standard Deviation: " + str(round(user_std_time, 2)) + " seconds")
        plt.axvline(users_average_time - user_std_time, color='red', linestyle='dashed', linewidth=1)

        median_time = np.median(user_classification_times)
        plt.axvline(median_time, color='orange', linestyle='solid', linewidth=1, label="Median: " + str(median_time) + " seconds")

        plt.title(f"{username} Classification Time Histogram")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

    def classificationTimeline(self, bar=True, binning_parameter = "Day", **kwargs):

        # Get the classification datetimes
        classification_datetimes = pd.to_datetime(self.extracted_dataframe["created_at"])

        # Initialize the binned datetimes dictionary
        binned_datetimes = {}

        # Iterate over all classification datetimes
        for classification_datetime in classification_datetimes:

            # Bin the datetimes
            if(binning_parameter == "Day"):
                day = classification_datetime.date()
                if day in binned_datetimes:
                    binned_datetimes[day].append(classification_datetime)
                else:
                    binned_datetimes[day] = [classification_datetime]
            elif(binning_parameter == "Week"):
                week = classification_datetime.isocalendar()[1]
                if week in binned_datetimes:
                    binned_datetimes[week].append(classification_datetime)
                else:
                    binned_datetimes[week] = [classification_datetime]
            elif(binning_parameter == "Month"):
                month = classification_datetime.month
                if month in binned_datetimes:
                    binned_datetimes[month].append(classification_datetime)
                else:
                    binned_datetimes[month] = [classification_datetime]
            elif(binning_parameter == "Year"):
                year = classification_datetime.year
                if year in binned_datetimes:
                    binned_datetimes[year].append(classification_datetime)
                else:
                    binned_datetimes[year] = [classification_datetime]

        # Convert the binned datetimes to a dictionary of counts
        binned_datetimes = {k: len(v) for k, v in binned_datetimes.items()}

        # Plot the timeline
        if(bar):
            plt.bar(binned_datetimes.keys(), binned_datetimes.values(), **kwargs)
        else:
            plt.plot(binned_datetimes.keys(), binned_datetimes.values(), **kwargs)
        plt.title("Classification Timeline")
        plt.xlabel(binning_parameter)
        plt.ylabel("Count")
        plt.show()

    @multioutput
    @uses_subject_ids
    def getSubject(self, subject_input):

        # Get the subject with the given subject ID in the subject set with the given subject set ID
        return Spout.get_subject(subject_input, self.subject_set_id)

    @uses_subject_ids
    @multioutput
    def getSubjectMetadata(self, subject_input):

        # Get the subject with the given subject ID
        if(self.subjects_file is not None):
            subject_dataframe = self.subjects_dataframe[self.subjects_dataframe["subject_id"] == subject_input]
            if(subject_dataframe.empty):
                return None
            else:
                return eval(self.subjects_dataframe[self.subjects_dataframe["subject_id"] == subject_input].iloc[0]["metadata"])
        else:
            subject = self.getSubject(subject_input)

            try:
                # Get the subject's metadata
                subject_metadata = subject.metadata

                # Return the subject's metadata
                return subject_metadata
            except AttributeError:
                # If the subject could not be found, then return None
                return None

    @uses_subject_ids
    @multioutput
    def getSubjectMetadataField(self, subject_input, field_name):

        # Get the subject's metadata
        subject_metadata = self.getSubjectMetadata(subject_input)

        if(subject_metadata is None):
            return None

        # Get the metadata field with the given field name
        field_value = subject_metadata.get(field_name)

        # Return the metadata field value
        return field_value

    @uses_subject_ids
    @multioutput
    def showSubject(self, subject_id, open_in_browser=False):

        # Get the WiseView link for the subject with the given subject ID
        wise_view_link = self.getSubjectMetadataField(subject_id, "WISEVIEW")

        # Return None if no WiseView link was found
        if(wise_view_link is None):
            print(f"No WiseView link found for subject {subject_id}, so it cannot be shown.")
            return None

        # Remove the WiseView link prefix and suffix
        wise_view_link = wise_view_link.removeprefix("[WiseView](+tab+")
        wise_view_link = wise_view_link.removesuffix(")")

        # Determine whether to open the subject in the default web browser
        if wise_view_link is None:
            print(f"No WiseView link found for subject {subject_id}")
            return None
        else:
            if(open_in_browser):
                webbrowser.open(wise_view_link)
                delay = 10
                print(f"Opening WiseView link for subject {subject_id}.")
                sleep(delay)

        # Return the WiseView link
        return wise_view_link

    @uses_subject_ids
    @multioutput
    def getSimbadLink(self, subject_id):

        simbad_link = self.getSubjectMetadataField(subject_id, "SIMBAD")

        if (simbad_link is None):
            print(f"No SIMBAD link found for subject {subject_id}, so it cannot be provided.")
            return None

            # Remove the SIMBAD link prefix and suffix
        simbad_link = simbad_link.removeprefix("[SIMBAD](+tab+")
        simbad_link = simbad_link.removesuffix(")")

        return simbad_link

    @uses_subject_ids
    @multioutput
    def getSimbadQuery(self, subject_id, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec, plot=False, separation=None):
        subject_metadata = self.getSubjectMetadata(subject_id)

        if(subject_metadata is None):
            print(f"No metadata found for subject {subject_id}, so a SIMBAD query cannot be performed.")
            return None

        RA = subject_metadata["RA"]
        DEC = subject_metadata["DEC"]
        coords = [RA, DEC]

        if(search_type == "Box" or search_type == "Box Search"):
            search_parameters = {"Coordinates": coords, "Type": search_type, "FOV": FOV}
        elif(search_type == "Cone" or search_type == "Cone Search"):
            search_parameters = {"Coordinates": coords, "Type": search_type, "radius": radius}
        else:
            raise ValueError(f"Invalid search type: {search_type}. Expected 'Cone', 'Cone Search', 'Box', or 'Box Search'.")

        simbad_searcher = SimbadSearcher(search_parameters)

        result = simbad_searcher.getQuery()

        if(plot):
            simbad_searcher.plotEntries(separation=separation)

        return result

    @uses_subject_ids
    @multioutput
    def getConditionalSimbadQuery(self, subject_id, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec, otypes=["BD*", "BD?", "BrownD*", "BrownD?", "BrownD*_Candidate", "PM*"], plot=False, separation=None):
        subject_metadata = self.getSubjectMetadata(subject_id)

        if (subject_metadata is None):
            print(f"No metadata found for subject {subject_id}, so a SIMBAD conditional query cannot be performed.")
            return None

        RA = subject_metadata["RA"]
        DEC = subject_metadata["DEC"]
        coords = [RA, DEC]

        # Introduce a buffer to the FOV to more reliably capture high proper motion objects
        extreme_proper_motion = 5 * u.arcsec / u.yr
        current_epoch = float(Time.now().decimalyear) * u.yr
        simbad_epoch = 2000 * u.yr
        time_difference = current_epoch - simbad_epoch
        max_separation = extreme_proper_motion * time_difference
        buffer_FOV = 2 * max_separation
        buffer_radius = max_separation

        if (search_type == "Box" or search_type == "Box Search"):
            search_parameters = {"Coordinates": coords, "Type": search_type, "FOV": FOV + buffer_FOV}
        elif (search_type == "Cone" or search_type == "Cone Search"):
            search_parameters = {"Coordinates": coords, "Type": search_type, "radius": radius + buffer_radius}
        else:
            raise ValueError(f"Invalid search type: {search_type}. Expected 'Cone', 'Cone Search', 'Box', or 'Box Search'.")

        simbad_searcher = SimbadSearcher(search_parameters)
        otypes_condition = simbad_searcher.buildConditionalArgument("OTYPES", "==", otypes)
        conditions = [otypes_condition]
        result = simbad_searcher.getConditionalQuery(conditions)

        if(plot):
            simbad_searcher.plotEntries(separation=separation)

        return result

    @uses_subject_ids
    @multioutput
    def sourceExistsInSimbad(self, subject_id, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec):

        return len(self.getSimbadQuery(subject_id, search_type=search_type, FOV=FOV, radius=radius)) > 0

    @uses_subject_ids
    @multioutput
    def getGaiaQuery(self, subject_id, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec, plot=False, separation=None):
        # Get the subject's metadata
        subject_metadata = self.getSubjectMetadata(subject_id)

        if (subject_metadata is None):
            print(f"No metadata found for subject {subject_id}, so a Gaia query cannot be performed.")
            return None

        RA = subject_metadata["RA"]
        DEC = subject_metadata["DEC"]
        coords = [RA, DEC]

        if (search_type == "Box" or search_type == "Box Search"):
            search_parameters = {"Coordinates": coords, "Type": search_type, "FOV": FOV}
        elif (search_type == "Cone" or search_type == "Cone Search"):
            search_parameters = {"Coordinates": coords, "Type": search_type, "radius": radius}
        else:
            raise ValueError(f"Invalid search type: {search_type}. Expected 'Cone', 'Cone Search', 'Box', or 'Box Search'.")

        gaia_searcher = GaiaSearcher(search_parameters)

        result = gaia_searcher.getQuery()

        if (plot):
            gaia_searcher.plotEntries(separation=separation)

        return result

    @uses_subject_ids
    @multioutput
    def getConditionalGaiaQuery(self, subject_id, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec, gaia_pm=100 * u.mas / u.yr, plot=False, separation=None):
        subject_metadata = self.getSubjectMetadata(subject_id)

        if (subject_metadata is None):
            print(f"No metadata found for subject {subject_id}, so a Gaia conditional query cannot be performed.")
            return None

        RA = subject_metadata["RA"]
        DEC = subject_metadata["DEC"]
        coords = [RA, DEC]

        # Introduce a buffer to the FOV to more reliably capture high proper motion objects
        extreme_proper_motion = 5 * u.arcsec / u.yr
        current_epoch = float(Time.now().decimalyear) * u.yr
        gaia_epoch = 2016 * u.yr
        time_difference = current_epoch - gaia_epoch
        max_separation = extreme_proper_motion * time_difference
        buffer_FOV = 2 * max_separation
        buffer_radius = max_separation

        if (search_type == "Box" or search_type == "Box Search"):
            search_parameters = {"Coordinates": coords, "Type": search_type, "FOV": FOV + buffer_FOV}
        elif (search_type == "Cone" or search_type == "Cone Search"):
            search_parameters = {"Coordinates": coords, "Type": search_type, "radius": radius + buffer_radius}
        else:
            raise ValueError(
                f"Invalid search type: {search_type}. Expected 'Cone', 'Cone Search', 'Box', or 'Box Search'.")

        gaia_searcher = GaiaSearcher(search_parameters)
        proper_motion_condition = gaia_searcher.buildConditionalArgument("pm", ">=", gaia_pm)
        result = gaia_searcher.getConditionalQuery(proper_motion_condition)

        if (plot):
            gaia_searcher.plotEntries(separation=separation)

        return result

    @uses_subject_ids
    @multioutput
    def sourceExistsInGaia(self, subject_id, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec):

        return len(self.getGaiaQuery(subject_id, search_type=search_type, FOV=FOV, radius=radius)) > 0

    @uses_subject_ids
    @multioutput
    def subjectExists(self, subject_id):

        # Get the subject's metadata
        metadata = self.getSubjectMetadata(subject_id)

        # Return whether the subject exists
        return metadata is not None

    @multioutput
    @staticmethod
    def bitmaskToType(bitmask):

        # Convert the bitmask to an integer
        try:
            bitmask = int(bitmask)
        except ValueError:
            raise ValueError("bitmask must be an integer or a string that can be converted to an integer.")

        # Initialize the bitmask dictionary
        bitmask_dict = {2**0: "SMDET Candidate", 2**1: "Blank", 2**2: "Known Brown Dwarf", 2**3: "Quasar", 2**4: "Random Sky Location", 2**5: "White Dwarf"}

        # Return the bitmask type associated with the bitmask value
        return bitmask_dict.get(bitmask, None)

    @uses_subject_ids
    @multioutput
    def getSubjectType(self, subject_id):
        # Get the bitmask for the subject
        bitmask = self.getSubjectMetadataField(subject_id, "#BITMASK")

        if(bitmask is None):
            print(f"Subject {subject_id} does not have a bitmask, so its type cannot be determined.")
            return None

        # Convert the bitmask to a subject type
        bitmask_type = self.bitmaskToType(bitmask)

        return bitmask_type

    @uses_subject_ids
    @multioutput
    def isAcceptableCandidate(self, subject_id, acceptance_ratio=None, acceptance_threshold=0):
        subject_type = self.getSubjectType(subject_id)
        subject_classifications = self.getSubjectClassifications(subject_id)

        # Count the number of successful classifications for each of the bitmask types
        total_classifications = subject_classifications["yes"] + subject_classifications["no"]
        movement_ratio = subject_classifications["yes"] / total_classifications
        non_movement_ratio = subject_classifications["no"] / total_classifications

        if (subject_type == "SMDET Candidate"):
            return (movement_ratio > acceptance_ratio) and (subject_classifications["yes"] > acceptance_threshold), subject_classifications
        else:
            return False, subject_classifications

    def findAcceptableCandidates(self, acceptance_ratio=None, acceptance_threshold=None, save=True):
        subject_ids = self.getSubjectIDs()
        accepted_subjects = []

        for subject_id in subject_ids:
            acceptable_boolean, subject_classifications_dict = self.isAcceptableCandidate(subject_id, acceptance_ratio=acceptance_ratio, acceptance_threshold=acceptance_threshold)
            if (acceptable_boolean):
                print("Subject " + str(subject_id) + f" is an acceptable candidate: {subject_classifications_dict}")
                accepted_subjects.append(subject_id)

        if(save):
            acceptable_candidates_dataframe = self.combineSubjectDataframes(self.getSubjectDataframe(accepted_subjects))
            Analyzer.saveSubjectDataframe(acceptable_candidates_dataframe, f"acceptable_candidates_acceptance_ratio_{acceptance_ratio}_acceptance_threshold_{acceptance_threshold}.csv")

        return accepted_subjects

    def checkAcceptableCandidates(self, accepted_subjects):
        not_in_simbad_subjects = []
        not_in_gaia_subjects = []
        not_in_either_subjects = []

        for index, subject_id in enumerate(accepted_subjects):
            print("Checking subject " + str(subject_id) + f" ({index + 1} out of {len(accepted_subjects)})")
            database_check_dict, database_query_dict = self.checkSubjectFieldOfView(subject_id)
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
                        if (database_name == "SIMBAD"):
                            print(f"Subject {subject_id} is in SIMBAD.")
                        elif (database_name == "Gaia"):
                            print(f"Subject {subject_id} is in Gaia.")

        not_in_simbad_subject_dataframes = self.getSubjectDataframe(not_in_simbad_subjects)
        not_in_gaia_subject_dataframes = self.getSubjectDataframe(not_in_gaia_subjects)
        not_in_either_subject_dataframes = self.getSubjectDataframe(not_in_either_subjects)

        not_in_simbad_subjects_dataframe = self.combineSubjectDataframes(not_in_simbad_subject_dataframes)
        not_in_gaia_subjects_dataframe = self.combineSubjectDataframes(not_in_gaia_subject_dataframes)
        not_in_either_subjects_dataframe = self.combineSubjectDataframes(not_in_either_subject_dataframes)

        self.saveSubjectDataframe(not_in_simbad_subjects_dataframe, "not_in_simbad_subjects.csv")
        self.saveSubjectDataframe(not_in_gaia_subjects_dataframe, "not_in_gaia_subjects.csv")
        self.saveSubjectDataframe(not_in_either_subjects_dataframe, "not_in_either_subjects.csv")

        generated_files = ["not_in_simbad_subjects.csv", "not_in_gaia_subjects.csv", "not_in_either_subjects.csv"]
        return generated_files

    def runAcceptableCandidateCheck(self, acceptance_ratio=0.5, acceptance_threshold=4, acceptable_candidates_csv=None):
        acceptable_candidates = []
        if (acceptable_candidates_csv is not None and os.path.exists(acceptable_candidates_csv)):
            print("Found acceptable candidates file.")
            acceptable_candidates_dataframe = self.loadSubjectDataframe(acceptable_candidates_csv)
            acceptable_candidates = acceptable_candidates_dataframe["subject_id"].values
        elif (acceptable_candidates_csv is None):
            print("No acceptable candidates file found. Generating new one.")
            acceptable_candidates = self.findAcceptableCandidates(acceptance_ratio=acceptance_ratio, acceptance_threshold=acceptance_threshold)
        elif (not os.path.exists(acceptable_candidates_csv)):
            raise FileNotFoundError(f"Cannot find acceptable candidates file: {acceptable_candidates_csv}")

        generated_files = self.checkAcceptableCandidates(acceptable_candidates)
        print("Generated files: " + str(generated_files))

    @uses_subject_ids
    @multioutput
    def searchSubjectFieldOfView(self, subject_id):

        FOV = 120 * u.arcsec

        database_check_dict = {"Simbad": self.sourceExistsInSimbad(subject_id, search_type="Box Search", FOV=FOV), "Gaia": self.sourceExistsInGaia(subject_id, search_type="Box Search", FOV=FOV)}

        # For each database, check if the subject's FOV search area has any known objects in it
        if(any(database_check_dict.values())):
            return True, database_check_dict
        else:
            return False, database_check_dict

    @uses_subject_ids
    @multioutput
    def checkSubjectFieldOfView(self, subject_id, otypes=["BD*", "BD?", "BrownD*", "BrownD?", "BrownD*_Candidate", "PM*"], gaia_pm=100 * u.mas / u.yr):

        simbad_query = self.getConditionalSimbadQuery(subject_id, search_type="Box Search", FOV=120*u.arcsec, otypes=otypes)
        gaia_query = self.getConditionalGaiaQuery(subject_id, search_type="Box Search", FOV=120*u.arcsec, gaia_pm=gaia_pm)

        database_query_dict = {"SIMBAD": simbad_query, "Gaia": gaia_query}
        database_check_dict = {}
        # Check each query to determine if it is empty or None
        for database, query in database_query_dict.items():
            if(query is None):
                database_check_dict[database] = False
            else:
                database_check_dict[database] = len(query) > 0

        return database_check_dict, database_query_dict

    def determineAcceptanceCounts(self, acceptance_ratio):

        # Get the subject IDs
        subject_ids = self.getSubjectIDs()

        # Initialize the success count dictionary
        success_count_dict = {}

        # Iterate through the subject IDs
        for subject_id in subject_ids:

            subject_type = self.getSubjectType(subject_id)

            # If the bitmask type is None, continue
            if subject_type is None:
                continue

            # If the bitmask type is not in the success count dictionary, add it
            if subject_type not in success_count_dict:
                success_count_dict[subject_type] = {"total": 0, "success": 0}

            # Increment the total count for the bitmask type
            success_count_dict[subject_type]["total"] += 1

            # Get the subject classifications
            subject_classifications = self.getSubjectClassifications(subject_id)

            # Count the number of successful classifications for each of the bitmask types
            total_classifications = subject_classifications["yes"] + subject_classifications["no"]
            movement_ratio = subject_classifications["yes"] / total_classifications
            non_movement_ratio = subject_classifications["no"] / total_classifications

            if(subject_type == "Known Brown Dwarf"):
                if(movement_ratio >= acceptance_ratio):
                    success_count_dict[subject_type]["success"] += 1
            elif(subject_type == "Quasar"):
                if(non_movement_ratio >= acceptance_ratio):
                    success_count_dict[subject_type]["success"] += 1
            elif(subject_type == "White Dwarf"):
                success_count_dict[subject_type]["success"] = None
            elif(subject_type == "SMDET Candidate"):
                success_count_dict[subject_type]["success"] = None
            elif(subject_type == "Random Sky Location"):
                if (non_movement_ratio >= acceptance_ratio):
                    success_count_dict[subject_type]["success"] += 1

        # Return the success count dictionary
        return success_count_dict

    @staticmethod
    def saveSubjectDataframe(subject_dataframe, filename):

        # Save the subject dataframe to a CSV file
        subject_dataframe.to_csv(filename, index=False)

    @staticmethod
    def loadSubjectDataframe(filename):

        # Load the subject dataframe from a CSV file
        subject_dataframe = pd.read_csv(filename)

        # Return the subject dataframe
        return subject_dataframe

    @uses_subject_ids
    def plotSubjectsSkyMap(self, subject_input: subjects_typing):
        subject_dataframes = self.getSubjectDataframe(subject_input)
        subject_dataframe = self.combineSubjectDataframes(subject_dataframes)
        subject_dataframe.to_csv("temp.csv", index=False)
        subject_csv_plotter = SubjectCSVPlotter("temp.csv")
        subject_csv_plotter.plot()
        os.remove("temp.csv")

    @uses_subject_ids
    def plotConditionalQueries(self, subject_input, database_name):
        subject_dataframes = self.getSubjectDataframe(subject_input)
        subject_dataframe = self.combineSubjectDataframes(subject_dataframes)

        for subject_id in subject_dataframe["subject_id"]:
            self.showSubject(subject_id, True)
            if (database_name == "Simbad"):
                query = self.getConditionalSimbadQuery(subject_id, FOV=120 * u.arcsec, separation=60 * u.arcsec, plot=True)
            elif (database_name == "Gaia"):
                query = self.getConditionalGaiaQuery(subject_id, FOV=120 * u.arcsec, separation=60 * u.arcsec, plot=True)
            elif (database_name == "not in either"):
                query = self.getConditionalSimbadQuery(subject_id, FOV=120 * u.arcsec, separation=60 * u.arcsec, plot=True)
                print("Simbad: ", query)
                query = self.getConditionalGaiaQuery(subject_id, FOV=120 * u.arcsec, separation=60 * u.arcsec, plot=True)
                print("Gaia: ", query)
            else:
                raise ValueError("Invalid database name.")


