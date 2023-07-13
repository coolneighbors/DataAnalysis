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

import panoptes_client
import unWISE_verse
from astropy.coordinates import SkyCoord
from unWISE_verse.Spout import Spout, check_login
import astropy.units as u
from astropy.time import Time
from typing import List, Dict, Tuple, Union, Optional, TextIO, Any, Callable, Iterable, Set
from Searcher import SimbadSearcher, GaiaSearcher
from Plotter import SubjectCSVPlotter
from Decorators import ignore_warnings, multioutput, plotting
# TODO: Add a weighting system for users' classifications based on their accuracy on the known subjects
# TODO: Check hashed IP's to see if they're the same user as a logged-in user for classification counts, etc.
# TODO: Redo documentation for Analyzer class and add type hints, result hints, and docstrings for all methods
# TODO: Investigate Gini_coefficient for classification distribution analysis (As suggested by Marc)
from numpy import ndarray

file_typing = Union[str, TextIOWrapper, TextIO]

# Input typing for @uses_subject_ids decorator
subjects_typing = Union[str, int, TextIOWrapper, TextIO, pd.DataFrame, Iterable[Union[str, int]]]

# Processes subject_input into a list of integer subject_ids
# TODO: Investigate if this can be done better (find first arg which is a list?, keyword arg for input?)

def uses_subject_ids(func):
    @functools.wraps(func)
    def conversion_wrapper(*args, **kwargs):
        subject_ids = []
        is_static_method = isinstance(func, staticmethod)
        is_class_method = isinstance(func, classmethod)

        if(kwargs.get("subject_input") is not None):
            raise ValueError("Keyword argument 'subject_input' is not allowed. Let subject_input be the first positional argument instead.")

        if (func.__qualname__.find(".") == -1 or func.__qualname__.find("<locals>") != -1):
            self_or_cls = None
            subject_input = args[0]
            args = args[1:]
        else:
            self_or_cls = args[0]
            subject_input = args[1]
            args = args[2:]

        if (subject_input is None):
            warnings.warn(f"Could not find any input for the subject(s) to convert to a list of subject ids. Returning empty list.")

        elif ((isinstance(subject_input, str) and os.path.exists(subject_input)) or isinstance(subject_input, TextIOWrapper) or isinstance(subject_input, TextIO)):
            subject_ids = pd.read_csv(Analyzer.verifyFile(subject_input, ".csv"), usecols=["subject_id"])["subject_id"].tolist()
        elif(isinstance(subject_input, str) or isinstance(subject_input, int) or isinstance(subject_input, np.int64)):
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
            subject_input_type = type(subject_input)
            if (not isinstance(subject_input, list) and not isinstance(subject_input, tuple)):
                warnings.warn(f"Subject type argument {subject_input} of type '{subject_input_type.__name__}' is not a list or tuple. Returning results as a list.")
                subject_input_type = list
            subject_ids = subject_input_type(subject_ids)
        else:
            raise TypeError("Invalid subjects type: " + str(type(subject_input)))

        if(self_or_cls is None):
            if(is_static_method or is_class_method):
                return func.__func__(subject_ids, *args, **kwargs)
            else:
                return func(subject_ids, *args, **kwargs)
        else:
            return func(self_or_cls, subject_ids, *args, **kwargs)

    return conversion_wrapper

def uses_user_identifiers(func):
    @functools.wraps(func)
    def conversion_wrapper(*args, **kwargs):
        is_static_method = isinstance(func, staticmethod)
        is_class_method = isinstance(func, classmethod)
        if (func.__qualname__.find(".") == -1 or func.__qualname__.find("<locals>") != -1):
            self_or_cls = None
            user_identifier = args[0]
            args = args[1:]
        else:
            self_or_cls = args[0]
            user_identifier = args[1]
            args = args[2:]

        if(user_identifier is None):
            raise ValueError("No user identifier provided.")

        try:
            user_identifier = int(user_identifier)
        except ValueError:
            pass

        if(self_or_cls is None):
            if(is_static_method or is_class_method):
                return func.__func__(user_identifier, *args, **kwargs)
            else:
                return func(user_identifier, *args, **kwargs)
        else:
            return func(self_or_cls, user_identifier, *args, **kwargs)
    return conversion_wrapper


def days_since_launch():
    launch_day = datetime.date(2023, 6, 27)
    today = datetime.date.today()
    return (today - launch_day).days + 1

class Analyzer:
    def __init__(self, extracted_file: file_typing, reduced_file: file_typing, subjects_file: Optional[file_typing] = None, save_login: bool = True) -> None:
        # Verify that the extracted_file is a valid file

        self.extracted_file, self.reduced_file= Analyzer.verifyFile((extracted_file, reduced_file), required_file_type=".csv")

        # Read the files as Pandas DataFrames
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

        self.classifier = Classifier(self)

        self.save()

    # Helper methods for Analyzer
    # ------------------------------------------------------------------------------------------------------------------

    def save(self, filename='analyzer.pickle'):
        """
        Save the Analyzer object as a pickle file.

        Parameters
        ----------
            filename : str
                A string representing the name of the file to save the Analyzer object as
                Defaults to 'analyzer.pickle'
        """

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename='analyzer.pickle'):
        """
        Load an Analyzer object from a pickle file.

        Parameters
        ----------
            filename : str
                A string representing the name of the file to load the Analyzer object from
                Defaults to 'analyzer.pickle'

        Returns
        -------
            analyzer : Analyzer
                The Analyzer object loaded from the pickle file
        """

        with open(filename, 'rb') as file:
            analyzer = pickle.load(file)
            return analyzer



    @staticmethod
    @multioutput
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

    # Methods related to classifications
    # ------------------------------------------------------------------------------------------------------------------
    # All classifications methods
    def getTotalClassifications(self):
        # Return the number of classifications
        return len(self.extracted_dataframe)

    @plotting
    def plotClassificationDistribution(self, total_subject_count=None, **kwargs):

        # Set the default title
        plt.title(f"Classification Distribution: Day {days_since_launch()}")

        # Set the default x and y labels
        plt.xlabel("Number of Classifications")
        plt.ylabel("Number of Subjects")

        number_of_classifications_dict = self.getCountDictionaryOfClassifications(total_subject_count)

        plt.bar(number_of_classifications_dict.keys(), number_of_classifications_dict.values(), **kwargs)

        for key in number_of_classifications_dict:
            plt.text(key, number_of_classifications_dict[key], number_of_classifications_dict[key], ha='center', va='bottom')

        plt.xticks(range(len(number_of_classifications_dict)))

    def computeTimeStatisticsForClassifications(self):

        # Get the unique usernames
        user_identifiers = self.getUniqueUserIdentifiers(user_identifier="username")

        # Initialize the list of classification times
        users_classification_times = []

        # Iterate over all unique usernames
        for user_identifier in user_identifiers:

            # Get the user's classifications
            user_classifications = self.getClassificationsByUser(user_identifier)

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
                        users_classification_times.append(time_difference.seconds)
                # Set the previous index to the current index
                previous_index = index

        # Compute the average time between classifications for all users
        users_average_time, users_std_time, users_median_time = self.computeTimeStatistics(users_classification_times)

        # Return the average time between classifications for all users
        return users_average_time, users_std_time, users_median_time

    def getCountDictionaryOfClassifications(self, total_subject_count=None):
        subject_ids = self.getSubjectIDs()
        number_of_classifications_dict = {}
        for subject_id in subject_ids:
            classification_dict = self.getClassificationsForSubject(subject_id)
            total_classifications = classification_dict['yes'] + classification_dict['no']
            if (total_classifications in number_of_classifications_dict):
                number_of_classifications_dict[total_classifications] += 1
            else:
                number_of_classifications_dict[total_classifications] = 1

        if (total_subject_count is not None):
            number_of_classifications_dict[0] = total_subject_count - len(subject_ids)
        number_of_classifications_dict = dict(sorted(number_of_classifications_dict.items()))
        return number_of_classifications_dict

    # Subset of classifications method
    def getSubsetTotalClassifications(self, minimum_count=0, total_subject_count=None):
        classification_count_dict = self.getCountDictionaryOfClassifications(total_subject_count)
        return sum(value for key, value in classification_count_dict.items() if key >= minimum_count)

    # User classification methods
    @uses_user_identifiers
    def getClassificationsByUser(self, user_identifier):
        # Check if the user_id is a string or an integer
        if (isinstance(user_identifier, str)):
            # If it's a string, then it's a username
            username = user_identifier
            return self.extracted_dataframe[self.extracted_dataframe["user_name"] == username]
        else:
            # If it's an integer, then it's a Zooniverse ID
            user_id = user_identifier
            return self.extracted_dataframe[self.extracted_dataframe["user_id"] == user_id]

    @multioutput
    @uses_user_identifiers
    def getTotalClassificationsByUser(self, user_identifier):
        # Return the number of classifications made by that user
        return len(self.getClassificationsByUser(user_identifier))

    # Top users classification methods
    def getTotalClassificationsByTopUsers(self, count_threshold=None, percentile=None):
        top_users_dict = self.getClassificationDictionaryOfTopUsers(count_threshold=count_threshold,
                                                                    percentile=percentile)
        return sum(top_users_dict.values())

    @plotting
    def plotTotalClassificationsByTopUsers(self, count_threshold=None, percentile=None, **kwargs):
        top_users_dict = self.getClassificationDictionaryOfTopUsers(count_threshold=count_threshold, percentile=percentile)
        # Plot the number of classifications made by each user but stop names from overlapping
        usernames = list(top_users_dict.keys())
        user_counts = list(top_users_dict.values())
        # Generate x-coordinates for the bars
        x = np.arange(len(top_users_dict))

        # Create the bar plot
        fig, ax = plt.subplots()

        # Set the default title
        if (percentile is not None):
            plt.title(f"Users in the Top {100 - percentile}% of Classifications")
        elif (count_threshold is not None):
            plt.title(f"Users with More Than {count_threshold} Classifications")

        # Set the default y label
        plt.ylabel("Number of Classifications", fontsize=15)

        bars = ax.bar(x, user_counts, **kwargs)

        for i, bar in enumerate(bars):
            # Display the user's count
            offset = 10
            ax.text(bar.get_x() + bar.get_width() / 2, user_counts[i] + offset, str(user_counts[i]), horizontalalignment='center', verticalalignment='bottom', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(usernames, ha='right', va='top', rotation=45, color="black")

    # Subject classification methods
    @multioutput
    @uses_subject_ids
    def getClassificationsForSubject(self, subject_input, weighted=False):

        if(not weighted):
            subject_dataframe = self.getSubjectDataframe(subject_input, dataframe_type="reduced")

            if (len(subject_dataframe) == 0):
                warnings.warn(f"Subject {subject_input} does not exist, returning None.")
                return None

            try:
                # Try to get the number of "yes" classifications
                yes_count = int(subject_dataframe["data.yes"].values[0])
            except ValueError:
                # If there are no "yes" classifications, then set the count to 0
                yes_count = 0

            try:
                # Try to get the number of "no" classifications
                no_count = int(subject_dataframe["data.no"].values[0])
            except ValueError:
                # If there are no "no" classifications, then set the count to 0
                no_count = 0

            classification_dict = {"yes": yes_count, "no": no_count, "total": yes_count + no_count}
            return classification_dict
        else:
            subject_dataframe = self.getSubjectDataframe(subject_input, dataframe_type="extracted")

            if (len(subject_dataframe) == 0):
                warnings.warn(f"Subject {subject_input} does not exist, returning None.")
                return None

            if (len(subject_dataframe) == 0):
                warnings.warn(f"Subject {subject_input} does not exist, returning None.")
                return None

            yes_count = 0
            no_count = 0
            for index, row in subject_dataframe.iterrows():
                default_insufficient_classifications = True
                try:
                    # Try to get the number of "yes" classifications
                    yes_count += int(row["data.yes"]) * self.classifier.getUserAccuracy(row["user_name"], default_insufficient_classifications=default_insufficient_classifications)
                except:
                    # If there are no "yes" classifications, then set the count to 0
                    yes_count += 0


                try:
                    # Try to get the number of "no" classifications
                    no_count += int(row["data.no"]) * self.classifier.getUserAccuracy(row["user_name"], default_insufficient_classifications=default_insufficient_classifications)
                except:
                    # If there are no "no" classifications, then set the count to 0
                    no_count += 0

        # Return the dictionary of the number of "yes" and "no" classifications
        classification_dict = {"yes": yes_count, "no": no_count, "total": yes_count + no_count}
        return classification_dict

    @multioutput
    @plotting
    @uses_subject_ids
    def plotClassificationsForSubject(self, subject_input, **kwargs):

        # Get the number of "yes" and "no" classifications for that subject as a dictionary
        classification_dict = self.getClassificationsForSubject(subject_input)
        # Get the number of "yes" and "no" classifications from the dictionary
        yes_count = classification_dict["yes"]
        no_count = classification_dict["no"]

        # Compute the total number of classifications
        total_count = yes_count + no_count

        # Compute the percentage of "yes" and "no" classifications
        yes_percent = yes_count / total_count
        no_percent = no_count / total_count

        labels = kwargs.pop("labels", ["Yes", "No"])
        autopct = kwargs.pop("autopct", '%1.1f%%')
        loc = kwargs.pop("loc", "upper left")

        # Plot the pie chart
        plt.pie([yes_percent, no_percent], labels=labels, autopct=autopct, **kwargs)
        plt.axis('equal')

        # Set the title
        plt.title("Subject ID: " + str(subject_input) + " Classifications")
        plt.legend([f"{yes_count} Yes classifications", f"{no_count} No classifications"], loc=loc)

    # Classification time statistics methods
    @multioutput
    @uses_user_identifiers
    def getClassificationTimesByUser(self, user_identifier):
        # Get the user's classifications
        user_classifications = self.getClassificationsByUser(user_identifier)
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

        if(len(user_classification_times) == 0 and len(user_times) > 0):
            warnings.warn(f"User with user_identifier '{user_identifier}' has too few consecutive classifications, returning an empty list.")
        elif(len(user_classification_times) == 0 and len(user_times) == 0):
            warnings.warn(f"User with user_identifier '{user_identifier}' has no classifications, returning an empty list.")

        return user_classification_times

    @multioutput
    @uses_user_identifiers
    def computeTimeStatisticsForUser(self, user_identifier):
        # Get the user's classification times
        user_classification_times = self.getClassificationTimesByUser(user_identifier)

        if (len(user_classification_times) == 0):
            raise ValueError(f"User with user_identifier {user_identifier} has no classifications or does not exist.")

        # Compute the average, standard deviation, and median of the user's classification times
        user_average_time, user_std_time, user_median_time = self.computeTimeStatistics(user_classification_times)

        # Return the average time between classifications for all users
        return user_average_time, user_std_time, user_median_time

    @staticmethod
    def computeTimeStatistics(classification_times):

        if(len(classification_times) == 0):
            raise ValueError(f"Classification times list is empty.")

        average_time = float(sum(classification_times) / len(classification_times))

        std_time = float(np.std(classification_times))

        median_time = float(np.median(classification_times))

        # Return the average time between classifications for a list of user(s) classification times
        return average_time, std_time, median_time

    @plotting
    def plotTimeHistogramForAllClassifications(self, **kwargs):

        bins = kwargs.pop("bins", 100)
        hist_range = kwargs.pop("range", None)

        # Get the unique usernames
        user_identifiers = self.getUniqueUserIdentifiers(user_identifier="username")

        all_classification_times = []

        # Iterate over all usernames
        for user_identifier in user_identifiers:
            # Get the user's classification times
            user_classification_times = self.getClassificationTimesByUser(user_identifier)

            # Convert the list of lists to a single list
            all_classification_times.extend(user_classification_times)

        # Compute the classification time statistics for all users
        all_average_time, all_std_time, all_median_time = self.computeTimeStatistics(all_classification_times)

        if (hist_range is None):
            hist_range = min(min(all_classification_times), all_average_time - all_std_time), max(max(all_classification_times), all_average_time + all_std_time)

        # Plot the histogram
        plt.hist(all_classification_times, bins=bins, range=hist_range, **kwargs)

        plt.axvline(all_average_time, color='red', linestyle='solid', linewidth=1, label=f"Average: {round(all_average_time, 2)} seconds")

        plt.axvline(all_average_time + all_std_time, color='red', linestyle='dashed', linewidth=1, label=f"Average ± Standard Deviation: {round(all_average_time, 2)} ± {round(all_std_time, 2)} seconds")
        plt.axvline(all_average_time - all_std_time, color='red', linestyle='dashed', linewidth=1)

        plt.axvline(all_median_time, color='orange', linestyle='solid', linewidth=1, label=f"Median: {round(all_median_time, 2)} seconds")
        plt.title("Classification Time Histogram")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Counts")
        plt.legend()

    @multioutput
    @plotting
    @uses_user_identifiers
    def plotTimeHistogramForUserClassifications(self, user_identifier, **kwargs):
        bins = kwargs.pop("bins", 100)
        hist_range = kwargs.pop("range", None)

        # Initialize the list of classification times
        user_classification_times = self.getClassificationTimesByUser(user_identifier)

        if(len(user_classification_times) == 0):
            warnings.warn(f"User with user_identifier '{user_identifier}' has no classification times, cannot plot their classification time histogram.")
            return None

        # Compute the classification time statistics for the user
        user_average_time, user_std_time, user_median_time = self.computeTimeStatistics(user_classification_times)

        if (hist_range is None):
            hist_range = min(min(user_classification_times), user_average_time - user_std_time), max(max(user_classification_times), user_average_time + user_std_time)

        # Plot the histogram
        plt.hist(user_classification_times, bins=bins, range=hist_range, **kwargs)

        plt.axvline(user_average_time, color='red', linestyle='solid', linewidth=1, label=f"Average: {round(user_average_time, 2)} seconds")

        plt.axvline(user_average_time + user_std_time, color='red', linestyle='dashed', linewidth=1, label=f"Average ± Standard Deviation: {round(user_average_time, 2)} ± {round(user_std_time, 2)} seconds")
        plt.axvline(user_average_time - user_std_time, color='red', linestyle='dashed', linewidth=1)

        plt.axvline(user_median_time, color='orange', linestyle='solid', linewidth=1, label=f"Median: {round(user_median_time, 2)} seconds")

        plt.title(f"{user_identifier} Classification Time Histogram")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Counts")
        plt.legend()

    @plotting
    def plotClassificationTimeline(self, bar=True, binning_parameter="Day", label=True, **kwargs):

        # Get the classification datetimes
        classification_datetimes = pd.to_datetime(self.extracted_dataframe["created_at"])

        # Initialize the binned datetimes dictionary
        binned_datetimes = {}

        # Iterate over all classification datetimes
        for classification_datetime in classification_datetimes:

            # Bin the datetimes
            if (binning_parameter == "Day"):
                day = classification_datetime.date()
                if day in binned_datetimes:
                    binned_datetimes[day].append(classification_datetime)
                else:
                    binned_datetimes[day] = [classification_datetime]
            elif (binning_parameter == "Week"):
                week = classification_datetime.isocalendar()[1]
                if week in binned_datetimes:
                    binned_datetimes[week].append(classification_datetime)
                else:
                    binned_datetimes[week] = [classification_datetime]
            elif (binning_parameter == "Month"):
                month = classification_datetime.month
                if month in binned_datetimes:
                    binned_datetimes[month].append(classification_datetime)
                else:
                    binned_datetimes[month] = [classification_datetime]
            elif (binning_parameter == "Year"):
                year = classification_datetime.year
                if year in binned_datetimes:
                    binned_datetimes[year].append(classification_datetime)
                else:
                    binned_datetimes[year] = [classification_datetime]

        # Convert the binned datetimes to a dictionary of counts
        binned_datetimes = {k: len(v) for k, v in binned_datetimes.items()}

        # Plot the timeline
        if (label):
            for key, value in binned_datetimes.items():
                plt.annotate(str(value), xy=(key, value), ha='center', va='bottom')

        if (bar):
            plt.bar(binned_datetimes.keys(), binned_datetimes.values(), **kwargs)
        else:
            plt.plot(binned_datetimes.keys(), binned_datetimes.values(), **kwargs)
        plt.title("Classification Timeline")
        plt.xlabel(binning_parameter)
        plt.ylabel("Count")

    # Methods related to users
    # ------------------------------------------------------------------------------------------------------------------
    # Principle method for getting users
    def getUniqueUserIdentifiers(self, include_logged_out_users=True, user_identifier="username"):
        # Note: I tried doing a cross-match between ip addresses and users, but there were no logged-out users
        # which had the same ip address as a logged-in user. So I don't think its worth the effort to implement.

        user_dataframe_key = None
        if(user_identifier == "username"):
            user_dataframe_key = "user_name"
        elif(user_identifier == "user id"):
            user_dataframe_key = "user_id"
        else:
            raise ValueError("user_identifier must be either 'username' or 'user id'.")

        if(include_logged_out_users and user_identifier == "user id"):
            raise ValueError("Cannot include logged out users when user_identifier is 'user id'.")

        if(include_logged_out_users):
            return list(self.extracted_dataframe[user_dataframe_key].unique())
        else:
            unique_users = list(self.extracted_dataframe[user_dataframe_key].unique())
            logged_in_unique_users = [user for user in unique_users if "not-logged-in" not in user]
            return logged_in_unique_users

    # Spout-interface method
    @multioutput
    @uses_user_identifiers
    def getUser(self, user_identifier):
        return Spout.get_user(user_identifier)

    # Methods related to top users
    def getClassificationDictionaryOfTopUsers(self, count_threshold=None, percentile=None):

        if (count_threshold is None and percentile is None):
            raise ValueError("You must provide either a count_threshold or a percentile.")
        elif(count_threshold is not None and percentile is not None):
            raise ValueError("You cannot provide both a count_threshold and a percentile.")

        unique_user_identifiers = self.getUniqueUserIdentifiers(user_identifier="username")
        user_classifications_dict = {}

        for unique_user_identifier in unique_user_identifiers:
            user_classifications_dict[unique_user_identifier] = self.getTotalClassificationsByUser(unique_user_identifier)

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

    def getTopUsernames(self, count_threshold=None, percentile=None):
        top_usernames_dict = self.getClassificationDictionaryOfTopUsers(count_threshold=count_threshold, percentile=percentile)
        return list(top_usernames_dict.keys())

    def getTopUsernamesCount(self, count_threshold=None, percentile=None):
        top_users_dict = self.getClassificationDictionaryOfTopUsers(count_threshold=count_threshold,percentile=percentile)
        return len(top_users_dict)

    def getTopUsers(self, count_threshold=None, percentile=None):
        top_usernames = self.getTopUsernames(count_threshold=count_threshold, percentile=percentile)
        return self.getUser(top_usernames)

    # Methods related to subjects and subject metadata
    # ------------------------------------------------------------------------------------------------------------------
    # Principle method for getting subjects
    def getSubjectIDs(self):
        # Return the list of subject IDs
        return [int(subject_id) for subject_id in self.reduced_dataframe["subject_id"].values]

    # Spout-interface method
    @multioutput
    @uses_subject_ids
    def getSubject(self, subject_input) -> Union[panoptes_client.Subject, List[panoptes_client.Subject]]:
        # Get the subject with the given subject ID in the subject set with the given subject set ID
        return Spout.get_subject(subject_input, self.subject_set_id)

    # Subject dataframe methods
    @multioutput
    @uses_subject_ids
    def getSubjectDataframe(self, subject_input: subjects_typing, dataframe_type: str = "default") -> Union[pd.DataFrame, List[pd.DataFrame]]:

        @multioutput
        @uses_subject_ids
        def getSubjectDataframeFromID(subject_input, dataframe_type: str = "default") -> pd.DataFrame:

            if(not isinstance(subject_input, int)):
                raise ValueError("The subject_id must be an integer.")

            if (dataframe_type == "default"):
                # If it is default, then return the metadata of the subject as a dataframe
                subject_metadata = self.getSubjectMetadata(subject_input)

                if(subject_metadata is None):
                    warnings.warn(f"Subject {subject_input} does not exist, returning empty Dataframe.")
                    return pd.DataFrame()

                # Add the standard subject_id column
                subject_metadata["subject_id"] = subject_input

                # Add the standard metadata column
                subject_metadata["metadata"] = str(subject_metadata)

                # Convert the metadata to a dataframe
                subject_metadata_dataframe = pd.DataFrame.from_dict(subject_metadata, orient="index").transpose()
                return subject_metadata_dataframe

            elif (dataframe_type == "reduced"):
                # If reduced, then return the reduced dataframe for that subject
                return self.reduced_dataframe[self.reduced_dataframe["subject_id"] == subject_input]
            elif (dataframe_type == "extracted"):
                # If not reduced, then return the extracted dataframe for that subject
                return self.extracted_dataframe[self.extracted_dataframe["subject_id"] == subject_input]

        return getSubjectDataframeFromID(subject_input, dataframe_type=dataframe_type)

    @staticmethod
    def combineSubjectDataframes(subject_dataframes: Iterable[pd.DataFrame]) -> pd.DataFrame:
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

    # Subject metadata methods
    @multioutput
    @uses_subject_ids
    def checkIfSubjectExists(self, subject_input):

        # Get the subject's metadata

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metadata = self.getSubjectMetadata(subject_input)
            return metadata is not None

    @multioutput
    @uses_subject_ids
    def getSubjectMetadata(self, subject_input):

        # Get the subject with the given subject ID
        if(self.subjects_file is not None):
            subject_dataframe = self.subjects_dataframe[self.subjects_dataframe["subject_id"] == subject_input]
            if(subject_dataframe.empty):
                warnings.warn(f"Subject ID {subject_input} was not found in subjects file, {self.subjects_file}: Returning None")
                return None
            else:
                return eval(self.subjects_dataframe[self.subjects_dataframe["subject_id"] == subject_input].iloc[0]["metadata"])
        else:
            subject = self.getSubject(subject_input)

            try:
                return subject.metadata
            except AttributeError:
                warnings.warn(f"Subject ID {subject_input} was not found: Returning None")
                return None

    @multioutput
    @uses_subject_ids
    def getSubjectMetadataField(self, subject_input, field_name):
        # Get the subject metadata for the subject with the given subject ID
        subject_metadata = self.getSubjectMetadata(subject_input)

        if (subject_metadata is None):
            warnings.warn(f"Subject ID {subject_input} was not found: Returning None")
            return None

        # Get the metadata field with the given field name
        field_value = subject_metadata.get(field_name, None)

        if (field_value is None):
            warnings.warn(f"Field name {field_name} was not found in subject ID {subject_input}'s metadata: Returning None")
            return None
        else:
            return field_value

    # Subject WiseView image method
    @multioutput
    @uses_subject_ids
    def showSubjectInWiseView(self, subject_input, open_in_browser=False):

        # Get the WiseView link for the subject with the given subject ID
        wise_view_link = self.getSubjectMetadataField(subject_input, "WISEVIEW")

        if wise_view_link is None:
            warnings.warn(f"No WiseView link found for subject ID {subject_input}: Returning None")
            return None

        # Remove the WiseView link prefix and suffix
        wise_view_link = wise_view_link.removeprefix("[WiseView](+tab+")
        wise_view_link = wise_view_link.removesuffix(")")

        # Determine whether to open the subject in the default web browser

        if (open_in_browser):
            webbrowser.open(wise_view_link)
            delay = 10
            print(f"Opening WiseView link for subject ID {subject_input}.")
            sleep(delay)

        # Return the WiseView link
        return wise_view_link

    # Subject SIMBAD link method
    @multioutput
    @uses_subject_ids
    def getSimbadLinkForSubject(self, subject_input):

        simbad_link = self.getSubjectMetadataField(subject_input, "SIMBAD")

        if (simbad_link is None):
            warnings.warn(f"No SIMBAD link found for subject ID {subject_input}: Returning None")
            return None

            # Remove the SIMBAD link prefix and suffix
        simbad_link = simbad_link.removeprefix("[SIMBAD](+tab+")
        simbad_link = simbad_link.removesuffix(")")

        return simbad_link

    # Methods related to database queries
    # ------------------------------------------------------------------------------------------------------------------
    # SIMBAD query methods
    @multioutput
    @uses_subject_ids
    def getSimbadQueryForSubject(self, subject_input, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec, plot=False, separation=None):
        subject_metadata = self.getSubjectMetadata(subject_input)

        if(subject_metadata is None):
            warnings.warn(f"Subject ID {subject_input} was not found, so a SIMBAD query cannot be performed: Returning None")
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

    @multioutput
    @uses_subject_ids
    def getConditionalSimbadQueryForSubject(self, subject_input, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec, otypes=["BD*", "BD?", "BrownD*", "BrownD?", "BrownD*_Candidate", "PM*"], plot=False, separation=None):
        subject_metadata = self.getSubjectMetadata(subject_input)

        if (subject_metadata is None):
            warnings.warn(f"Subject ID {subject_input} was not found, so a conditional SIMBAD query cannot be performed: Returning None")
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

    @multioutput
    @uses_subject_ids
    def sourceExistsInSimbadForSubject(self, subject_input, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec):

        return len(self.getSimbadQueryForSubject(subject_input, search_type=search_type, FOV=FOV, radius=radius)) > 0

    # Gaia query methods
    @multioutput
    @uses_subject_ids
    def getGaiaQueryForSubject(self, subject_input, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec, plot=False, separation=None):
        # Get the subject's metadata
        subject_metadata = self.getSubjectMetadata(subject_input)

        if (subject_metadata is None):
            warnings.warn(f"Subject ID {subject_input} was not found, so a Gaia query cannot be performed: Returning None")
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

    @multioutput
    @uses_subject_ids
    def getConditionalGaiaQueryForSubject(self, subject_input, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec, gaia_pm=100 * u.mas / u.yr, plot=False, separation=None):
        subject_metadata = self.getSubjectMetadata(subject_input)

        if (subject_metadata is None):
            warnings.warn(f"Subject ID {subject_input} was not found, so a conditional Gaia query cannot be performed: Returning None")
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

    @multioutput
    @uses_subject_ids
    def sourceExistsInGaiaForSubject(self, subject_input, search_type="Box Search", FOV=120*u.arcsec, radius=60*u.arcsec):

        return len(self.getGaiaQueryForSubject(subject_input, search_type=search_type, FOV=FOV, radius=radius)) > 0

    # Methods related to selecting acceptable subjects as candidates for review
    # ------------------------------------------------------------------------------------------------------------------
    # Subject type methods
    @staticmethod
    @multioutput
    def bitmaskToSubjectType(bitmask):

        # Convert the bitmask to an integer
        try:
            bitmask = int(bitmask)
        except ValueError:
            raise ValueError("bitmask must be an integer or a string that can be converted to an integer.")

        # Initialize the bitmask dictionary
        bitmask_dict = {2**0: "SMDET Candidate", 2**1: "Blank", 2**2: "Known Brown Dwarf", 2**3: "Quasar", 2**4: "Random Sky Location", 2**5: "White Dwarf"}

        # Return the bitmask type associated with the bitmask value
        return bitmask_dict.get(bitmask, None)

    @multioutput
    @uses_subject_ids
    def getSubjectType(self, subject_input):
        # Get the bitmask for the subject
        bitmask = self.getSubjectMetadataField(subject_input, "#BITMASK")

        if(bitmask is None):
            warnings.warn(f"Subject ID {subject_input} was not found, so a subject type cannot be determined: Returning None")
            return None

        # Convert the bitmask to a subject type
        subject_type = Analyzer.bitmaskToSubjectType(bitmask)

        return subject_type

    # Acceptable candidate methods
    @multioutput
    @uses_subject_ids
    def checkIfCandidateIsAcceptable(self, subject_input, acceptance_ratio, acceptance_threshold=0, weighted=False):
        subject_type = self.getSubjectType(subject_input)
        subject_classifications = self.getClassificationsForSubject(subject_input, weighted=weighted)

        if (subject_classifications is None):
            warnings.warn(f"Subject ID {subject_input} was not found, so an acceptable candidate cannot be determined: Returning None")
            return None, None

        # Count the number of successful classifications for each of the bitmask types

        total_classifications = subject_classifications["total"]
        movement_ratio = subject_classifications["yes"] / total_classifications
        non_movement_ratio = subject_classifications["no"] / total_classifications
        print(f"Movement ratio: {movement_ratio}", "Subject Type: ", subject_type)
        if (subject_type == "SMDET Candidate"):
            return (movement_ratio > acceptance_ratio) and (subject_classifications["yes"] > acceptance_threshold), subject_classifications
        else:
            return False, subject_classifications

    def findAcceptableCandidates(self, acceptance_ratio=None, acceptance_threshold=None, weighted=False, save=True):
        subject_ids = self.getSubjectIDs()
        accepted_subjects = []

        for subject_id in subject_ids:
            acceptable_boolean, subject_classifications_dict = self.checkIfCandidateIsAcceptable(subject_id, acceptance_ratio=acceptance_ratio, acceptance_threshold=acceptance_threshold, weighted=weighted)

            if(acceptable_boolean is None or subject_classifications_dict is None):
                warnings.warn(f"Subject ID {subject_id} was not found, so it cannot be determined if it is an acceptable candidate: Skipping")
                continue

            if (acceptable_boolean):
                print("Subject " + str(subject_id) + f" is an acceptable candidate: {subject_classifications_dict}")
                accepted_subjects.append(subject_id)

        if(save):
            acceptable_candidates_dataframe = self.combineSubjectDataframes(self.getSubjectDataframe(accepted_subjects))
            Analyzer.saveSubjectDataframeToFile(acceptable_candidates_dataframe,
                                                f"acceptable_candidates_acceptance_ratio_{acceptance_ratio}_acceptance_threshold_{acceptance_threshold}.csv")

        return accepted_subjects

    def sortAcceptableCandidatesByDatabase(self, accepted_subjects):
        not_in_simbad_subjects = []
        not_in_gaia_subjects = []
        not_in_either_subjects = []

        for index, subject_id in enumerate(accepted_subjects):
            print("Checking subject " + str(subject_id) + f" ({index + 1} out of {len(accepted_subjects)})")
            database_check_dict, database_query_dict = self.checkSubjectFOV(subject_id)
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

        self.saveSubjectDataframeToFile(not_in_simbad_subjects_dataframe, "not_in_simbad_subjects.csv")
        self.saveSubjectDataframeToFile(not_in_gaia_subjects_dataframe, "not_in_gaia_subjects.csv")
        self.saveSubjectDataframeToFile(not_in_either_subjects_dataframe, "not_in_either_subjects.csv")

        generated_files = ["not_in_simbad_subjects.csv", "not_in_gaia_subjects.csv", "not_in_either_subjects.csv"]
        return generated_files

    # Principle method for getting acceptable candidates and sorting them by database
    def performCandidatesSort(self, acceptance_ratio=0.5, acceptance_threshold=4, weighted=False, acceptable_candidates_csv=None):
        acceptable_candidates = []
        if (acceptable_candidates_csv is not None and os.path.exists(acceptable_candidates_csv)):
            print("Found acceptable candidates file.")
            acceptable_candidates_dataframe = self.loadSubjectDataframeFromFile(acceptable_candidates_csv)
            acceptable_candidates = acceptable_candidates_dataframe["subject_id"].values
        elif (acceptable_candidates_csv is None):
            print("No acceptable candidates file found. Generating new one.")
            acceptable_candidates = self.findAcceptableCandidates(acceptance_ratio=acceptance_ratio, acceptance_threshold=acceptance_threshold, weighted=weighted)
        elif (not os.path.exists(acceptable_candidates_csv)):
            raise FileNotFoundError(f"Cannot find acceptable candidates file: {acceptable_candidates_csv}")

        generated_files = self.sortAcceptableCandidatesByDatabase(acceptable_candidates)
        print("Generated files: " + str(generated_files))

    # FOV search and checking methods
    @multioutput
    @uses_subject_ids
    def searchSubjectFOV(self, subject_input):

        FOV = 120 * u.arcsec

        database_check_dict = {"Simbad": self.sourceExistsInSimbadForSubject(subject_input, search_type="Box Search", FOV=FOV), "Gaia": self.sourceExistsInGaiaForSubject(subject_input, search_type="Box Search", FOV=FOV)}

        # For each database, check if the subject's FOV search area has any known objects in it
        if(any(database_check_dict.values())):
            return True, database_check_dict
        else:
            return False, database_check_dict

    @multioutput
    @uses_subject_ids
    def checkSubjectFOV(self, subject_input, otypes=["BD*", "BD?", "BrownD*", "BrownD?", "BrownD*_Candidate", "PM*"], gaia_pm=100 * u.mas / u.yr):

        simbad_query = self.getConditionalSimbadQueryForSubject(subject_input, search_type="Box Search", FOV=120 * u.arcsec, otypes=otypes)
        gaia_query = self.getConditionalGaiaQueryForSubject(subject_input, search_type="Box Search", FOV=120 * u.arcsec, gaia_pm=gaia_pm)

        database_query_dict = {"SIMBAD": simbad_query, "Gaia": gaia_query}
        database_check_dict = {}
        # Check each query to determine if it is empty or None
        for database, query in database_query_dict.items():
            if(query is None):
                database_check_dict[database] = False
            else:
                database_check_dict[database] = len(query) > 0

        return database_check_dict, database_query_dict

    # Acceptance counts method
    def calculateAcceptanceCountsBySubjectType(self, acceptance_ratio):

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
            subject_classifications = self.getClassificationsForSubject(subject_id)

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

    # Helper methods for saving and loading subject dataframes as needed
    @staticmethod
    def saveSubjectDataframeToFile(subject_dataframe, filename):

        # Save the subject dataframe to a CSV file
        subject_dataframe.to_csv(filename, index=False)

    @staticmethod
    def loadSubjectDataframeFromFile(filename):

        # Load the subject dataframe from a CSV file
        subject_dataframe = pd.read_csv(filename)

        # Return the subject dataframe
        return subject_dataframe

    # Methods related to plotting subjects
    # ------------------------------------------------------------------------------------------------------------------
    @uses_subject_ids
    def plotSkyMapForSubjects(self, subject_input: subjects_typing):
        subject_dataframes = self.getSubjectDataframe(subject_input)
        subject_dataframe = self.combineSubjectDataframes(subject_dataframes)
        subject_dataframe.to_csv("temp.csv", index=False)
        subject_csv_plotter = SubjectCSVPlotter("temp.csv")
        subject_csv_plotter.plot()
        os.remove("temp.csv")

    @multioutput
    @uses_subject_ids
    def plotQueryForSubject(self, subject_input, database_name, show_in_wiseview=False):
        subject_dataframe = self.getSubjectDataframe(subject_input)

        for subject_id in subject_dataframe["subject_id"]:
            if(show_in_wiseview):
                self.showSubjectInWiseView(subject_id, True)
            if (database_name.lower() == "simbad"):
                query = self.getSimbadQueryForSubject(subject_id, FOV=120 * u.arcsec, plot=True, separation=60 * u.arcsec)
                print("Simbad: ", query)
            elif (database_name.lower() == "gaia"):
                query = self.getGaiaQueryForSubject(subject_id, FOV=120 * u.arcsec, plot=True, separation=60 * u.arcsec)
                print("Gaia: ", query)
            elif (database_name.lower() == "not in either"):
                query = self.getSimbadQueryForSubject(subject_id, FOV=120 * u.arcsec, plot=True, separation=60 * u.arcsec)
                print("Simbad: ", query)
                query = self.getGaiaQueryForSubject(subject_id, FOV=120 * u.arcsec, plot=True, separation=60 * u.arcsec)
                print("Gaia: ", query)
            else:
                raise ValueError("Invalid database name.")

    @multioutput
    @uses_subject_ids
    def plotConditionalQueriesForSubject(self, subject_input, database_name):
        subject_dataframe = self.getSubjectDataframe(subject_input)

        for subject_id in subject_dataframe["subject_id"]:
            self.showSubjectInWiseView(subject_id, True)
            if (database_name == "Simbad"):
                query = self.getConditionalSimbadQueryForSubject(subject_id, FOV=120 * u.arcsec, plot=True,
                                                                 separation=60 * u.arcsec)
            elif (database_name == "Gaia"):
                query = self.getConditionalGaiaQueryForSubject(subject_id, FOV=120 * u.arcsec, plot=True,
                                                               separation=60 * u.arcsec)
            elif (database_name == "not in either"):
                query = self.getConditionalSimbadQueryForSubject(subject_id, FOV=120 * u.arcsec, plot=True,
                                                                 separation=60 * u.arcsec)
                print("Simbad: ", query)
                query = self.getConditionalGaiaQueryForSubject(subject_id, FOV=120 * u.arcsec, plot=True,
                                                               separation=60 * u.arcsec)
                print("Gaia: ", query)
            else:
                raise ValueError("Invalid database name.")


class Classifier:

    insufficient_classifications_default_accuracy = 0.5

    def __init__(self, analyzer: Analyzer):
        self.user_information = {}
        self.analyzer = analyzer
        print("Calculating user performances...")
        ignore_warnings(self.calculateAllUserPerformances)(include_logged_out_users=True)
        print("User performances calculated.")

    @multioutput
    @uses_user_identifiers
    def getUserAccuracy(self, user_identifier, default_insufficient_classifications=True):
        if (user_identifier not in self.user_information):
            self.calculateUserPerformance(user_identifier)

        user_accuracy = self.user_information[user_identifier]["Performance"]["Accuracy"]
        if (default_insufficient_classifications and user_accuracy is None):
            return self.insufficient_classifications_default_accuracy
        else:
            return user_accuracy


    @multioutput
    @uses_user_identifiers
    def getUserVerifiedClassifications(self, user_identifier):
        if (user_identifier not in self.user_information):
            self.calculateUserPerformance(user_identifier)
        return self.user_information[user_identifier]["Classifications"]

    @multioutput
    @uses_user_identifiers
    def getUserInformation(self, user_identifier, default_insufficient_classifications=True):
        if (user_identifier not in self.user_information):
            self.calculateUserPerformance(user_identifier)

        user_accuracy = self.user_information[user_identifier]["Performance"]["Accuracy"]
        if(default_insufficient_classifications and user_accuracy is None):
            modified_user_information = copy(self.user_information[user_identifier])
            modified_user_information["Performance"]["Accuracy"] = self.insufficient_classifications_default_accuracy
            return modified_user_information
        else:
            return self.user_information[user_identifier]

    @multioutput
    @uses_user_identifiers
    def calculateUserPerformance(self, user_identifier):
        user_classifications_dataframe = self.analyzer.getClassificationsByUser(user_identifier)
        user_information_dictionary = {"Classifications": {}}

        verified_subject_types = {"Known Brown Dwarf": True, "Quasar": False, "Random Sky Location": False}
        verified_subject_performance_scales = {"Known Brown Dwarf": 1, "Quasar": 1, "Random Sky Location": 1}
        for verified_subject_type in verified_subject_types:
            user_information_dictionary["Classifications"][verified_subject_type] = {"total": 0, "success": 0, "failure": 0}

        for index in user_classifications_dataframe.index:
            subject_id = user_classifications_dataframe["subject_id"][index]

            subject_type = self.analyzer.getSubjectType(subject_id)

            if (subject_type is None):
                warnings.warn("Subject type is None for subject ID: " + str(subject_id))
                continue

            try:
                # Try to get the number of "yes" classifications
                yes_count = int(user_classifications_dataframe["data.yes"][index])
            except ValueError:
                # If there are no "yes" classifications, then set the count to 0
                yes_count = 0

            try:
                # Try to get the number of "no" classifications
                no_count = int(user_classifications_dataframe["data.no"][index])
            except ValueError:
                # If there are no "no" classifications, then set the count to 0
                no_count = 0

            classification_dict = {"yes": bool(yes_count), "no": bool(no_count)}

            if(subject_type not in verified_subject_types):
                continue
            else:
                user_information_dictionary["Classifications"][subject_type]["total"] += 1
                should_be_mover = verified_subject_types[subject_type]

                if(should_be_mover == classification_dict["yes"]):
                    user_information_dictionary["Classifications"][subject_type]["success"] += 1
                else:
                    user_information_dictionary["Classifications"][subject_type]["failure"] += 1

        if(user_identifier not in self.user_information):
            # TODO: Investigate Cohen's kappa coefficient for binary classification
            # https://en.wikipedia.org/wiki/Cohen%27s_kappa#:~:text=Cohen's%20kappa%20coefficient%20(%CE%BA%2C%20lowercase,for%20qualitative%20(categorical)%20items.
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/#:~:text=Cohen%20suggested%20the%20Kappa%20result,1.00%20as%20almost%20perfect%20agreement.
            user_information_dictionary["Performance"] = {"Accuracy": 0.0}
            self.user_information[user_identifier] = user_information_dictionary

            total_classifications = 0
            for subject_type in user_information_dictionary["Classifications"]:
                total_classifications += user_information_dictionary["Classifications"][subject_type]["total"]

            # Calculate the accuracy
            if (total_classifications == 0):
                warnings.warn(f"{user_identifier} has no valid classifications for performance. Setting accuracy as None")
                self.user_information[user_identifier]["Performance"]["Accuracy"] = None
            else:
                successful_ratio_total = 0.0
                subject_type_scaling_total = 0.0
                for subject_type in verified_subject_types:
                    if(user_information_dictionary["Classifications"][subject_type]["total"] == 0):
                        continue
                    elif(verified_subject_performance_scales[subject_type] != 0):
                        successful_ratio_total += float(verified_subject_performance_scales[subject_type]) * float(user_information_dictionary["Classifications"][subject_type]["success"]) / float(user_information_dictionary["Classifications"][subject_type]["total"])
                        subject_type_scaling_total += float(verified_subject_performance_scales[subject_type])
                self.user_information[user_identifier]["Performance"]["Accuracy"] = successful_ratio_total / subject_type_scaling_total
        else:
            raise ValueError(f"User identifier {user_identifier} already exists in user performance dictionary.")

    def calculateAllUserPerformances(self, include_logged_out_users=False):
        user_identifiers = self.analyzer.getUniqueUserIdentifiers(user_identifier="username", include_logged_out_users=include_logged_out_users)
        self.calculateUserPerformance(user_identifiers)

    def getAllUserAccuracies(self, include_logged_out_users=False, default_insufficient_classifications=True):
        user_identifiers = self.analyzer.getUniqueUserIdentifiers(user_identifier="username", include_logged_out_users=include_logged_out_users)
        return self.getUserAccuracy(user_identifiers, default_insufficient_classifications=default_insufficient_classifications)

    def getAllUserInformation(self, include_logged_out_users=False, default_insufficient_classifications=True):
        user_identifiers = self.analyzer.getUniqueUserIdentifiers(user_identifier="username", include_logged_out_users=include_logged_out_users)
        user_information_dictionaries = self.getUserInformation(user_identifiers, default_insufficient_classifications=default_insufficient_classifications)
        return {user_identifier: user_information_dictionary for user_identifier, user_information_dictionary  in zip(user_identifiers, user_information_dictionaries)}

    def getMostAccurateUsernames(self, include_logged_out_users=False, default_insufficient_classifications=True, classification_minimum=0, verified_classifications_minimum=0, accuracy_threshold=0.0):
        user_information_dictionaries = copy(self.getAllUserInformation(include_logged_out_users=include_logged_out_users, default_insufficient_classifications=default_insufficient_classifications))
        new_user_information_dictionaries = copy(user_information_dictionaries)
        for user_identifier in user_information_dictionaries:
            total_classifications = self.analyzer.getTotalClassificationsByUser(user_identifier)
            total_verified_classifications = 0
            for subject_type in user_information_dictionaries[user_identifier]["Classifications"]:
                total_verified_classifications += user_information_dictionaries[user_identifier]["Classifications"][subject_type]["total"]

            if(total_classifications < classification_minimum):
                del new_user_information_dictionaries[user_identifier]
            elif(total_verified_classifications < verified_classifications_minimum):
                del new_user_information_dictionaries[user_identifier]
            elif(user_information_dictionaries[user_identifier]["Performance"]["Accuracy"] is None):
                del new_user_information_dictionaries[user_identifier]
            elif(user_information_dictionaries[user_identifier]["Performance"]["Accuracy"] < accuracy_threshold):
                del new_user_information_dictionaries[user_identifier]

        most_accurate_usernames = [user_identifier for user_identifier in sorted(new_user_information_dictionaries, key=lambda x: new_user_information_dictionaries[x]["Performance"]["Accuracy"], reverse=True)]
        return most_accurate_usernames


    @multioutput
    @plotting
    @uses_user_identifiers
    def plotUserPerformance(self, user_identifier, **kwargs):
        user_accuracy = self.getUserAccuracy(user_identifier)

        if(user_accuracy is None):
            warnings.warn(f"{user_identifier} has no valid classifications for performance. Skipping plot.")
            return None

        user_classifications = self.getUserVerifiedClassifications(user_identifier)
        user_classification_dataframe = pd.DataFrame.from_dict(user_classifications,  orient="index")
        user_classification_dataframe = user_classification_dataframe.rename(columns={"success": "Success", "failure": "Failure", "total": "Total"})
        formatted_accuracy = round(100 * user_accuracy, 2)
        user_classification_dataframe.plot.bar(y=["Success", "Failure"], stacked=True, title=f"User Accuracy for {user_identifier}: {formatted_accuracy}%", **kwargs)

        # Put the success percentage on the top of each bar
        index = 0
        for subject_type, subject_type_row in user_classification_dataframe.iterrows():
            total = subject_type_row["Total"]
            success = subject_type_row["Success"]
            if(total != 0):
                plt.text(index, success, f"{round(100 * success / total, 2)}%", ha="center", va="bottom")
            index += 1

        plt.xticks(rotation=0)
        plt.xlabel("Subject Type")
        plt.ylabel("Number of Classifications")

    def plotAllUsersPerformanceHistogram(self, include_logged_out_users=False, default_insufficient_classifications=True, **kwargs):
        user_accuracies = self.getAllUserAccuracies(include_logged_out_users=include_logged_out_users, default_insufficient_classifications=default_insufficient_classifications)
        plt.title("User Performance Histogram")
        self.plotPerformanceHistogram(user_accuracies, **kwargs)

    def plotTopUsersPerformanceHistogram(self, count_threshold=None, percentile=None, default_insufficient_classifications=True, **kwargs):
        top_usernames = self.analyzer.getTopUsernames(count_threshold=count_threshold, percentile=percentile)
        print(f"Plotting performance histogram for {len(top_usernames)} users.")
        top_user_accuracies = self.getUserAccuracy(top_usernames, default_insufficient_classifications=default_insufficient_classifications)
        plt.title("Top User Performance Histogram")
        self.plotPerformanceHistogram(top_user_accuracies, **kwargs)

    @staticmethod
    @plotting
    def plotPerformanceHistogram(accuracies, **kwargs):
        # Remove None accuracies
        accuracy_values = [x for x in accuracies if x is not None]
        bins = kwargs.pop("bins", 20)
        plt.hist(accuracy_values, bins=bins, **kwargs)
        plt.xlabel("User Accuracy", fontsize=14)
        plt.ylabel("Number of Users", fontsize=14)

    @plotting
    def plotTopUsersPerformances(self, count_threshold=None, percentile=None, default_insufficient_classifications=True, **kwargs):
        top_usernames = self.analyzer.getTopUsernames(count_threshold=count_threshold, percentile=percentile)
        print(f"Plotting performance for {len(top_usernames)} users.")

        top_user_accuracies = self.getUserAccuracy(top_usernames, default_insufficient_classifications=default_insufficient_classifications)

        # Sort the accuracies and usernames by accuracy
        top_user_accuracies, top_usernames = zip(*sorted(zip(top_user_accuracies, top_usernames), reverse=True))

        # Generate x-coordinates for the bars
        x = np.arange(len(top_usernames))

        # Create the bar plot
        fig, ax = plt.subplots()

        # Set the default title
        if (percentile is not None):
            plt.title(f"Users in the Top {100 - percentile}% of Classifications")
        elif (count_threshold is not None):
            plt.title(f"Users with More Than {count_threshold} Classifications")

        # Set the default y label
        plt.ylabel("User Accuracy", fontsize=15)

        bars = ax.bar(x, top_user_accuracies, **kwargs)
        for i, bar in enumerate(bars):
            # Display the user's accuracy above the bar
            offset = 0.01
            ax.text(bar.get_x() + bar.get_width() / 2, top_user_accuracies[i] + offset, f"{round(100*top_user_accuracies[i], 2)}%", horizontalalignment='center', verticalalignment='bottom', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(top_usernames, ha='right', va='top', rotation=45, color="black")

    @plotting
    def plotMostAccurateUsers(self,  include_logged_out_users=False, default_insufficient_classifications=True, classification_minimum=0, verified_classifications_minimum=0, accuracy_threshold=0.0, **kwargs):
        most_accurate_users = self.getMostAccurateUsernames(include_logged_out_users=include_logged_out_users, default_insufficient_classifications=default_insufficient_classifications, classification_minimum=classification_minimum, verified_classifications_minimum=verified_classifications_minimum, accuracy_threshold=accuracy_threshold)
        print(f"Plotting performance for {len(most_accurate_users)} users.")

        most_accurate_users_accuracies = self.getUserAccuracy(most_accurate_users, default_insufficient_classifications=default_insufficient_classifications)

        # Generate x-coordinates for the bars
        x = np.arange(len(most_accurate_users_accuracies))

        # Create the bar plot
        fig, ax = plt.subplots()

        # Set the default title
        plt.title(f"Most Accurate Users")

        # Set the default y label
        plt.ylabel("User Accuracy", fontsize=15)

        bars = ax.bar(x, most_accurate_users_accuracies, **kwargs)
        for i, bar in enumerate(bars):
            # Display the user's accuracy above the bar
            offset = 0.01
            ax.text(bar.get_x() + bar.get_width() / 2, most_accurate_users_accuracies[i] + offset, f"{round(100 * most_accurate_users_accuracies[i], 2)}%", horizontalalignment='center', verticalalignment='bottom', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(most_accurate_users, ha='right', va='top', rotation=45, color="black")









