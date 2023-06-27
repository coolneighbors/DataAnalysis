import math
import re
import warnings
import webbrowser
from copy import copy

from abc import ABC, abstractmethod

import astropy.coordinates
import numpy as np
import pandas
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from flipbooks.WiseViewQuery import WiseViewQuery
from matplotlib import pyplot as plt

boolean_operators = ["&", "|"]

# Abstract class for all checkers
class Checker(ABC):
    @abstractmethod
    def getQuery(self, *args):
        pass

    @abstractmethod
    def checkQuery(self):
        pass

    @staticmethod
    def convertCoordinates(coordinates):
        if (isinstance(coordinates, SkyCoord)):
            coordinates = coordinates.transform_to("ICRS")
            ra = coordinates.ra.degree
            dec = coordinates.dec.degree
        elif (isinstance(coordinates, list)):
            ra = float(coordinates[0])
            dec = float(coordinates[1])
        elif (isinstance(coordinates, tuple)):
            ra = float(coordinates[0])
            dec = float(coordinates[1])
        elif(isinstance(coordinates, str)):
            coordinates = SkyCoord(coordinates)
            ra = coordinates.ra.degree
            dec = coordinates.dec.degree
        else:
            print("Coordinates:", coordinates)
            raise ValueError("Coordinates must be a SkyCoord, list, or tuple.")
        return ra, dec

    @staticmethod
    def convertAngle(angle):
        if (isinstance(angle, str)):
            angle = float(angle)
        elif (isinstance(angle, u.quantity.Quantity)):
            angle = angle.to(u.arcsec).value
        return angle

    @staticmethod
    def convertInput(coordinates, FOV, radius):
        FOV = Checker.convertAngle(FOV)

        radius = Checker.convertAngle(radius)

        ra, dec = Checker.convertCoordinates(coordinates)

        FOV, radius, search_type = Checker.verifySearchType(FOV, radius)

        return ra, dec, FOV, radius, search_type

    @staticmethod
    def verifySearchType(FOV, radius):
        search_type = None
        if (FOV is None and radius is None):
            raise ValueError("Either FOV or radius must be specified.")
        elif (FOV is not None and radius is not None):
            raise ValueError("Only one of FOV or radius can be specified.")
        elif (FOV is not None):
            radius = float(FOV) / math.sqrt(2)
            search_type = "Box Search"
        elif (radius is not None):
            search_type = "Cone Search"

        return FOV, radius, search_type

    @staticmethod
    def plotEntries(center_coordinates, result_dataframe, **kwargs):
        #TODO: Add a plot title
        #TODO: Fix angle distances, they are not being calculated correctly.
        FOV = kwargs.get('FOV', None)
        radius = kwargs.get('radius', None)

        ra, dec, FOV, radius, search_type = Checker.convertInput(center_coordinates, FOV, radius)
        print("RADIUS:", radius)
        deg_radius = radius / 3600

        print(deg_radius)

        distance_threshold = kwargs.get('distance_threshold', None)
        distance_threshold = Checker.convertAngle(distance_threshold)

        # Plot the Gaia entries in a 2D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.ticklabel_format(useOffset=False, style='plain')

        # Force equal aspect ratios for the axes
        ax.set_aspect('equal')

        # Plot the center coordinates
        plt.plot(ra, dec, 'r+', markersize=10)

        # Plot the entries
        print("Plotting entries...")
        print(result_dataframe['RA_d'])

        try:
            plt.scatter(result_dataframe['ra'].values, result_dataframe['dec'].values, s=3, c='k')
        except KeyError:
            try:
                try:
                    float(result_dataframe['RA'].values[0])
                    float(result_dataframe['DEC'].values[0])

                    plt.scatter(result_dataframe['RA'].values, result_dataframe['DEC'].values, s=3, c='k')
                except ValueError:
                    # Find the angular distance between the center coordinates and the entries
                    ra_values = result_dataframe['RA_d'].values
                    dec_values = result_dataframe['DEC_d'].values

                    # Convert the coordinates to SkyCoords
                    center_coordinates = SkyCoord(ra, dec, unit=(u.degree, u.degree))
                    entry_coordinates = SkyCoord(ra+(1/3600), dec, unit=(u.degree, u.degree))

                    # Calculate the angular distance between the center coordinates and the entries
                    angular_distance = center_coordinates.separation(entry_coordinates).arcsec
                    print("Angular distance:", angular_distance)

                    plt.scatter(result_dataframe['RA_d'].values, result_dataframe['DEC_d'].values, s=100, c='k')
                    print("PRinted")
            except KeyError:
                raise(KeyError("Could not find RA and DEC columns in the result dataframe."))

        # legend = plt.legend(loc='upper right', fontsize=12)
        handles = []
        labels = []

        # Draw the dashed box around the field of view
        if (search_type == "Box Search"):
            deg_FOV = FOV / 3600

            # Set the coordinates for the field of view box
            fov_x_start = ra - deg_FOV / 2
            fov_x_end = ra + deg_FOV / 2
            fov_y_start = dec - deg_FOV / 2
            fov_y_end = dec + deg_FOV / 2

            fov_rect = plt.Rectangle((fov_x_start, fov_y_start), fov_x_end - fov_x_start, fov_y_end - fov_y_start, edgecolor='blue', facecolor='none', linestyle='dashed')
            ax.add_patch(fov_rect)

            plt.xlim(fov_x_start, fov_x_end)
            plt.ylim(fov_y_start, fov_y_end)

            # Add to existing legend
            handles.append(fov_rect)
            labels.append("Field of View")

        # Draw a dashed circle around the field of view
        if (search_type == "Cone Search"):
            from regions import CircleSkyRegion

            search_circle = CircleSkyRegion(center=SkyCoord(ra, dec, unit=(u.degree, u.degree)), radius=deg_radius * u.degree)
            #search_circle = plt.Circle((ra, dec), deg_radius, edgecolor='green', facecolor='none', linestyle='dashed')
            ax.add_patch(search_circle.as_patch())

            # Set the axis limits
            plt.xlim(ra - deg_radius, ra + deg_radius)
            plt.ylim(dec - deg_radius, dec + deg_radius)

            handles.append(search_circle)
            labels.append("Search Radius")

        if (distance_threshold is not None):
            deg_distance_threshold = distance_threshold / 3600
            print("Distance Threshold:", deg_distance_threshold)
            distance_threshold_circle = plt.Circle((ra, dec), deg_distance_threshold, edgecolor='red', facecolor='none',
                                                   linestyle='dashed')
            ax.add_patch(distance_threshold_circle)

            # Add to existing legend
            handles.append(distance_threshold_circle)
            labels.append("Distance Threshold")

        # Add the legend
        plt.legend(handles=handles, labels=labels, loc='upper right', fontsize=10)

        # Set the axis labels
        plt.xlabel("RA (deg)")
        plt.ylabel("Dec (deg)")

        # Invert the x-axis
        ax.invert_xaxis()

        # Show the plot
        plt.show()

#TODO: Investigate vectorized queries
#TODO: Verify if I am requesting too often from SIMBAD

class SIMBADChecker(Checker):
    def __init__(self, *conditional_args):
        """
        Initialize the SIMBADChecker class

        Parameters
        ----------
        conditional_args : tuple
            A tuple of conditional arguments to be used in the SIMBAD query

        Notes:
        ------
        The conditional arguments are strings that are used to construct the SIMBAD query. The format of the conditional
        arguments is as follows:

        conditional_arg = VOTable_field_name conditional_operator value

        The conditional_operator can be any of the following:
        - <
        - >
        - <=
        - >=
        - =
        - ==
        - !=
        - ~=,threshold_value
        - like
        - !like
        - in
        - !in

        The value is a single value or string to compare against. If the conditional_operator is ~= then the
        threshold_value is whatever immediately follows the comma.

        """
        # Initialize the conditional arguments for the SIMBAD query
        self.conditional_args = conditional_args

    @staticmethod
    def isInSIMBAD(ra, dec, radius, *args):
        result_table = SIMBADChecker.getSIMBADQuery(ra, dec, radius, *args)
        return result_table is not None

    @staticmethod
    def getSIMBADQuery(ra, dec, FOV=None, radius=None, *args):

        ra = float(ra)
        dec = float(dec)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            simbad_query = Simbad()

            # Add some extra default VOTable fields to the query
            simbad_query.add_votable_fields('ra(d)', 'dec(d)', "ids")

            # Add the VOTable fields to the query
            for arg in args:
                simbad_query.add_votable_fields(arg)

            # Define your target coordinates
            target_coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

            # Construct the region query
            result_table = simbad_query.query_region(target_coords, radius=radius * u.arcsec)

            # Apply FOV filter on search results
            if (FOV is not None and result_table is not None):
                deg_FOV = FOV / 3600
                result_table = result_table[result_table['RA_d'] >= ra - deg_FOV / 2]
                result_table = result_table[result_table['RA_d'] <= ra + deg_FOV / 2]
                result_table = result_table[result_table['DEC_d'] >= dec - deg_FOV / 2]
                result_table = result_table[result_table['DEC_d'] <= dec + deg_FOV / 2]

            return result_table


    @staticmethod
    def getCriteriaArgs(conditional_strings):
        column_names = []
        criteria_args = ()

        for conditional_string in conditional_strings:
            all_sub_conditions = SIMBADChecker.getAllSubConditions(conditional_string)

            for sub_condition in all_sub_conditions:
                split_conditional_string = SIMBADChecker.safelySplitConditionalString(sub_condition)
                column_name = split_conditional_string[0]
                column_names.append(column_name)

            column_names = list(set(column_names))

        for column_name in column_names:
            default_criteria = ["MAIN_ID", "RA", "DEC", "RA_PREC", "DEC_PREC", "COO_ERR_MAJA", "COO_ERR_MINA",
                                "COO_ERR_ANGLE", "COO_QUAL", "COO_WAVELENGTH", "COO_BIBCODE", "RA_d", "DEC_d", "IDS",
                                "SCRIPT_NUMBER_ID"]
            default_criteria = [default_criteria_element.lower() for default_criteria_element in default_criteria]
            if (column_name.lower() not in list(criteria_args) and column_name.lower() not in default_criteria):
                criteria_args = list(criteria_args)
                criteria_args.append(column_name.lower())
                criteria_args = tuple(criteria_args)

        return criteria_args

    def getQuery(self, coordinates, *criteria_args, **kwargs):
        FOV = kwargs.get("FOV", None)
        radius = kwargs.get("radius", None)

        ra, dec, FOV, radius, self.search_type = self.convertInput(coordinates, FOV, radius)

        criteria_args = self.getCriteriaArgs(self.conditional_args)

        self.full_result_table = self.getSIMBADQuery(ra, dec, FOV, radius, *criteria_args)
        if(self.full_result_table is not None):
            self.full_result_dataframe = self.full_result_table.to_pandas()
        else:
            self.full_result_dataframe = None

        return copy(self.full_result_dataframe)

    @staticmethod
    def getAllSubConditions(conditional_string):
        conditional_string = copy(conditional_string)

        if (conditional_string[0] == "(" and conditional_string[-1] == ")"):
            conditional_string = conditional_string[1:-1]

        if (any(boolean_operator in conditional_string for boolean_operator in boolean_operators)):
            pattern = r"\(([^()]+)\)"
            sub_conditions = re.findall(pattern, conditional_string)
            # Add the parentheses back to the sub-conditions
            sub_conditions = ["(" + sub_condition + ")" for sub_condition in sub_conditions]
            return sub_conditions
        else:
            return ["(" + conditional_string + ")"]

    @staticmethod
    def getSubConditions(conditional_string):
        conditional_string = copy(conditional_string)

        if (conditional_string[0] == "(" and conditional_string[-1] == ")"):
            conditional_string = conditional_string[1:-1]

        if (any(boolean_operator in conditional_string for boolean_operator in boolean_operators)):
            pattern = r"\(((?:[^()]*|\([^()]*\))*)\)"
            sub_conditions = re.findall(pattern, conditional_string)
            # Add the parentheses back to the sub-conditions
            sub_conditions = ["(" + sub_condition + ")" for sub_condition in sub_conditions]
            return sub_conditions
        else:
            return ["(" + conditional_string + ")"]

    @staticmethod
    def safelySplitConditionalString(conditional_string):
        if (any(boolean_operator in conditional_string for boolean_operator in boolean_operators)):
            raise ValueError("Cannot safely split multi-conditional string with a boolean operator in it")

        if(conditional_string[0] == "(" and conditional_string[-1] == ")"):
            conditional_string = conditional_string[1:-1]
            if(conditional_string[0] == "(" and conditional_string[-1] == ")"):
                conditional_string = conditional_string[1:-1]

        split_conditional_string = conditional_string.split(" ")

        if (len(split_conditional_string) != 3):
            split_conditional_string = [split_conditional_string[0], split_conditional_string[1], " ".join(split_conditional_string[2:])]

        return split_conditional_string

    @staticmethod
    def applyOperator(operator, column_value, value):
        if (operator == "<"):
            return column_value < value
        elif (operator == ">"):
            return column_value > value
        elif (operator == "<="):
            return column_value <= value
        elif (operator == ">="):
            return column_value >= value
        elif (operator == "="):
            return column_value == value
        elif (operator == "=="):
            return column_value == value
        elif (operator == "!="):
            return column_value != value
        elif ("~=" in operator):
            threshold = float(operator.split(",")[1])
            column_value = float(column_value)
            value = float(value)
            return abs(column_value - value) <= threshold
        elif(operator == "like"):
            return value.lower() in column_value.lower()
        elif(operator == "!like"):
            return value.lower() not in column_value.lower()
        elif(operator == "!in"):
            return value not in column_value
        elif(operator == "in"):
            return value in column_value
        else:
            raise ValueError("Invalid operator: " + operator)

    @staticmethod
    def checkCondition(dataframe, conditional_string):
        sub_conditions = SIMBADChecker.getSubConditions(conditional_string)
        sub_condition_boolean_lists = []
        for sub_condition in sub_conditions:
            sub_sub_conditions = SIMBADChecker.getSubConditions(sub_condition)
            if(len(sub_sub_conditions) == 1):
                split_conditional_string = SIMBADChecker.safelySplitConditionalString(sub_condition)
                column_name = split_conditional_string[0]
                operator = split_conditional_string[1]
                value = split_conditional_string[2]

                column_booleans = []
                column_values = dataframe[column_name].values
                column_type = type(column_values[0])
                value = column_type(value)

                for column_value in column_values:
                    try:
                        if ("|" in column_value):
                            column_bool_list = []
                            for sub_column_value in column_value.split("|"):
                                column_bool_list.append(SIMBADChecker.applyOperator(operator, sub_column_value, value))
                            if("!" in operator):
                                column_booleans.append(all(column_bool_list))
                            else:
                                column_booleans.append(any(column_bool_list))
                        else:
                            column_booleans.append(SIMBADChecker.applyOperator(operator, column_value, value))
                    except TypeError:
                        column_booleans.append(SIMBADChecker.applyOperator(operator, column_value, value))
                sub_condition_boolean_lists.append(column_booleans)
            else:
                sub_condition_boolean_lists.append(SIMBADChecker.checkCondition(dataframe, sub_condition))


        dataframe_boolean_list = []
        for dataframe_index in range(len(sub_condition_boolean_lists[0])):
            modified_conditional_string = copy(conditional_string)
            for condition_index, sub_condition_boolean_list in enumerate(sub_condition_boolean_lists):
                boolean_value = sub_condition_boolean_list[dataframe_index]
                modified_conditional_string = modified_conditional_string.replace(sub_conditions[condition_index], str(boolean_value))
            dataframe_boolean_list.append(eval(modified_conditional_string))
        return pandas.Series(dataframe_boolean_list)


    def checkQuery(self):
        if(self.full_result_table is None):
            return None
        else:
            result_dataframe = copy(self.full_result_dataframe)
            self.result_table = copy(self.full_result_table)

            for conditional_arg in self.conditional_args:
                if(result_dataframe.empty):
                    return copy(result_dataframe)
                else:
                    result_dataframe.reset_index(drop=True, inplace=True)
                    boolean_series = self.checkCondition(result_dataframe, conditional_arg)
                    result_dataframe = result_dataframe[boolean_series]
                    self.result_table = self.result_table[boolean_series.values]

            return copy(result_dataframe)

    @staticmethod
    def buildConditionalArgument(column_names, operators, values, boolean_operator="|"):
        exceptions = ["RA_d", "DEC_d"]
        uppercase_exceptions = [exceptions_element.upper() for exceptions_element in exceptions]
        # Combine the exceptions and uppercase_exceptions lists as the keys and values of a dictionary
        exceptions_dict = dict(zip(uppercase_exceptions, exceptions))

        if(not isinstance(values, list)):
            values = [values]

        if(isinstance(column_names, str)):
            if(not column_names.upper() in uppercase_exceptions):
                column_names = column_names.upper()
            else:
                column_names = exceptions_dict[column_names.upper()]

        elif(isinstance(column_names, list)):
            new_column_names = []
            for column_name in column_names:
                if(not column_name.upper() in uppercase_exceptions):
                    new_column_names.append(column_name.upper())
                else:
                    new_column_names.append(exceptions_dict[column_name.upper()])
            column_names = new_column_names

        if(not isinstance(column_names, list)):
            column_names = len(values)*[column_names]

        if(not isinstance(operators, list)):
            operators = len(values)*[operators]

        if(len(column_names) != len(operators) or len(column_names) != len(values)):
            raise ValueError("column_names, operators, and values must have the same length.")

        conditional_args = []
        for i in range(len(column_names)):
            conditional_args.append("({} {} {})".format(column_names[i], operators[i], values[i]))

        if(len(conditional_args) == 1):
            return "(" + conditional_args[0] + ")"
        else:
            return "(" + boolean_operator.join(conditional_args) + ")"

    @staticmethod
    def combineConditionalArguments(*conditional_args, combine_with_value="|"):
        if(combine_with_value not in boolean_operators):
            raise ValueError("combine_with must be one of: {}".format(boolean_operators))

        if(len(conditional_args) == 1):
            return conditional_args[0]
        else:
            return "(" + combine_with_value.join(list(conditional_args)) + ")"

class GaiaChecker(Checker):

    def __init__(self, distance_threshold=0.0025):
        self.distance_threshold = distance_threshold
        Gaia.ROW_LIMIT = 100  # Ensure the default row limit.
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    @staticmethod
    def isInGaia(ra, dec, radius):
        r = GaiaChecker.getGaiaQuery(ra, dec, radius)
        return len(r) > 0

    @staticmethod
    def getGaiaQuery(ra, dec, FOV=None, radius=None):
        ra = float(ra)
        dec = float(dec)

        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
        j = Gaia.cone_search_async(coord, radius=u.Quantity(radius, u.arcsec))
        r = j.get_results()

        # Apply FOV filter on search results
        if(FOV is not None and r is not None):
            deg_FOV = FOV/3600
            r = r[r['ra'] >= ra - deg_FOV/2]
            r = r[r['ra'] <= ra + deg_FOV/2]
            r = r[r['dec'] >= dec - deg_FOV/2]
            r = r[r['dec'] <= dec + deg_FOV/2]

        return r

    @staticmethod
    def showGaiaEntries(result_dataframe, FOV=120, open_link=False):
        for index, source in result_dataframe.iterrows():
            source_ra = source['ra']
            source_dec = source['dec']
            WVQ = WiseViewQuery(ra=source_ra, dec=source_dec, size=WiseViewQuery.FOVToPixelSize(FOV))
            wise_view_link = WVQ.generateWiseViewURL()
            if (open_link):
                webbrowser.open(wise_view_link)
            else:
                print(wise_view_link)

    def getQuery(self, coordinates, **kwargs):
        FOV = kwargs.get('FOV', None)
        radius = kwargs.get('radius', None)
        
        ra, dec, FOV, radius, self.search_type = self.convertInput(coordinates, FOV, radius)
        
        self.full_result_table = GaiaChecker.getGaiaQuery(ra, dec, FOV, radius)
        if(self.full_result_table is not None):
            self.full_result_dataframe = self.full_result_table.to_pandas()
        else:
            self.full_result_dataframe = None

        return copy(self.full_result_dataframe)

    def checkQuery(self):
        if(self.full_result_table is None):
            return None
        else:
            self.result_dataframe = copy(self.full_result_dataframe)
            self.result_table = copy(self.full_result_table)
            distance_threshold_in_degrees = self.distance_threshold / 3600
            print("Distance threshold in degrees: {}".format(distance_threshold_in_degrees))
            self.result_dataframe = self.result_dataframe[self.result_dataframe['dist'] <= distance_threshold_in_degrees]
            self.result_table = self.result_table[self.result_table['dist'] <= distance_threshold_in_degrees]
            return copy(self.result_dataframe)


