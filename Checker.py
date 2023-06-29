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
from astropy.visualization import astropy_mpl_style
from astropy.visualization.wcsaxes import Quadrangle, SphericalCircle, WCSAxes
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from flipbooks.WiseViewQuery import WiseViewQuery
from matplotlib import pyplot as plt
from astropy import wcs
from astropy.io import fits
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Bbox
from regions import CircleSkyRegion, RectangleSkyRegion

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
            coordinates = coordinates.transform_to("icrs")
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
            FOV = radius * math.sqrt(2)

        return FOV, radius, search_type

    @staticmethod
    def convertWorldToPixels(ra, dec, wcs):
        return wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))

    @staticmethod
    def convertPixelsToWorld(x, y, wcs):
        return wcs.pixel_to_world(x, y)

    @staticmethod
    def createWCS(ra, dec):
        # Create a WCS object
        fits_header = fits.Header()
        fits_header['CTYPE1'] = 'RA---TAN'
        fits_header['CTYPE2'] = 'DEC--TAN'
        fits_header['CDELT1'] = -0.000763888888889
        fits_header['CDELT2'] = 0.000763888888889
        fits_header['CRVAL1'] = ra  # Reference RA value in degrees
        fits_header['CRVAL2'] = dec  # Reference DEC value in degrees
        fits_header['CRPIX1'] = 0  # Reference pixel in X (RA) direction
        fits_header['CRPIX2'] = 0  # Reference pixel in Y (DEC) direction

        default_wcs = wcs.WCS(fits_header)
        return default_wcs

    @staticmethod
    def getFOVBox(center_coordinates, FOV, wcs, **kwargs):

        # Get an accurate spherical circle that is inscribed in the FOV of the search
        inscribed_FOV_circle = SphericalCircle(center_coordinates, FOV / 2 * u.arcsec, resolution=300)

        # Get the vertices of the circle
        inscribed_FOV_circle_vertices = inscribed_FOV_circle.get_verts()

        # Get the bounding box of the circle using its vertices
        inscribed_FOV_circle_bbox = Bbox.from_extents(inscribed_FOV_circle_vertices[:, 0].min(),
                                                      inscribed_FOV_circle_vertices[:, 1].min(),
                                                      inscribed_FOV_circle_vertices[:, 0].max(),
                                                      inscribed_FOV_circle_vertices[:, 1].max())

        # Create a rectangle patch from the bounding box
        world_fov_rectangle = Rectangle((inscribed_FOV_circle_bbox.xmin, inscribed_FOV_circle_bbox.ymin), inscribed_FOV_circle_bbox.width, inscribed_FOV_circle_bbox.height, **kwargs)

        bottom_left = Checker.convertWorldToPixels(inscribed_FOV_circle_bbox.xmin, inscribed_FOV_circle_bbox.ymin, wcs)
        bottom_right = Checker.convertWorldToPixels(inscribed_FOV_circle_bbox.xmax, inscribed_FOV_circle_bbox.ymin, wcs)
        top_left = Checker.convertWorldToPixels(inscribed_FOV_circle_bbox.xmin, inscribed_FOV_circle_bbox.ymax, wcs)

        # Get width and height of the rectangle
        width = bottom_right[0] - bottom_left[0]
        height = top_left[1] - bottom_left[1]

        bottom_left = (float(bottom_left[0]), float(bottom_left[1]))

        pixel_fov_rectangle = Rectangle(bottom_left, width, height, **kwargs)

        return pixel_fov_rectangle, world_fov_rectangle


    @staticmethod
    def getFOVLimits(center_coordinates, FOV, wcs):
        # Get the bounding box of the FOV
        pixel_fov_rectangle, world_fov_rectangle = Checker.getFOVBox(center_coordinates, FOV, wcs)

        ra_min, dec_min = world_fov_rectangle.get_xy()
        ra_max = ra_min + world_fov_rectangle.get_width()
        dec_max = dec_min + world_fov_rectangle.get_height()

        return ra_min, ra_max, dec_min, dec_max

    @staticmethod
    def plotEntries(center_coordinates, result_dataframe, **kwargs):
        # TODO: Add way to add labels to the sources
        # TODO: Add way to plot multiple queries on the same plot with different colors
        if(result_dataframe is None):
            print("Cannot plot entries. Result dataframe is None.")
            return None

        FOV = kwargs.get('FOV', None)
        radius = kwargs.get('radius', None)
        add_grid = kwargs.get('grid', False)

        ra, dec, FOV, radius, search_type = Checker.convertInput(center_coordinates, FOV, radius)
        deg_radius = radius / 3600

        distance_threshold = kwargs.get('distance_threshold', None)
        distance_threshold = Checker.convertAngle(distance_threshold)
        #plt.style.use(astropy_mpl_style)

        # Plot the Gaia entries in a 2D plot
        fig = plt.figure(figsize=(10, 10))

        # Create a WCS object
        default_wcs = Checker.createWCS(ra, dec)

        ax = fig.add_subplot(111, projection=default_wcs)

        def setAxesLimits(ra_min, ra_max, dec_min, dec_max):
            xlim_skycoord = (SkyCoord(ra_min, dec, unit="deg"), SkyCoord(ra_max, dec, unit="deg"))
            xlim_pixel = (float(default_wcs.world_to_pixel(xlim_skycoord[0])[0]), float(default_wcs.world_to_pixel(xlim_skycoord[1])[0]))
            ax.set_xlim(xlim_pixel)

            ylim_skycoord = (SkyCoord(ra, dec_min, unit="deg"), SkyCoord(ra, dec_max, unit="deg"))
            ylim_pixel = (float(default_wcs.world_to_pixel(ylim_skycoord[0])[1]), float(default_wcs.world_to_pixel(ylim_skycoord[1])[1]))
            ax.set_ylim(ylim_pixel)

        ra_axis = ax.coords['ra']
        dec_axis = ax.coords['dec']

        ra_axis.set_major_formatter('dd:mm:ss')
        #ra_axis.set_ticks(spacing=30 * u.arcsec)
        dec_axis.set_major_formatter('dd:mm:ss')
        #dec_axis.set_ticks(spacing=30 * u.arcsec)
        ax.grid(add_grid)

        # Plot the center coordinates
        ax.plot(ra, dec, '+', transform=ax.get_transform('world'), color='green', markersize=10)

        # List of lists of possible column names for the coordinates
        coordinate_column_names = [['ra', 'dec'], ['RA', 'DEC'], ['RA_d', 'DEC_d']]

        # Plot the entries
        successful_plot = False
        for index, coordinate_column_name in enumerate(coordinate_column_names):
            # If the dataframe contains the RA and DEC columns, plot them
            if (coordinate_column_name[0] not in result_dataframe.columns or coordinate_column_name[1] not in result_dataframe.columns):
                continue

            # Check if the datafrane is empty
            if (len(result_dataframe) == 0 or result_dataframe.empty):
                successful_plot = True
                break

            ax.scatter(result_dataframe[coordinate_column_name[0]], result_dataframe[coordinate_column_name[1]], transform=ax.get_transform('world'), s=3, c='k')
            successful_plot = True

        if(not successful_plot):
            raise KeyError("Could not find RA and DEC columns in the result dataframe.")

        handles = []
        labels = []

        # Draw the dashed box around the field of view
        if (search_type == "Box Search"):

            pixel_fov_rectangle, world_fov_rectangle = Checker.getFOVBox(SkyCoord(ra, dec, unit=(u.deg, u.deg)), FOV, default_wcs, edgecolor='blue', facecolor='none', linestyle='dashed')
            ax.add_patch(pixel_fov_rectangle)

            # Set the limits of the axes

            setAxesLimits(*Checker.getFOVLimits(SkyCoord(ra, dec, unit=(u.deg, u.deg)), FOV, default_wcs))

            # Add to existing legend
            handles.append(pixel_fov_rectangle)

            labels.append("Field of View")

        # Draw a dashed circle around the field of view
        if (search_type == "Cone Search"):
            setAxesLimits(*Checker.getFOVLimits(SkyCoord(ra, dec, unit=(u.deg, u.deg)), 2*radius, default_wcs))

            # Make a spherical circle
            search_circle = SphericalCircle((ra, dec) * u.deg, deg_radius * u.deg, transform=ax.get_transform('world'), edgecolor='blue', facecolor='none', linestyle='dashed')
            ax.add_patch(search_circle)

            handles.append(search_circle)
            labels.append("Search Radius")

        if (distance_threshold is not None):
            deg_distance_threshold = distance_threshold / 3600
            distance_threshold_circle = SphericalCircle((ra, dec) * u.deg, deg_distance_threshold * u.deg, transform=ax.get_transform('world'), edgecolor='red', facecolor='none', linestyle='dashed')
            ax.add_patch(distance_threshold_circle)

            # Add to existing legend
            handles.append(distance_threshold_circle)
            labels.append("Distance Threshold")

        # Add the legend
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=10)

        # Set the axis labels
        plt.xlabel("RA")
        plt.ylabel("Dec")
        plt.title(f"Valid Query Results: {len(result_dataframe)}")

        # Reverse x-axis
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
    def isInSIMBAD(coordinates, FOV=None, radius=None, *args):
        result_table = SIMBADChecker.getSIMBADQuery(coordinates, FOV, radius, *args)
        if(result_table is None):
            return False
        elif(len(result_table) == 0):
            return False
        else:
            return True

    @staticmethod
    def getSIMBADQuery(coordinates, FOV=None, radius=None, *args):
        ra, dec, FOV, radius, search_type = SIMBADChecker.convertInput(coordinates, FOV, radius)

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
            if(search_type == "Box Search"):
                if(dec < 0):
                    dec = abs(dec)
                    box_search_string = f"region(box, {ra} -{dec}, {FOV}s {FOV}s)"
                else:
                    box_search_string = f"region(box, {ra} +{dec}, {FOV}s {FOV}s)"
                print("BOX SEARCH STRING: ", box_search_string)
                result_table = simbad_query.query_criteria(box_search_string)
                print("RESULT TABLE: ", result_table)
            elif(search_type == "Cone Search"):
                result_table = simbad_query.query_region(target_coords, radius=radius * u.arcsec)

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

        criteria_args = self.getCriteriaArgs(self.conditional_args)

        self.full_result_table = self.getSIMBADQuery(coordinates, FOV, radius, *criteria_args)
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
        Gaia.ROW_LIMIT = 200  # Ensure the default row limit.
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    @staticmethod
    def isInGaia(coordinates, FOV=None, radius=None):
        r = GaiaChecker.getGaiaQuery(coordinates, FOV, radius)
        return len(r) > 0

    @staticmethod
    def getGaiaQuery(coordinates, FOV=None, radius=None):

        ra, dec, FOV, radius, search_type = GaiaChecker.convertInput(coordinates, FOV, radius)

        # Create a WCS object
        fits_header = fits.Header()
        fits_header['CTYPE1'] = 'RA---TAN'
        fits_header['CTYPE2'] = 'DEC--TAN'
        fits_header['CDELT1'] = -0.000763888888889
        fits_header['CDELT2'] = 0.000763888888889
        fits_header['CRVAL1'] = ra  # Reference RA value in degrees
        fits_header['CRVAL2'] = dec  # Reference DEC value in degrees
        fits_header['CRPIX1'] = 0  # Reference pixel in X (RA) direction
        fits_header['CRPIX2'] = 0  # Reference pixel in Y (DEC) direction

        my_wcs = wcs.WCS(fits_header)

        if(search_type == "Box Search"):
            center_coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
            ra_min, ra_max, dec_min, dec_max = Checker.getFOVLimits(center_coords, FOV, my_wcs)
            j = Gaia.launch_job_async(f"SELECT designation, ra, dec, parallax, pmra, pmdec, DISTANCE({ra}, {dec}, ra, dec) AS dist " + "FROM gaiadr3.gaia_source " + f"WHERE ra BETWEEN {ra_min} AND {ra_max} AND dec BETWEEN {dec_min} AND {dec_max} " + "ORDER BY dist ASC")
            r = j.get_results()
        elif(search_type == "Cone Search"):
            coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
            j = Gaia.cone_search_async(coords, radius=u.Quantity(radius, u.arcsec))
            r = j.get_results()

        return r
    """
        ra, dec, FOV, radius, search_type = GaiaChecker.convertInput(coordinates, FOV, radius)
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
        print("FOV: {}".format(FOV))
        if(search_type == "Box Search"):
            r = Gaia.query_object_async(coordinate=coords, width=FOV * u.arcsec, height=FOV * u.arcsec)
            print(r)
        elif(search_type == "Cone Search"):
            j = Gaia.cone_search_async(coords, radius=u.Quantity(radius, u.arcsec))
            r = j.get_results()
        else:
            raise ValueError("search_type must be either 'Box Search' or 'Cone Search'")

        return r
    """
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
        
        self.full_result_table = GaiaChecker.getGaiaQuery(coordinates, FOV, radius)

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


