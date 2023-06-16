import csv
import math
from copy import copy

from matplotlib import pyplot as plt

class Plotter:
    def __init__(self, ra_values, dec_values):
        """
        Initializes a Plotter object. This object is used to plot a set of RA and DEC values on a sky-map scatter plot.

        Parameters
        ----------
            ra_values : list
                A list of RA values to plot.
            dec_values : list
                A list of DEC values to plot.
        Notes
        -----
            The RA and DEC values are assumed to be in degrees.
            RA values are should be in the range [-180, 180].
            DEC values are should be to be in the range [-90, 90].
            However, the Plotter object will automatically correct any values outside of these ranges.
        """

        # Initialize the RA and DEC values.
        self.ra_values = ra_values
        self.dec_values = dec_values

    def projection_convert(self, ra, dec):
        """
        Converts RA and DEC values into their proper ranges and converts them from degrees to radians,
        as required by the matplotlib projection.

        Parameters
        ----------
            ra : float, list
                The RA value(s) to convert.
            dec : float, list
                The DEC value(s) to convert.
        Returns
        -------
            ra : float, str, list
                The converted RA value(s).
            dec : float, str, list
                The converted DEC value(s).
        """

        # Make a copy of the RA and DEC values.
        ra = copy(ra)
        dec = copy(dec)

        # Check the type of the RA and DEC values.
        if(isinstance(ra, str)):
            # Convert the RA and DEC values to floats if they are strings.
            ra = float(ra)
            dec = float(dec)
        elif(isinstance(ra, list)):
            # Apply the conversion to each RA and DEC value if they are lists.
            for i in range(len(ra)):
                ra[i], dec[i] = self.projection_convert(ra[i], dec[i])
            return ra, dec

        # Define a function to correct the RA and DEC values if they are outside of their proper ranges.
        def correct_angle(angle, start_angle, end_angle):
            angle_range = end_angle - start_angle
            corrected_angle = (angle - start_angle) % angle_range + start_angle
            return corrected_angle

        # Correct the RA and DEC values.
        ra = correct_angle(ra, -180, 180)
        dec = correct_angle(dec, -90, 90)

        # Convert the RA and DEC values from degrees to radians.
        ra = math.radians(ra)
        dec = math.radians(dec)

        # Return the converted RA and DEC values.
        return ra, dec

    def plot(self, projection='mollweide', show=True, **scatter_plot_kwargs):
        """
        Plots the RA and DEC values on a sky-map scatter plot.

        Parameters
        ----------
            projection : str
                The projection to use for the sky-map scatter plot.
                The default projection is 'mollweide'.
            show : bool
                Whether or not to show the plot after it is created.
                The default value is True.
            scatter_plot_kwargs : dict
                Any additional keyword arguments to pass to the scatter plot.
                The default values are:
                    s = 1
                    color = 'blue'
                    alpha = 0.5
        """

        # Check if the plot already exists.
        if len(plt.gcf().axes) == 0:
            # Create a new plot if it does not exist with the specified projection.
            fig = plt.gcf()
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            ax.grid()
        else:
            # Otherwise, use the existing axes.
            ax = plt.gcf().axes[0]

        # Set the default values for the scatter plot.
        s = scatter_plot_kwargs.get('s', 1)
        scatter_plot_kwargs['s'] = s

        color = scatter_plot_kwargs.get('color', 'blue')
        scatter_plot_kwargs['color'] = color

        alpha = scatter_plot_kwargs.get('alpha', 0.5)
        scatter_plot_kwargs['alpha'] = alpha

        # Convert the RA and DEC values to the proper ranges and units.
        converted_ra_values, converted_dec_values = self.projection_convert(self.ra_values, self.dec_values)

        # Plot the RA and DEC values.
        ax.scatter(converted_ra_values, converted_dec_values, **scatter_plot_kwargs)

        # Show the plot if specified.
        if show:
            plt.show()

class CSVPlotter(Plotter):
    def __init__(self, csv_filename, ra_column_name='RA', dec_column_name='DEC'):
        """
        Initializes a CSVPlotter object. This object is used to plot a set of RA and DEC values on a
        sky-map scatter plot from a CSV file.

        Parameters
        ----------
            csv_filename : str
                The name of the CSV file to read the RA and DEC values from.
            ra_column_name : str
                The name of the column in the CSV file that contains the RA values.
                The default value is 'RA'.
            dec_column_name : str
                The name of the column in the CSV file that contains the DEC values.
                The default value is 'DEC'.
        Notes
        -----
            The RA and DEC values are assumed to be in degrees.
            RA values are should be in the range [-180, 180].
            DEC values are should be to be in the range [-90, 90].
            However, the CSVPlotter object will automatically correct any values outside of these ranges.
        """

        # Extract the RA and DEC values from the CSV file.
        self.ra_values, self.dec_values = self.extract_ra_dec_from_csv(csv_filename, ra_column_name, dec_column_name)

        super().__init__(self.ra_values, self.dec_values)

    def extract_ra_dec_from_csv(self, csv_filename, ra_header='RA', dec_header='DEC'):
        """
        Extracts the RA and DEC values from a CSV file.

        Parameters
        ----------
            csv_filename : str
                The name of the CSV file to read the RA and DEC values from.
            ra_header : str
                The name of the column in the CSV file that contains the RA values.
                The default value is 'RA'.
            dec_header : str
                The name of the column in the CSV file that contains the DEC values.
                The default value is 'DEC'.
        Returns
        -------
            ra_list : list
                A list of RA values.
            dec_list : list
                A list of DEC values.
        """

        # Initialize the RA and DEC lists.
        ra_list = []
        dec_list = []

        # Open the CSV file
        with open(csv_filename, 'r') as csv_file:

            # Create a CSV reader object.
            reader = csv.DictReader(csv_file)

            # Iterate through each row in the CSV file.
            for row in reader:
                # Extract the RA and DEC values from the row.
                ra = row.get(ra_header)
                dec = row.get(dec_header)

                # Add the RA and DEC values to the lists, converting to floats if necessary.
                ra_list.append(float(ra))
                dec_list.append(float(dec))

        # Return the RA and DEC lists.
        return ra_list, dec_list


class SubjectPlotter(Plotter):
    def __init__(self, subjects, ra_metadata_name='RA', dec_metadata_name='DEC'):
        """
        Initializes a SubjectPlotter object. This object is used to plot a set of RA and DEC values on a
        sky-map scatter plot from a list of subjects.

        Parameters
        ----------
            subjects : list
                A list of subjects to extract the RA and DEC values from.
            ra_metadata_name : str
                The name of the metadata field in the subjects that contains the RA values.
                The default value is 'RA'.
            dec_metadata_name : str
                The name of the metadata field in the subjects that contains the DEC values.
                The default value is 'DEC'.
        Notes
        -----
            The RA and DEC values are assumed to be in degrees.
            RA values are should be in the range [-180, 180].
            DEC values are should be to be in the range [-90, 90].
            However, the SubjectPlotter object will automatically correct any values outside of these ranges.
        """

        # Extract the RA and DEC values from the subjects.
        self.ra_values, self.dec_values = self.extract_ra_dec_from_subjects(subjects, ra_metadata_name, dec_metadata_name)

        super().__init__(self.ra_values, self.dec_values)

    def extract_ra_dec_from_subjects(self, subjects, ra_field_name='RA', dec_field_name='DEC'):
        """
        Extracts the RA and DEC values from a list of subjects.

        Parameters
        ----------
            subjects : list
                A list of subjects to extract the RA and DEC values from.
            ra_field_name : str
                The name of the metadata field in the subjects that contains the RA values.
                The default value is 'RA'.
            dec_field_name : str
                The name of the metadata field in the subjects that contains the DEC values.
                The default value is 'DEC'.
        Returns
        -------
            ra_list : list
                A list of RA values.
            dec_list : list
                A list of DEC values.
        """

        # Initialize the RA and DEC lists.
        ra_list = []
        dec_list = []

        # Iterate through each subject in the list.
        for subject in subjects:
            # Extract the RA and DEC values from the subject.
            ra = subject.metadata.get(ra_field_name)
            dec = subject.metadata.get(dec_field_name)

            # Add the RA and DEC values to the lists, converting to floats if necessary.
            ra_list.append(float(ra))
            dec_list.append(float(dec))

        # Return the RA and DEC lists.
        return ra_list, dec_list

class SubjectCSVPlotter(Plotter):
    def __init__(self, subject_csv, ra_metadata_name='RA', dec_metadata_name='DEC'):
        """
        Initializes a SubjectCSVPlotter object. This object is used to plot a set of RA and DEC values on a
        sky-map scatter plot from a CSV file containing subject metadata.

        Parameters
        ----------
            subject_csv : str
                The name of the CSV file to read the RA and DEC values from.
            ra_metadata_name : str
                The name of the metadata field in the subjects that contains the RA values.
                The default value is 'RA'.
            dec_metadata_name : str
                The name of the metadata field in the subjects that contains the DEC values.
                The default value is 'DEC'.
        Notes
        -----
            The RA and DEC values are assumed to be in degrees.
            RA values are should be in the range [-180, 180].
            DEC values are should be to be in the range [-90, 90].
            However, the SubjectCSVPlotter object will automatically correct any values outside of these ranges.
        """
        # Extract the RA and DEC values from the subjects in the CSV file.
        self.ra_values, self.dec_values = self.extract_ra_dec_from_subject_csv(subject_csv, ra_metadata_name, dec_metadata_name)

        super().__init__(self.ra_values, self.dec_values)

    def extract_ra_dec_from_subject_csv(self, subject_csv, ra_field_name='RA', dec_field_name='DEC'):
        """
        Extracts the RA and DEC values from a CSV file.

        Parameters
        ----------
            subject_csv : str
                The name of the CSV file to read the RA and DEC values from.
            ra_field_name : str
                The name of the metadata field in the subjects that contains the RA values.
                The default value is 'RA'.
            dec_field_name : str
                The name of the metadata field in the subjects that contains the DEC values.
                The default value is 'DEC'.
        Returns
        -------
            ra_list : list
                A list of RA values.
            dec_list : list
                A list of DEC values.
        """

        # Initialize the RA and DEC lists.
        ra_list = []
        dec_list = []

        # Open the CSV file.
        with open(subject_csv, 'r') as csv_file:
            # Create a CSV reader object.
            reader = csv.DictReader(csv_file)

            # Iterate through each row in the CSV file.
            for row in reader:
                # Extract the RA and DEC values from the row.
                metadata = eval(row.get('metadata'))
                ra = metadata.get(ra_field_name)
                dec = metadata.get(dec_field_name)

                # Add the RA and DEC values to the lists, converting to floats if necessary.
                ra_list.append(float(ra))
                dec_list.append(float(dec))

        # Return the RA and DEC lists.
        return ra_list, dec_list

