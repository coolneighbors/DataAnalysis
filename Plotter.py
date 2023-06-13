import csv
import math
from copy import copy

from matplotlib import pyplot as plt

class Plotter:
    def __init__(self, ra_values, dec_values):
        self.ra_values = ra_values
        self.dec_values = dec_values

    def projection_convert(self, ra, dec):
        ra = copy(ra)
        dec = copy(dec)

        if(isinstance(ra, str)):
            ra = float(ra)
            dec = float(dec)
        elif(isinstance(ra, list)):
            for i in range(len(ra)):
                ra[i], dec[i] = self.projection_convert(ra[i], dec[i])
            return ra, dec

        def correct_angle(angle, start_angle, end_angle):
            angle_range = end_angle - start_angle
            corrected_angle = (angle - start_angle) % angle_range + start_angle
            return corrected_angle

        ra = correct_angle(ra, -180, 180)
        dec = correct_angle(dec, -90, 90)

        ra = math.radians(ra)
        dec = math.radians(dec)

        return ra, dec

    def plot(self, scatter_plot_kwargs, projection='mollweide', show=True):
        if len(plt.gcf().axes) == 0:
            fig = plt.gcf()
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            ax.grid()
        else:
            ax = plt.gcf().axes[0]

        s = scatter_plot_kwargs.get('s', 1)
        scatter_plot_kwargs['s'] = s

        color = scatter_plot_kwargs.get('color', 'blue')
        scatter_plot_kwargs['color'] = color

        alpha = scatter_plot_kwargs.get('alpha', 0.5)
        scatter_plot_kwargs['alpha'] = alpha

        converted_ra_values, converted_dec_values = self.projection_convert(self.ra_values, self.dec_values)
        ax.scatter(converted_ra_values, converted_dec_values, **scatter_plot_kwargs)

        if show:
            plt.show()

class CSVPlotter(Plotter):
    def __init__(self, csv_filename, ra_column_name='RA', dec_column_name='DEC'):
        self.ra_values, self.dec_values = self.extract_ra_dec_from_csv(csv_filename, ra_column_name, dec_column_name)
        super().__init__(self.ra_values, self.dec_values)

    def extract_ra_dec_from_csv(self, csv_filename, ra_header='RA', dec_header='DEC'):
        ra_list = []
        dec_list = []

        with open(csv_filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                ra = row.get(ra_header)
                dec = row.get(dec_header)
                ra_list.append(float(ra))
                dec_list.append(float(dec))

        return ra_list, dec_list


class SubjectPlotter(Plotter):
    def __init__(self, subjects, ra_metadata_name='RA', dec_metadata_name='DEC'):
        self.ra_values, self.dec_values = self.extract_ra_dec_from_subjects(subjects, ra_metadata_name, dec_metadata_name)
        super().__init__(self.ra_values, self.dec_values)

    def extract_ra_dec_from_subjects(self, subjects, ra_header='RA', dec_header='DEC'):
        ra_list = []
        dec_list = []

        for subject in subjects:
            ra = subject.metadata.get(ra_header)
            dec = subject.metadata.get(dec_header)
            ra_list.append(float(ra))
            dec_list.append(float(dec))

        return ra_list, dec_list

class SubjectCSVPlotter(Plotter):
    def __init__(self, subject_csv, ra_metadata_name='RA', dec_metadata_name='DEC'):
        self.ra_values, self.dec_values = self.extract_ra_dec_from_subject_csv(subject_csv, ra_metadata_name, dec_metadata_name)
        super().__init__(self.ra_values, self.dec_values)

    def extract_ra_dec_from_subject_csv(self, subject_csv, ra_header='RA', dec_header='DEC'):
        ra_list = []
        dec_list = []

        with open(subject_csv, 'r') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                metadata = eval(row.get('metadata'))
                ra = metadata.get(ra_header)
                dec = metadata.get(dec_header)
                ra_list.append(float(ra))
                dec_list.append(float(dec))

        return ra_list, dec_list