import datetime
import functools
import os
import warnings
from io import TextIOWrapper
from typing import Iterable, TextIO

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def ignore_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper

def multioutput(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for index, arg in enumerate(args):
            if(isinstance(arg, Iterable) and not isinstance(arg, str) and not isinstance(arg, TextIOWrapper) and not isinstance(arg, TextIO)):
                iterable_type = type(arg)

                if(not isinstance(arg, list) and not isinstance(arg, tuple)):
                    warnings.warn(f"Iterable argument {arg} of type '{iterable_type.__name__}' is not a list or tuple. Returning results as a list.")
                    iterable_type = list

                before_args = args[:index]
                after_args = args[index + 1:]
                return iterable_type([func(*before_args, iterable_argument, *after_args, **kwargs) for iterable_argument in arg])

        return func(*args, **kwargs)
    return wrapper

def plotting(func):
    @functools.wraps(func)
    def keyword_argument_handler(*args, **kwargs):
        # Has the form of (function, argument, keyword_arguments)
        def create_plotting_dictionary(function, positional_arguments=(), keyword_arguments={}):
            if (positional_arguments is None):
                positional_arguments = ()
            elif (not isinstance(positional_arguments, tuple)):
                positional_arguments = (positional_arguments,)

            if (keyword_arguments is None):
                keyword_arguments = {}
            elif (not isinstance(keyword_arguments, dict)):
                raise ValueError(f"Keyword arguments '{keyword_arguments}' must be a dictionary.")

            return {"function": function, "positional_arguments": positional_arguments, "keyword_arguments": keyword_arguments}

        plotting_kwargs = {"show": create_plotting_dictionary(plt.show)}

        create_new_figure = kwargs.pop("new_figure", True)

        if (create_new_figure):
            plt.figure()

        if ("title" in kwargs):
            title = kwargs["title"]
            del kwargs["title"]
            title_fontsize = kwargs.pop("title_fontsize", 12)
            plotting_kwargs["title"] = create_plotting_dictionary(plt.title, title, {"fontsize": title_fontsize})

        if("title_fontsize" in kwargs):
            title_fontsize = kwargs["title_fontsize"]
            del kwargs["title_fontsize"]
            title = plt.gca().title
            plotting_kwargs["title_fontsize"] = create_plotting_dictionary(title.set_fontsize, title_fontsize)

        if ("xlabel" in kwargs):
            xlabel = kwargs["xlabel"]
            del kwargs["xlabel"]
            axis_fontsize = kwargs.get("axis_fontsize", 10)
            plotting_kwargs["xlabel"] = create_plotting_dictionary(plt.xlabel, xlabel, {"fontsize": axis_fontsize})

        if ("ylabel" in kwargs):
            ylabel = kwargs["ylabel"]
            del kwargs["ylabel"]
            axis_fontsize = kwargs.get("axis_fontsize", 10)
            plotting_kwargs["ylabel"] = create_plotting_dictionary(plt.ylabel, ylabel, {"fontsize": axis_fontsize})

        if ("axis_fontsize" in kwargs):
            axis_fontsize = kwargs["axis_fontsize"]
            del kwargs["axis_fontsize"]
            x_axis = plt.gca().xaxis
            x_label = x_axis.get_label()
            y_axis = plt.gca().yaxis
            y_label = y_axis.get_label()
            axis_labels_set_fontsize = lambda fontsize: (x_label.set_fontsize(fontsize), y_label.set_fontsize(fontsize))
            plotting_kwargs["axis_fontsize"] = create_plotting_dictionary(axis_labels_set_fontsize, axis_fontsize)

        if ("xlim" in kwargs):
            xlim = kwargs["xlim"]
            del kwargs["xlim"]
            plotting_kwargs["xlim"] = create_plotting_dictionary(plt.xlim, xlim)

        if ("ylim" in kwargs):
            ylim = kwargs["ylim"]
            del kwargs["ylim"]
            plotting_kwargs["ylim"] = create_plotting_dictionary(plt.ylim, ylim)

        if ("figsize" in kwargs):
            figsize = kwargs["figsize"]
            del kwargs["figsize"]
            plotting_kwargs["figsize"] = create_plotting_dictionary(plt.gcf().set_size_inches, figsize)

        if("show" in kwargs):
            show = kwargs["show"]
            del kwargs["show"]
            if(not show):
                del plotting_kwargs["show"]

        if("save" in kwargs):
            save = kwargs["save"]
            del kwargs["save"]
            if(save):
                filename = kwargs.pop("filename", (func.__qualname__ + ".png").replace("<", "").replace(">", ""))
                print(f"Saving figure to {filename}")
                save_kwargs = {"dpi" : 300.0}
                plotting_kwargs["save"] = create_plotting_dictionary(plt.savefig, filename, save_kwargs)

        # Run the plotting function
        f = func(*args, **kwargs)

        final_keywords = ["save", "show"]
        final_keywords_dictionaries = [plotting_kwargs.pop(keyword, None) for keyword in final_keywords]

        def run_keyword_function(plotting_dictionary, verbose=False):
            function = plotting_dictionary["function"]
            positional_arguments = plotting_dictionary["positional_arguments"]
            keyword_arguments = plotting_dictionary["keyword_arguments"]
            if(verbose):
                print(f"Running function {function.__name__} with positional arguments {positional_arguments} and keyword arguments {keyword_arguments}")
            function(*positional_arguments, **keyword_arguments)

        # Run the plotting functions
        for key, plotting_dictionary in plotting_kwargs.items():
            if (plotting_dictionary is None):
                raise ValueError(f"Keyword argument '{key}' cannot be None.")

            run_keyword_function(plotting_dictionary)

        # Run the plotting functions which occur after the main plotting functions
        for final_keywords_dictionary in final_keywords_dictionaries:
            if(final_keywords_dictionary is not None):
                run_keyword_function(final_keywords_dictionary)

        # Return the plotting function's result
        return f

    return keyword_argument_handler

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time =  datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        time_in_ms = (end_time - start_time).total_seconds() * 1000

        # If the time in ms is less than 1000, print it in ms. Otherwise, print it in seconds, minutes, or hours.
        formatted_time = ""
        if(time_in_ms < 1000):
            formatted_time = f"{time_in_ms:.2f} ms"
        elif(time_in_ms < 1000 * 60):
            formatted_time = f"{time_in_ms / 1000:.2f} seconds"
        elif(time_in_ms < 1000 * 60 * 60):
            formatted_time = f"{time_in_ms / (1000 * 60):.2f} minutes"
        else:
            formatted_time = f"{time_in_ms / (1000 * 60 * 60):.2f} hours"

        print(f"Function '{func.__name__}' took {formatted_time} to run.")
        return result
    return wrapper