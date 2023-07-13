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
        plotting_kwargs = {"show": (plt.show, None)}

        if ("title" in kwargs):
            title = kwargs["title"]
            del kwargs["title"]
            plotting_kwargs["title"] = (plt.title, title)

        if ("xlabel" in kwargs):
            xlabel = kwargs["xlabel"]
            del kwargs["xlabel"]
            plotting_kwargs["xlabel"] = (plt.xlabel, xlabel)

        if ("ylabel" in kwargs):
            ylabel = kwargs["ylabel"]
            del kwargs["ylabel"]
            plotting_kwargs["ylabel"] = (plt.ylabel, ylabel)

        if ("xlim" in kwargs):
            xlim = kwargs["xlim"]
            del kwargs["xlim"]
            plotting_kwargs["xlim"] = (plt.xlim, xlim)

        if ("ylim" in kwargs):
            ylim = kwargs["ylim"]
            del kwargs["ylim"]
            plotting_kwargs["ylim"] = (plt.ylim, ylim)

        if("show" in kwargs):
            show = kwargs["show"]
            del kwargs["show"]
            if(not show):
                del plotting_kwargs["show"]

        if("save" in kwargs):
            save = kwargs["save"]
            del kwargs["save"]
            if(save):
                filename = kwargs.get("filename", func.__qualname__ + ".png")
                plotting_kwargs["save"] = (plt.savefig, filename)

        f = func(*args, **kwargs)
        show_tuple = plotting_kwargs.pop("show", None)
        for key, value in plotting_kwargs.items():
            if (value is None):
                raise ValueError(f"Keyword argument '{key}' cannot be None.")

            function, arguments = value

            if (arguments is None):
                function()
            else:
                if(not isinstance(arguments, tuple)):
                    arguments = (arguments,)
                function(*arguments)

        if(show_tuple is not None):
            show_tuple[0]()

        return f

    return keyword_argument_handler

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time =  datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        time_in_ms = (end_time - start_time).total_seconds() * 1000
        print(f"Function {func.__name__} took {time_in_ms} ms to run.")
        return result
    return wrapper