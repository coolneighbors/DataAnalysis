import DataToolkit.Decorators as Decorators
from DataToolkit.Analyzer import uses_subject_ids, uses_user_identifiers
def timing():
    # Suppose we have a function which counts to a number.
    def count_to(n):
        count = 0
        for i in range(n):
            count += 1
        return count

    # We can use the timer decorator to time how long it takes to run the function.
    final_count = Decorators.timer(count_to)(1000000)

    final_count = Decorators.timer(count_to)(100000000)

    # This is helpful for determining the time complexity of the function.

    # You can also achieve the same outcome by doing:
    @Decorators.timer
    def count_to(n):
        count = 0
        for i in range(n):
            count += 1
        return count

    # We can use the timer decorator to time how long it takes to run the function.
    final_count = count_to(1000000)

    final_count = count_to(100000000)

def plot_input():
    # Import
    import matplotlib.pyplot as plt
    import numpy as np

    # Suppose we have a function which plots input and output values.

    @Decorators.plotting
    def plot_line(x, y, **kwargs):
        print(f"Keyword arguments: {kwargs}")
        plt.plot(x, y, **kwargs)

    # You can use the plotting decorator to set certain plotting parameters.
    # For example, you can set the title of the plot and the x and y limits.
    x = np.linspace(0, 1, 100)
    y = x**2
    plotting_kwargs = {"title": "Quadratic Plot", "xlim" : [0,1], "ylim" : [0,1], "save" : True, "color" : "red"}
    plot_line(x, y, **plotting_kwargs)

    # Keyword arguments which are not processed by the plotting decorator are passed into the function
    # (in this case plot_line) to be used internally such as the color of the line being plotted.

    # Here is a full list of the keyword arguments which can be used with the plotting decorator:
    # "title" : str
    # "xlabel" : str
    # "ylabel" : str
    # "xlim" : list
    # "ylim" : list
    # "axis_fontsize" : int
    # "title_fontsize" : int
    # "new_figure" : bool (set automatically to True if not specified)
    # figsize : list
    # show : bool (set automatically to True if not specified)
    # save : bool
    # filename : str (works with save, if not specified, the filename is the name of the function being called)
    # dpi : float (works with save, if not specified, dpi is set to 300)

def multiple_output():
    # Sometimes there are functions which natively only take in one input and return one output.
    # However, we may want to provide a list of inputs and get a list of outputs in one easy step.

    # Suppose we have a function which takes in a number and returns the square of that number.
    @Decorators.multioutput
    def square(x):
        return x**2

    # We can use the multioutput decorator to allow the function to take in a list of numbers and return a list of squares.

    # Here is an example of how to use the multioutput decorator.
    x = [1, 2, 3, 4, 5]
    print("Input: ", x)
    squares = square(x)
    print("Output: ", squares)

    # If you want to use the original functionality of a single input and single output, you can do so by passing in a single input.
    print("Input: ", 5)
    square_of_5 = square(5)
    print("Output: ", square_of_5)

    # However, if you provide a list with a single input, the function will return a list with a single output.
    print("Input: ", [5])
    square_of_5 = square([5])
    print("Output: ", square_of_5)

def ignoring_warnings():
    # Sometimes, you may want to ignore warnings which are raised by a function.
    # Suppose we have a function which raises a warning.
    import warnings
    def raise_warning():
        warnings.warn("This is a warning.")
        print("This is a print statement.")

    raise_warning()

    @Decorators.ignore_warnings
    def raise_warning():
        warnings.warn("This is a warning.")
        print("This is a print statement.")

    # We can use the ignore_warnings decorator to ignore the warning but still run the function.
    raise_warning()


def analyzer_decorators():
    # The Analyzer class has a few decorators which are used to modify the inputs of some of its methods.
    # The following decorators are used to modify the inputs of the Analyzer class' methods:
    # - uses_subject_ids
    # - uses_user_identifiers

    # These decorators will convert valid inputs into the correct type needed.

    # Suppose we have a function which takes in a subject id and uses it to do something. The multioutput decorator is used as well to
    #handle multi-inputs to the function.
    @Decorators.multioutput
    @uses_subject_ids
    def do_something(subject_input):
        print(f"Subject id: {subject_input}")

    # If we pass in a string, the function will convert it into an integer since we want subject ids.

    # Here is an example of how to use the do_something function.
    print("Test 1: ")
    do_something("1")
    print("Test 2: ")
    do_something(1)
    print("Test 3: ")
    do_something([1, 2, 3])
    print("Test 4: ")
    try:
        do_something("1c")
    except ValueError as e:
        # This will raise a ValueError since "1c" is not a valid subject id.
        print(e)

    # The uses_subject_ids decorator can handle the following inputs:
    # - int
    # - str (as long as it is a valid subject id)
    # - list of ints
    # - list of strs
    # - list of mixed ints and strs
    # - np.int64
    # - str (as long as it is a valid csv file path)
    # - TextIOWrapper or TextIO

    print("\n")

    # Suppose we have a function which takes in a user identifier and uses it to do something. The multioutput decorator is used as well to
    # handle multi-inputs to the function.
    @Decorators.multioutput
    @uses_user_identifiers
    def do_something(user_input):
        print(f"User identifier: {user_input}")

    # If we pass in a string which cannot be converted into an integer, the function will treat it as a username.

    # Here is an example of how to use the do_something function.
    print("Test 1: ")
    do_something("1")
    print("Test 2: ")
    do_something(1)
    print("Test 3: ")
    do_something([1, 2, 3])
    print("Test 4: ")
    do_something("Steven")
    print("Test 5: ")
    do_something(["Steven", "1", 2])

    # The uses_user_identifiers decorator can handle the following inputs:
    # - int
    # - str (as long as it is a valid user id or username)
    # - list of ints
    # - list of strs
    # - list of mixed ints and strs



if (__name__ == "__main__"):
    timing()
    plot_input()
    multiple_output()
    ignoring_warnings()
    analyzer_decorators()



