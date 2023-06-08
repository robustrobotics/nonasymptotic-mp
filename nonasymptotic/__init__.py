try:
    import ompl
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    import os
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(0, os.environ['OMPL_PATH'])
    import ompl
