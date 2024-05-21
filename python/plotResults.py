#!/usr/bin/env python
"""
    This script is used to plot the results of the algorithm
"""

import matplotlib.pyplot as plt

def plotResults(results):
    """
        Plot the results of the algorithm
    """
    pass

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        raise ValueError("Usage: python plotResults.py <results_file>")
    results = sys.argv[1]
    plotResults(results)