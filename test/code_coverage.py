#!/usr/bin/env python3

"""
Generate coverage reports for the testModels and testSBML scripts
exported format is cobertura xml
"""

import coverage
import unittest
import os
import sys

import test_sbml_conversion
import test_objective
import test_optimize

# only consider pypesto module and
cov = coverage.Coverage(source=['pypesto'])

# ignore code blocks containing import statements
cov.exclude('import')
cov.start()

# build the testSuite from testModels and testSBML
suite = unittest.TestSuite()
suite.addTest(test_sbml_conversion.AmiciObjectiveTest())
suite.addTest(test_objective.ObjectiveTest())
suite.addTest(test_optimize.OptimizerTest())
testRunner = unittest.TextTestRunner(verbosity=0)
result = testRunner.run(suite)

cov.stop()
cov.xml_report(outfile='coverage_py.xml')

# propagate failure
sys.exit(not result.wasSuccessful())
