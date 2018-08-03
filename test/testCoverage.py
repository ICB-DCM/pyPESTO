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

# only consider pesto module and 
cov = coverage.Coverage(source=['pesto'])

# ignore code blocks containing import statements
cov.exclude('import')
cov.start()

# build the testSuite from testModels and testSBML
suite = unittest.TestSuite()
suite.addTest(test_sbml_conversion.OptimizerTest())
testRunner = unittest.TextTestRunner(verbosity=0)
result = testRunner.run(suite)

cov.stop()
cov.xml_report(outfile='coverage_py.xml')

# propagate failure
sys.exit(not result.wasSuccessful())