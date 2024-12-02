#!/usr/bin/env python
import unittest
import coverage
import sys
from pathlib import Path

def run_tests_with_coverage():
    """Run all tests with coverage reporting."""
    # Start code coverage monitoring
    cov = coverage.Coverage(
        branch=True,
        source=['src'],
        omit=[
            '*/tests/*',
            '**/__pycache__/*',
            '*/__init__.py'
        ]
    )
    cov.start()

    # Discover and run tests
    loader = unittest.TestLoader()
    tests_dir = Path(__file__).parent / 'tests'
    suite = loader.discover(str(tests_dir))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Stop coverage monitoring and generate report
    cov.stop()
    cov.save()
    
    print('\nCoverage Report:')
    cov.report()
    
    # Generate HTML coverage report
    cov.html_report(directory='coverage_report')
    print('\nDetailed HTML coverage report generated in coverage_report/index.html')

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests_with_coverage()
    sys.exit(0 if success else 1)