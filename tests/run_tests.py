#!/usr/bin/env python3
"""
CRITICAL: Main Test Runner

This script runs all tests in the test suite and provides comprehensive
reporting of test results, coverage, and performance metrics.
"""

import unittest
import sys
import os
import time
import argparse
from pathlib import Path
import subprocess

# Add project root to path
# run_tests.py is in tests/run_tests.py, so we need to go up 1 level to get to project root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def discover_tests(test_dir="tests"):
    """Discover all test modules in the test directory"""
    loader = unittest.TestLoader()
    
    # Handle both relative and absolute paths
    if test_dir == "tests":
        # Default case: run all tests in the tests directory
        start_dir = REPO_ROOT / "tests"
    elif test_dir.startswith("tests/"):
        start_dir = REPO_ROOT / test_dir
    else:
        start_dir = REPO_ROOT / "tests" / test_dir
    
    if not start_dir.exists():
        print(f"❌ Test directory not found: {start_dir}")
        return None
    
    # Discover tests with more explicit pattern matching
    suite = loader.discover(
        str(start_dir), 
        pattern="test_*.py",
        top_level_dir=str(REPO_ROOT)
    )
    
    # Debug: Print discovered tests
    if suite.countTestCases() == 0:
        print(f"⚠️  No tests discovered in {start_dir}")
        print(f"   Looking for files matching pattern: test_*.py")
        
        # List all Python files in the directory
        python_files = list(start_dir.rglob("test_*.py"))
        if python_files:
            print(f"   Found {len(python_files)} test files:")
            for file in python_files:
                print(f"     - {file}")
        else:
            print(f"   No test files found in {start_dir}")
    
    return suite


def run_tests(suite, verbosity=2, failfast=False):
    """Run the test suite and return results"""
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=failfast,
        buffer=True
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time


def print_test_summary(result, duration):
    """Print a comprehensive test summary"""
    print("\n" + "="*80)
    print("CRITICAL: TEST SUMMARY")
    print("="*80)
    
    # Test counts
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped
    
    print(f"📊 Test Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failures}")
    print(f"   ⚠️  Errors: {errors}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   ⏱️  Duration: {duration:.2f} seconds")
    
    # Success rate
    if total_tests > 0:
        success_rate = (passed / total_tests) * 100
        print(f"   📈 Success Rate: {success_rate:.1f}%")
    
    # Status
    if failures == 0 and errors == 0:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n💥 SOME TESTS FAILED! 💥")
        return False


def print_failure_details(result):
    """Print detailed information about test failures"""
    if result.failures or result.errors:
        print("\n" + "="*80)
        print("CRITICAL: FAILURE DETAILS")
        print("="*80)
        
        # Print failures
        if result.failures:
            print("\n❌ FAILURES:")
            for i, (test, traceback) in enumerate(result.failures, 1):
                print(f"\n{i}. {test}")
                print("-" * 60)
                print(traceback)
        
        # Print errors
        if result.errors:
            print("\n⚠️  ERRORS:")
            for i, (test, traceback) in enumerate(result.errors, 1):
                print(f"\n{i}. {test}")
                print("-" * 60)
                print(traceback)


def run_specific_tests(test_patterns, verbosity=2, failfast=False):
    """Run specific test patterns"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for pattern in test_patterns:
        try:
            # Try to load the test
            test = loader.loadTestsFromName(pattern)
            suite.addTest(test)
        except Exception as e:
            print(f"❌ Failed to load test pattern '{pattern}': {e}")
    
    if suite.countTestCases() == 0:
        print("❌ No tests found matching the specified patterns")
        return False
    
    result, duration = run_tests(suite, verbosity, failfast)
    success = print_test_summary(result, duration)
    print_failure_details(result)
    
    return success


def run_coverage_analysis():
    """Run test coverage analysis if coverage is available"""
    try:
        import coverage
        print("\n🔍 Running coverage analysis...")
        
        # Start coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        suite = discover_tests()
        if suite:
            result, duration = run_tests(suite, verbosity=1)
            success = print_test_summary(result, duration)
        else:
            success = False
        
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        # Generate coverage report
        print("\n📊 Coverage Report:")
        cov.report()
        
        # Generate HTML coverage report
        html_dir = REPO_ROOT / "tests" / "coverage_html"
        html_dir.mkdir(exist_ok=True)
        cov.html_report(directory=str(html_dir))
        print(f"📁 HTML coverage report generated in: {html_dir}")
        
        return success
        
    except ImportError:
        print("⚠️  Coverage module not available. Install with: pip install coverage")
        return None


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Run the HAR Research test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python tests/run_tests.py
  
  # Run specific test modules
  python tests/run_tests.py -t tests.unit.test_model_config_builder
  python tests/run_tests.py -t tests.unit.test_yaml_model_loading
  
  # Run with coverage analysis
  python tests/run_tests.py --coverage
  
  # Run with verbose output
  python tests/run_tests.py -v
  
  # Run and stop on first failure
  python tests/run_tests.py --failfast
        """
    )
    
    parser.add_argument("-t", "--tests", nargs="+", 
                       help="Specific test patterns to run")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--failfast", action="store_true",
                       help="Stop on first failure")
    parser.add_argument("--coverage", action="store_true",
                       help="Run with coverage analysis")
    parser.add_argument("--unit", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--integration", action="store_true",
                       help="Run only integration tests")
    parser.add_argument("--gui", action="store_true",
                       help="Run only GUI tests")
    
    args = parser.parse_args()
    
    # Set verbosity
    verbosity = 2 if args.verbose else 1
    
    print("🧪 HAR Research Test Suite")
    print("="*50)
    
    # Determine which tests to run
    if args.tests:
        # Run specific tests
        success = run_specific_tests(args.tests, verbosity, args.failfast)
    elif args.coverage:
        # Run with coverage
        success = run_coverage_analysis()
        if success is None:
            success = False
    else:
        # Run all tests or specific categories
        if args.unit:
            test_dir = "tests/unit"
        elif args.integration:
            test_dir = "tests/integration"
        elif args.gui:
            test_dir = "tests/gui"
        else:
            test_dir = "tests"
        
        suite = discover_tests(test_dir)
        if suite is None:
            success = False
        else:
            result, duration = run_tests(suite, verbosity, args.failfast)
            success = print_test_summary(result, duration)
            print_failure_details(result)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
