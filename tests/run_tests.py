"""
Test runner for the model management framework test suite.

Provides unified test execution, reporting, and performance benchmarking
for all model management components.
"""

import unittest
import sys
import time
import argparse
from pathlib import Path

# Add the src directory and current directory to the path for imports
src_path = Path(__file__).parent.parent / "src"
test_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(test_path))

# Import test modules
from tests.unit.test_model_manager import TestModelManager
from tests.unit.test_memory_monitor import TestMemoryMonitor
from tests.integration.test_model_lifecycle import TestModelLifecycle
from tests.integration.test_performance_benchmarks import TestModelManagementPerformance


def create_test_suite(test_types=None):
    """Create a test suite with specified test types."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if not test_types:
        test_types = ['unit', 'integration', 'performance']

    # Unit tests
    if 'unit' in test_types:
        suite.addTests(loader.loadTestsFromTestCase(TestModelManager))
        suite.addTests(loader.loadTestsFromTestCase(TestMemoryMonitor))

    # Integration tests
    if 'integration' in test_types:
        suite.addTests(loader.loadTestsFromTestCase(TestModelLifecycle))

    # Performance benchmarks
    if 'performance' in test_types:
        suite.addTests(loader.loadTestsFromTestCase(TestModelManagementPerformance))

    return suite


class DetailedTestResult(unittest.TextTestResult):
    """Enhanced test result reporter with timing and categorization."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_timings = {}
        self.test_categories = {
            'unit': [],
            'integration': [],
            'performance': []
        }

    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.perf_counter()

        # Categorize test
        test_class = test.__class__.__name__
        if 'Performance' in test_class or 'Benchmark' in test_class:
            category = 'performance'
        elif 'Integration' in test_class or 'Lifecycle' in test_class:
            category = 'integration'
        else:
            category = 'unit'

        self.test_categories[category].append(test)

    def stopTest(self, test):
        super().stopTest(test)
        if hasattr(self, 'test_start_time'):
            duration = time.perf_counter() - self.test_start_time
            self.test_timings[str(test)] = duration

    def printErrors(self):
        super().printErrors()
        self.print_timing_summary()
        self.print_category_summary()

    def print_timing_summary(self):
        if not self.test_timings:
            return

        self.stream.writeln("\n" + "="*70)
        self.stream.writeln("TEST TIMING SUMMARY")
        self.stream.writeln("="*70)

        # Sort by duration (slowest first)
        sorted_tests = sorted(self.test_timings.items(), key=lambda x: x[1], reverse=True)

        self.stream.writeln(f"{'Test':<50} {'Duration':<10}")
        self.stream.writeln("-" * 70)

        for test_name, duration in sorted_tests:
            # Shorten test name for display
            short_name = test_name.split('.')[-1] if '.' in test_name else test_name
            if len(short_name) > 47:
                short_name = short_name[:44] + "..."

            self.stream.writeln(f"{short_name:<50} {duration:.3f}s")

        total_time = sum(self.test_timings.values())
        avg_time = total_time / len(self.test_timings)

        self.stream.writeln("-" * 70)
        self.stream.writeln(f"{'Total test time:':<50} {total_time:.3f}s")
        self.stream.writeln(f"{'Average test time:':<50} {avg_time:.3f}s")

    def print_category_summary(self):
        self.stream.writeln("\n" + "="*70)
        self.stream.writeln("TEST CATEGORY SUMMARY")
        self.stream.writeln("="*70)

        for category, tests in self.test_categories.items():
            if not tests:
                continue

            self.stream.writeln(f"\n{category.upper()} TESTS:")

            # Calculate category statistics
            category_tests = [str(t) for t in tests]
            category_timings = {t: self.test_timings.get(t, 0) for t in category_tests}

            passed = len([t for t in tests if str(t) not in [str(f[0]) for f in self.failures + self.errors]])
            failed = len(tests) - passed

            total_time = sum(category_timings.values())

            self.stream.writeln(f"  Tests run: {len(tests)}")
            self.stream.writeln(f"  Passed: {passed}")
            self.stream.writeln(f"  Failed: {failed}")
            self.stream.writeln(f"  Total time: {total_time:.3f}s")

            if len(tests) > 0:
                avg_time = total_time / len(tests)
                self.stream.writeln(f"  Average time: {avg_time:.3f}s")


def run_tests(test_types=None, verbosity=2, failfast=False):
    """Run the test suite with enhanced reporting."""
    suite = create_test_suite(test_types)

    # Create custom test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        resultclass=DetailedTestResult,
        failfast=failfast,
        buffer=True  # Capture stdout/stderr for cleaner output
    )

    print("="*70)
    print("MODEL MANAGEMENT FRAMEWORK TEST SUITE")
    print("="*70)

    if test_types:
        print(f"Running test types: {', '.join(test_types)}")
    else:
        print("Running all test types: unit, integration, performance")

    print(f"Test suite contains {suite.countTestCases()} test cases")
    print()

    # Run the tests
    start_time = time.perf_counter()
    result = runner.run(suite)
    end_time = time.perf_counter()

    # Print final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")

    if result.failures or result.errors:
        print("\nFAILED - Some tests did not pass")
        return False
    else:
        print("\nSUCCESS - All tests passed!")
        return True


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description='Model Management Framework Test Suite')

    parser.add_argument(
        '--type', '-t',
        choices=['unit', 'integration', 'performance', 'all'],
        default='all',
        help='Type of tests to run (default: all)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=2,
        help='Increase verbosity level (use -v, -vv, or -vvv)'
    )

    parser.add_argument(
        '--failfast', '-f',
        action='store_true',
        help='Stop on first failure'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tests without running them'
    )

    args = parser.parse_args()

    # Determine test types to run
    if args.type == 'all':
        test_types = None  # Run all types
    else:
        test_types = [args.type]

    if args.list:
        # List all tests
        suite = create_test_suite()
        print("Available tests:")
        for test in suite:
            if hasattr(test, '_testMethodName'):
                test_class = test.__class__.__name__
                test_method = test._testMethodName
                print(f"  {test_class}.{test_method}")
        return

    # Run the tests
    success = run_tests(test_types, args.verbose, args.failfast)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()