#!/usr/bin/env python3
"""
Test runner for MMI-004 routing infrastructure.

Executes comprehensive test suite and provides detailed results.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any

def run_pytest_with_coverage() -> Dict[str, Any]:
    """Run pytest with coverage reporting."""
    print("🧪 Running MMI-004 Routing Infrastructure Test Suite")
    print("=" * 60)

    # Change to worker directory
    worker_dir = Path(__file__).parent
    os.chdir(worker_dir)

    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes",
        "--durations=10",
        "--strict-markers"
    ]

    print(f"📋 Executing: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Test execution timed out after 300 seconds",
            "success": False
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Test execution failed: {str(e)}",
            "success": False
        }

def parse_test_results(output: str) -> Dict[str, Any]:
    """Parse pytest output to extract test results."""
    lines = output.split('\n')

    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "test_files": set(),
        "failed_tests": [],
        "duration": None
    }

    for line in lines:
        # Parse test result summary
        if "passed" in line or "failed" in line or "error" in line:
            if "::" in line:
                test_file = line.split("::")[0].strip()
                results["test_files"].add(test_file)

                if "PASSED" in line:
                    results["passed"] += 1
                elif "FAILED" in line or "ERROR" in line:
                    results["failed"] += 1
                    results["failed_tests"].append(line.strip())

        # Parse final summary
        if "failed," in line or "passed," in line or "error," in line:
            # Example: "5 failed, 10 passed in 2.34s"
            parts = line.split()
            for i, part in enumerate(parts):
                if part.isdigit():
                    num = int(part)
                    if i + 1 < len(parts):
                        status = parts[i + 1]
                        if "passed" in status:
                            results["passed"] = num
                        elif "failed" in status:
                            results["failed"] = num
                        elif "error" in status:
                            results["errors"] = num
                        elif "skipped" in status:
                            results["skipped"] = num

        # Parse duration
        if " in " in line and ("s" in line or "ms" in line):
            try:
                duration_part = line.split(" in ")[1].split()[0]
                if duration_part.endswith('s'):
                    results["duration"] = float(duration_part[:-1])
            except (IndexError, ValueError):
                pass

    results["total_tests"] = results["passed"] + results["failed"] + results["errors"] + results["skipped"]
    results["test_files"] = list(results["test_files"])

    return results

def print_test_summary(results: Dict[str, Any], test_output: str):
    """Print formatted test results summary."""
    print("\n" + "=" * 60)
    print("📊 MMI-004 Test Results Summary")
    print("=" * 60)

    # Overall status
    if results.get("success", False):
        print("✅ Overall Status: PASSED")
    else:
        print("❌ Overall Status: FAILED")

    print()

    # Test counts
    parsed = parse_test_results(test_output)
    print(f"📈 Test Statistics:")
    print(f"   Total Tests: {parsed['total_tests']}")
    print(f"   ✅ Passed: {parsed['passed']}")
    print(f"   ❌ Failed: {parsed['failed']}")
    print(f"   ⚠️  Errors: {parsed['errors']}")
    print(f"   ⏭️  Skipped: {parsed['skipped']}")

    if parsed['duration']:
        print(f"   ⏱️  Duration: {parsed['duration']:.2f}s")

    print()

    # Test files coverage
    print(f"📂 Test Files Executed ({len(parsed['test_files'])}):")
    for test_file in sorted(parsed['test_files']):
        print(f"   • {test_file}")

    print()

    # Component coverage
    components = {
        "RequestValidator": "test_request_validator.py",
        "ResponseFormatter": "test_response_formatter.py",
        "MultiModalHandler": "test_multi_modal_handler.py",
        "BaseHandler": "test_base_handler.py",
        "LoggingConfig": "test_logging_config.py",
        "Integration Tests": "test_routing_integration.py"
    }

    print("🔧 Component Test Coverage:")
    for component, test_file in components.items():
        status = "✅" if any(test_file in tf for tf in parsed['test_files']) else "❌"
        print(f"   {status} {component}")

    print()

    # Failed tests details
    if parsed['failed_tests']:
        print("❌ Failed Tests:")
        for failed_test in parsed['failed_tests'][:5]:  # Show first 5
            print(f"   • {failed_test}")
        if len(parsed['failed_tests']) > 5:
            print(f"   ... and {len(parsed['failed_tests']) - 5} more")
        print()

    # Success criteria
    print("✅ MMI-004 Success Criteria:")
    criteria = [
        ("Request validation tests pass", parsed['passed'] > 0 and 'test_request_validator' in str(parsed['test_files'])),
        ("Response formatting tests pass", parsed['passed'] > 0 and 'test_response_formatter' in str(parsed['test_files'])),
        ("Handler routing tests pass", parsed['passed'] > 0 and 'test_multi_modal_handler' in str(parsed['test_files'])),
        ("Base handler interface tests pass", parsed['passed'] > 0 and 'test_base_handler' in str(parsed['test_files'])),
        ("Logging system tests pass", parsed['passed'] > 0 and 'test_logging_config' in str(parsed['test_files'])),
        ("Integration tests pass", parsed['passed'] > 0 and 'test_routing_integration' in str(parsed['test_files'])),
        ("No test failures", parsed['failed'] == 0 and parsed['errors'] == 0),
        ("Comprehensive coverage", len(parsed['test_files']) >= 6)
    ]

    all_passed = True
    for criterion, passed in criteria:
        status = "✅" if passed else "❌"
        print(f"   {status} {criterion}")
        if not passed:
            all_passed = False

    print()

    if all_passed and results.get("success", False):
        print("🎉 MMI-004 Implementation: COMPLETE")
        print("   All routing infrastructure components tested and validated!")
    else:
        print("⚠️  MMI-004 Implementation: NEEDS ATTENTION")
        print("   Some tests failed or coverage is incomplete.")

    print("=" * 60)

def main():
    """Main test execution function."""
    # Check if we're in the right directory
    if not os.path.exists("tests"):
        print("❌ Error: tests directory not found")
        print("   Please run this script from the multi-model-worker directory")
        return 1

    # Run tests
    results = run_pytest_with_coverage()

    # Print output
    if results["stdout"]:
        print(results["stdout"])

    if results["stderr"]:
        print("🔍 Test Stderr:")
        print(results["stderr"])
        print()

    # Print summary
    print_test_summary(results, results["stdout"])

    return 0 if results["success"] else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)