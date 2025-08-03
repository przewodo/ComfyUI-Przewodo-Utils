"""
Test runner script for ComfyUI-Przewodo-Utils.

This script provides an easy way to run different types of tests
with appropriate configurations.
"""
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_type="all", verbose=False, coverage=False, gpu=False, slow=False):
    """
    Run tests with specified configuration.
    
    Args:
        test_type: Type of tests to run ("unit", "integration", "workflow", "all")
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        gpu: Include GPU tests
        slow: Include slow tests
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test paths based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "workflow":
        cmd.append("tests/workflows/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Add markers and options
    markers = []
    
    if not gpu:
        markers.append("not gpu")
    
    if not slow:
        markers.append("not slow")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing"])
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Fail on unknown markers
        "--disable-warnings",  # Reduce noise
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest:")
        print("pip install pytest pytest-cov")
        return 1

def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Run tests for ComfyUI-Przewodo-Utils")
    
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "workflow", "all"],
        default="all",
        nargs="?",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Include GPU tests (requires CUDA)"
    )
    
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running tests"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        deps_result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "pytest", "pytest-cov", "pytest-mock", "torch", "torchvision"
        ])
        if deps_result.returncode != 0:
            print("Failed to install dependencies")
            return 1
    
    # Run tests
    return run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        gpu=args.gpu,
        slow=args.slow
    )

if __name__ == "__main__":
    sys.exit(main())
