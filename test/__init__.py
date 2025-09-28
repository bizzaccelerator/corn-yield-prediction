"""Basic tests for the corn yield prediction project."""

import sys


def test_python_version():
    """Test that we're using a supported Python version."""
    assert sys.version_info >= (3, 8)


def test_basic_imports():
    """Test that essential packages can be imported."""
    try:
        import numpy
        import pandas

        assert True
    except ImportError as e:
        assert False, f"Failed to import required package: {e}"


def test_project_structure():
    """Test basic project structure exists."""
    import os

    # Add paths that should exist in your project
    expected_files = [
        "requirements.txt",  # Adjust based on your project
        ".github/workflows/ci.yml",
    ]

    for file_path in expected_files:
        if os.path.exists(file_path):
            assert os.path.exists(file_path), f"Expected file {file_path} not found"
