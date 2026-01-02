"""Pytest fixtures for PDF converter tests."""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def sample_pdf_paths(temp_dir):
    """Create mock PDF file paths."""
    pdf_files = []
    for name in ["textbook.pdf", "notes.pdf", "paper.pdf"]:
        pdf_path = temp_dir / name
        pdf_path.write_bytes(b"%PDF-1.4 mock content")
        pdf_files.append(pdf_path)
    return pdf_files


@pytest.fixture
def single_pdf_path(temp_dir):
    """Create a single mock PDF file."""
    pdf_path = temp_dir / "single.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock content")
    return [pdf_path]


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """# Chapter 1: Introduction

This is a sample textbook chapter with various markdown elements.

## 1.1 Overview

Here is some introductory text that explains the topic. This paragraph
contains enough words to pass the minimum word count check in quality
analysis.

### Key Concepts

- Concept one is important
- Concept two builds on the first
- Concept three ties everything together

## 1.2 Data Tables

| Column A | Column B | Column C |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |
| Data 7   | Data 8   | Data 9   |
| Data 10  | Data 11  | Data 12  |
| Data 13  | Data 14  | Data 15  |

## 1.3 Code Examples

```python
def hello_world():
    print("Hello, World!")
```

## 1.4 Mathematical Formulas

The quadratic formula is:

$$
x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}
$$

## 1.5 Images and Links

![Diagram](images/diagram.png)

For more information, see [the documentation](https://example.com).

## Summary

This chapter covered the basics. In the next chapter, we will explore
more advanced topics including algorithms and data structures.

Additional filler text to ensure we have enough words for testing.
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit
in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
""" * 10  # Repeat to ensure word count > 1000


@pytest.fixture
def short_markdown():
    """Very short markdown content for testing failure detection."""
    return "# Title\n\nShort content."


@pytest.fixture
def garbled_markdown():
    """Markdown with high special character ratio."""
    return "!@#$%^&*(){}[]|\\:\";<>?,./" * 100 + " normal text here"


@pytest.fixture
def markdown_with_tables():
    """Markdown with multiple tables (3 tables, each with one |--- row)."""
    return """# Tables Document

| A | B |
|---|---|
| 1 | 2 |

| C | D | E |
|---|---|---|
| 3 | 4 | 5 |

| F |
|---|
| 9 |
"""


@pytest.fixture
def output_dir_with_markdown(temp_dir, sample_markdown):
    """Create output directory with a markdown file."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    md_file = output_dir / "converted.md"
    md_file.write_text(sample_markdown)
    return output_dir
