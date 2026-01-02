# Unit Testing Guide

This document explains how to run and verify the unit tests for KidKazz RAG.

---

## Phase 1: PDF Converter

### Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install test dependencies
pip install pytest pytest-cov

# 3. Run all tests
PYTHONPATH=. pytest tests/ -v

# 4. Run with coverage report
PYTHONPATH=. pytest tests/ --cov=src --cov-report=term-missing
```

---

## Test Results Summary

```
========================= test session starts ==========================
platform linux -- Python 3.13.11, pytest-9.0.2
collected 84 items

tests/test_analyzer.py   ............................ [ 40%]
tests/test_converter.py  ........................    [ 67%]
tests/test_selector.py   ...........................  [100%]

========================= 84 passed in 0.39s ===========================
```

### Coverage Report

```
Name                             Stmts   Miss  Cover   Missing
--------------------------------------------------------------
src/pdf_converter/__init__.py        4      0   100%
src/pdf_converter/analyzer.py       61      1    98%   93
src/pdf_converter/converter.py      86     21    76%   170-194, 204, 231, 240-242, 264-265
src/pdf_converter/selector.py       32      0   100%
--------------------------------------------------------------
TOTAL                              183     22    88%
```

---

## Test Modules

### 1. test_selector.py - PDF Selection & Tool Recommendation

**Location:** `tests/test_selector.py`

**Run:**
```bash
PYTHONPATH=. pytest tests/test_selector.py -v
```

**Test Classes:**

#### TestSelectPdf (10 tests)

| Test | Description | Expected |
|------|-------------|----------|
| `test_empty_list_returns_none` | Empty PDF list | Returns `None` |
| `test_single_pdf_auto_selects` | Single PDF in list | Auto-selects that PDF |
| `test_valid_choice_selects_correct_pdf` | User picks "2" | Returns 2nd PDF |
| `test_first_choice_selects_first_pdf` | User picks "1" | Returns 1st PDF |
| `test_last_choice_selects_last_pdf` | User picks "3" | Returns 3rd PDF |
| `test_invalid_choice_defaults_to_first` | User picks "invalid" | Returns 1st PDF |
| `test_out_of_range_choice_defaults_to_first` | User picks "99" | Returns 1st PDF |
| `test_zero_choice_defaults_to_first` | User picks "0" | Returns 1st PDF |
| `test_negative_choice_defaults_to_first` | User picks "-1" | Returns 1st PDF |
| `test_none_choice_defaults_to_first` | No choice provided | Returns 1st PDF |

#### TestCalculateToolScores (8 tests)

| Test | Input | Expected Winner |
|------|-------|-----------------|
| `test_math_content_favors_nougat` | math + scanned + quality | nougat highest |
| `test_tables_content_favors_docling` | tables + digital + balance | docling highest |
| `test_text_content_favors_marker` | text + digital + speed | marker highest |
| `test_speed_priority_boosts_marker` | Compare speed vs quality | marker higher with speed |
| `test_quality_priority_boosts_nougat_and_docling` | Compare quality vs speed | nougat/docling higher with quality |
| `test_scanned_pdf_boosts_nougat_and_docling` | Compare scanned vs digital | nougat/docling higher with scanned |
| `test_all_scores_are_non_negative` | All combinations | All scores >= 0 |
| `test_unknown_content_type_still_works` | Unknown content type | Returns valid scores |

#### TestGetToolRecommendation (5 tests)

| Test | Inputs | Expected |
|------|--------|----------|
| `test_math_heavy_recommends_nougat` | math, scanned, quality | "nougat" |
| `test_tables_heavy_recommends_docling` | tables, digital, quality | "docling" |
| `test_text_with_speed_recommends_marker` | text, digital, speed | "marker" |
| `test_mixed_content_recommends_docling` | mixed, mixed, balance | "docling" |
| `test_returns_valid_tool_name` | All combinations | One of: marker, docling, nougat |

#### TestScoringRulesConsistency (3 tests)

| Test | Description |
|------|-------------|
| `test_all_tools_in_content_type_rules` | All tools scored for each content type |
| `test_all_tools_in_pdf_type_rules` | All tools scored for each PDF type |
| `test_all_tools_in_priority_rules` | All tools scored for each priority |

---

### 2. test_analyzer.py - Markdown Quality Analysis

**Location:** `tests/test_analyzer.py`

**Run:**
```bash
PYTHONPATH=. pytest tests/test_analyzer.py -v
```

**Test Classes:**

#### TestGetContentStats (11 tests)

| Test | What It Checks |
|------|----------------|
| `test_counts_characters` | Total character count |
| `test_counts_words` | Word count matches `split()` |
| `test_counts_lines` | Line count via `\n` counting |
| `test_counts_headings` | Markdown headings (`#`, `##`, etc.) |
| `test_counts_tables` | Table separators (`\|---`) |
| `test_counts_code_blocks` | Code block pairs (` ``` `) |
| `test_counts_images` | Image references (`![](...)`) |
| `test_counts_math_blocks` | Math blocks (`$$` and `\[`) |
| `test_counts_links_excluding_images` | Links minus image refs |
| `test_empty_content_returns_zeros` | Empty string → zero stats |
| `test_estimated_pages_calculation` | Words / 300 = pages |

#### TestDetectQualityIssues (6 tests)

| Test | Condition | Expected Issue |
|------|-----------|----------------|
| `test_no_issues_for_good_content` | Well-formed markdown | No issues |
| `test_detects_short_content` | < 1000 words | "Very short output" |
| `test_detects_high_special_char_ratio` | > 15% special chars | "High special character ratio" |
| `test_detects_broken_tables` | Low pipe-to-table ratio | "Possible broken table" |
| `test_empty_content_reports_issues` | Empty content | At least 1 issue |
| `test_detects_excessive_empty_lines` | Many `\n\n\n` | "Excessive empty lines" |

#### TestAnalyzeQuality (6 tests)

| Test | Description |
|------|-------------|
| `test_returns_quality_report` | Returns QualityReport object |
| `test_report_contains_filename` | Filename stored in report |
| `test_report_contains_stats` | ContentStats in report |
| `test_report_has_issues_property` | `has_issues` property works |
| `test_good_content_has_no_issues` | Good content → `has_issues=False` |
| `test_default_filename` | Default is "unknown" |

#### TestPreviewMarkdown (5 tests)

| Test | Input | Expected |
|------|-------|----------|
| `test_short_content_not_truncated` | < limit | Full content |
| `test_long_content_is_truncated` | > limit | Truncated |
| `test_truncated_content_has_indicator` | > limit | Contains "[truncated..." |
| `test_exact_length_not_truncated` | = limit | Not truncated |
| `test_default_limit_is_3000` | No limit arg | 3000 chars |

#### TestFindLatestMarkdown (5 tests)

| Test | Condition | Expected |
|------|-----------|----------|
| `test_finds_markdown_file` | .md exists | Returns Path |
| `test_returns_none_for_empty_directory` | No files | Returns None |
| `test_returns_none_for_nonexistent_directory` | Dir not exist | Returns None |
| `test_finds_most_recent_file` | Multiple .md | Returns newest |
| `test_finds_nested_markdown` | .md in subdir | Finds it |

---

### 3. test_converter.py - PDF Conversion Logic

**Location:** `tests/test_converter.py`

**Run:**
```bash
PYTHONPATH=. pytest tests/test_converter.py -v
```

**Test Classes:**

#### TestValidatePdfPath (4 tests)

| Test | Input | Expected |
|------|-------|----------|
| `test_valid_pdf_path` | Existing PDF | `(True, "")` |
| `test_nonexistent_path` | Non-existent | `(False, "not found...")` |
| `test_directory_path` | Directory | `(False, "not a file...")` |
| `test_non_pdf_file` | .txt file | `(False, "not a PDF...")` |

#### TestValidateTool (3 tests)

| Test | Input | Expected |
|------|-------|----------|
| `test_valid_tools` | marker/docling/nougat | `(True, "")` |
| `test_invalid_tool` | "unknown" | `(False, "Unsupported...")` |
| `test_case_sensitive` | "MARKER" | `(False, ...)` (case matters) |

#### TestFindOutputFile (6 tests)

| Test | Setup | Expected |
|------|-------|----------|
| `test_finds_docling_output` | `doc_docling.md` exists | Finds it |
| `test_finds_marker_output` | `doc.md` exists | Finds it |
| `test_finds_nougat_mmd_output` | `doc.mmd` exists | Finds it |
| `test_finds_nested_output` | .md in subdir | Finds it |
| `test_returns_none_when_not_found` | No matching file | Returns None |
| `test_finds_most_recent_fallback` | Multiple .md | Returns newest |

#### TestConvertPdf (5 tests)

| Test | Condition | Expected |
|------|-----------|----------|
| `test_invalid_pdf_returns_failure` | Non-existent PDF | `success=False` |
| `test_invalid_tool_returns_failure` | Unknown tool | `success=False` |
| `test_dry_run_validates_only` | `dry_run=True` | Validates only, no convert |
| `test_creates_output_directory` | Dir doesn't exist | Creates it |
| `test_returns_conversion_result` | Any valid input | Returns ConversionResult |

#### TestConvertPdfWithMocking (5 tests)

| Test | Mock Scenario | Validates |
|------|---------------|-----------|
| `test_marker_command_structure` | Mock subprocess | Correct CLI args for marker |
| `test_nougat_command_structure` | Mock subprocess | Correct CLI args for nougat |
| `test_handles_command_failure` | returncode=1 | Error message captured |
| `test_handles_missing_tool` | FileNotFoundError | "not installed" message |
| `test_handles_timeout` | TimeoutExpired | "timed out" message |

#### TestToolMeta (2 tests)

| Test | Description |
|------|-------------|
| `test_all_supported_tools_have_meta` | All tools have metadata entry |
| `test_meta_has_required_fields` | Each has (name, desc, time) |

---

## Test Fixtures

**Location:** `tests/conftest.py`

| Fixture | Type | Description |
|---------|------|-------------|
| `temp_dir` | Path | Temporary directory, auto-cleaned |
| `sample_pdf_paths` | List[Path] | 3 mock PDF files |
| `single_pdf_path` | List[Path] | 1 mock PDF file |
| `sample_markdown` | str | Realistic markdown (>1000 words) |
| `short_markdown` | str | Very short content |
| `garbled_markdown` | str | High special char ratio |
| `markdown_with_tables` | str | Multiple tables |
| `output_dir_with_markdown` | Path | Dir with .md file |

---

## Running Specific Tests

```bash
# Single file
PYTHONPATH=. pytest tests/test_selector.py -v

# Single class
PYTHONPATH=. pytest tests/test_analyzer.py::TestGetContentStats -v

# Single test
PYTHONPATH=. pytest tests/test_selector.py::TestSelectPdf::test_empty_list_returns_none -v

# Pattern match
PYTHONPATH=. pytest tests/ -k "math" -v

# With print output
PYTHONPATH=. pytest tests/ -v -s

# Stop on first failure
PYTHONPATH=. pytest tests/ -x

# Run last failed
PYTHONPATH=. pytest tests/ --lf
```

---

## Understanding Output

### Passed Test
```
tests/test_selector.py::TestSelectPdf::test_empty_list_returns_none PASSED [  1%]
```

### Failed Test
```
tests/test_analyzer.py::TestGetContentStats::test_counts_tables FAILED [  5%]

=================================== FAILURES ===================================
____________________ TestGetContentStats.test_counts_tables ____________________
tests/test_analyzer.py:44: in test_counts_tables
    assert stats.tables == 3
E   assert 6 == 3
```

### Status Meanings
- `PASSED` - Test succeeded
- `FAILED` - Assertion failed
- `ERROR` - Exception raised
- `SKIPPED` - Test skipped (marked or condition)
- `XFAIL` - Expected failure
- `XPASS` - Expected failure but passed

---

## Coverage Interpretation

```
Name                             Stmts   Miss  Cover   Missing
--------------------------------------------------------------
src/pdf_converter/converter.py      86     21    76%   170-194
```

| Column | Meaning |
|--------|---------|
| Stmts | Total executable statements |
| Miss | Statements not executed |
| Cover | Percentage covered |
| Missing | Line numbers not covered |

### Improving Coverage

Lines 170-194 in converter.py are the actual subprocess calls to external tools (marker, nougat). These are mocked in tests because the actual tools aren't installed in the test environment.

---

## Next Phases

Additional test modules will be added for each phase:

| Phase | Test File | Status |
|-------|-----------|--------|
| 1. PDF Converter | `test_selector.py`, `test_analyzer.py`, `test_converter.py` | Done |
| 2. LlamaIndex Chunking | `test_chunking.py` | Pending |
| 3. Embedding Generation | `test_embedding.py` | Pending |
| 4. Helix-DB Integration | `test_database.py` | Pending |
| 5. MCP Server | `test_mcp.py` | Pending |
