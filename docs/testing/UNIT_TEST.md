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

```text
========================= test session starts ==========================
platform linux -- Python 3.13.11, pytest-9.0.2
collected 223 items

tests/test_analyzer.py   ............................ [ 12%]
tests/test_chunker.py    ........................................... [ 31%]
tests/test_converter.py  ........................    [ 42%]
tests/test_embedder.py   ...........................  [ 54%]
tests/test_metadata.py   .......................................  [ 72%]
tests/test_parser.py     ............................  [ 84%]
tests/test_selector.py   ...........................  [100%]

========================= 223 passed in 0.39s ===========================
```

### Coverage Report

```text
Name                             Stmts   Miss  Cover   Missing
--------------------------------------------------------------
src/chunker/__init__.py              5      0   100%
src/chunker/chunker.py             185     25    86%
src/chunker/embedder.py            103     31    70%
src/chunker/metadata.py             93      0   100%
src/chunker/parser.py              141      0   100%
src/pdf_converter/__init__.py        4      0   100%
src/pdf_converter/analyzer.py       61      1    98%   93
src/pdf_converter/converter.py      86     21    76%   170-194, 204, 231, 240-242, 264-265
src/pdf_converter/selector.py       32      0   100%
--------------------------------------------------------------
TOTAL                              710     78    89%
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
```text
tests/test_selector.py::TestSelectPdf::test_empty_list_returns_none PASSED [  1%]
```

### Failed Test
```text
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

```text
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

## Phase 2: Hierarchical Chunking

### 4. test_parser.py - Markdown Structure Parsing

**Location:** `tests/test_parser.py`

**Run:**
```bash
PYTHONPATH=. pytest tests/test_parser.py -v
```

**Test Classes:**

#### TestParseMarkdownStructure (12 tests)

| Test | Description |
|------|-------------|
| `test_empty_content` | Handle empty content |
| `test_content_without_headings` | Single root with all content |
| `test_single_heading` | Parse single heading |
| `test_multiple_headings_same_level` | Multiple h1 headings |
| `test_nested_headings` | h2 under h1 nesting |
| `test_deeply_nested_headings` | h1 > h2 > h3 nesting |
| `test_sibling_sections_with_children` | Multiple h2 under h1 |
| `test_content_before_first_heading` | Capture intro content |
| `test_heading_with_special_chars` | Headings with `code` etc. |
| `test_full_content_property` | Returns content with children |
| `test_word_count_property` | Calculate word count |

#### TestExtractSpecialBlocks (8 tests)

| Test | Block Type | Expected |
|------|------------|----------|
| `test_empty_content` | Empty | Empty list |
| `test_code_block_backticks` | Code (```) | Extracts with language |
| `test_code_block_tildes` | Code (~~~) | Extracts code |
| `test_table_detection` | Table | Extracts table |
| `test_math_block_double_dollar` | Math ($$) | Extracts math |
| `test_math_block_latex_brackets` | Math (\\[\\]) | Extracts math |
| `test_multiple_blocks` | Mixed | All types extracted |
| `test_preserves_line_numbers` | Any | Start/end lines tracked |

#### TestBuildSectionPath, TestFlattenSections, TestGetSectionByLine, TestGetHeadingHierarchy (8 tests)

| Test | Description |
|------|-------------|
| `test_path_to_child` | Build path from root to child |
| `test_path_to_root_child` | Path for direct child of root |
| `test_path_not_found` | Empty list if target not found |
| `test_empty_document` | Empty list for no headings |
| `test_flattens_nested_structure` | Flatten in document order |
| `test_finds_section` | Find section by line number |
| `test_returns_none_for_root_only` | None for content before headings |
| `test_returns_heading` | Return section heading |

---

### 5. test_chunker.py - Hierarchical Chunking Logic

**Location:** `tests/test_chunker.py`

**Run:**
```bash
PYTHONPATH=. pytest tests/test_chunker.py -v
```

**Test Classes:**

#### TestEstimateTokens (4 tests)

| Test | Description |
|------|-------------|
| `test_empty_string` | Returns 0 for empty |
| `test_short_text` | Estimate for short text |
| `test_longer_text` | Estimate for longer text |
| `test_proportional_to_length` | Longer = more tokens |

#### TestSplitTextWithOverlap (5 tests)

| Test | Description |
|------|-------------|
| `test_empty_text` | Empty list for empty |
| `test_whitespace_only` | Empty list for whitespace |
| `test_short_text_single_chunk` | Single chunk for short |
| `test_splits_long_text` | Multiple chunks for long |
| `test_preserves_sentences` | Split at sentence boundaries |

#### TestCreateHierarchicalChunks (12 tests)

| Test | Description |
|------|-------------|
| `test_empty_content` | Empty list for empty |
| `test_simple_document` | Creates chunks |
| `test_creates_level2_chunks` | Creates leaf chunks |
| `test_creates_level1_for_large_sections` | Creates parent chunks |
| `test_parent_child_relationships` | Links parent-child |
| `test_prev_next_relationships` | Links prev-next |
| `test_section_path_populated` | Section path set |
| `test_respects_doc_id_prefix` | Uses doc_id in IDs |
| `test_handles_code_blocks` | Preserves code blocks |
| `test_handles_tables` | Preserves tables |
| `test_intro_content_captured` | Captures intro content |

#### TestGetChunkById, TestGetParentChunk, TestGetChildChunks, TestGetSiblingChunks, TestGetContextWindow (12 tests)

Navigation and context retrieval tests for chunk relationships.

---

### 6. test_metadata.py - Metadata Enrichment

**Location:** `tests/test_metadata.py`

**Run:**
```bash
PYTHONPATH=. pytest tests/test_metadata.py -v
```

**Test Classes:**

#### TestInferSemanticType (10 tests)

| Test | Content Pattern | Expected Type |
|------|-----------------|---------------|
| `test_detects_definition` | "is defined as" | definition |
| `test_detects_definition_is_a` | "is a" | definition |
| `test_detects_example` | "For example" | example |
| `test_detects_example_eg` | "For instance" | example |
| `test_detects_procedure` | "First... Then..." | procedure |
| `test_detects_theorem` | "Theorem 1:" | theorem |
| `test_detects_proof` | "Proof:" | theorem |
| `test_defaults_to_narrative` | General text | narrative |
| `test_empty_content` | Empty | narrative |

#### TestExtractTopicTags (7 tests)

| Test | Description |
|------|-------------|
| `test_extracts_bold_text` | **bold** terms |
| `test_extracts_italic_text` | _italic_ terms |
| `test_extracts_defined_terms` | "is a" patterns |
| `test_extracts_capitalized_phrases` | Multi-word caps |
| `test_limits_max_tags` | Respects max_tags |
| `test_empty_content` | Empty list for empty |
| `test_no_duplicates` | No duplicate tags |

#### TestDetectContentFeatures (8 tests)

| Test | Feature | Detection |
|------|---------|-----------|
| `test_detects_table` | has_table | Markdown table |
| `test_detects_code_block` | has_code | ``` blocks |
| `test_detects_inline_code` | has_code | `inline` |
| `test_detects_math_dollar` | has_math | $ math $ |
| `test_detects_math_double_dollar` | has_math | $$ blocks |
| `test_detects_bullet_list` | has_list | - items |
| `test_detects_numbered_list` | has_list | 1. items |
| `test_no_features` | All false | Plain text |

#### TestChunkMetadata, TestEnrichChunkMetadata, TestEnrichAllChunks, TestFilterChunksByType, TestFilterChunksByTopic (17 tests)

Metadata creation, enrichment, and filtering tests.

---

### 7. test_embedder.py - Embedding Generation

**Location:** `tests/test_embedder.py`

**Run:**
```bash
PYTHONPATH=. pytest tests/test_embedder.py -v
```

**Test Classes:**

#### TestMockEmbedder (13 tests)

| Test | Description |
|------|-------------|
| `test_initialization` | Default values |
| `test_custom_dimension` | Custom embedding dim |
| `test_embed_text_returns_correct_dimension` | Correct output size |
| `test_embed_text_deterministic` | Same text = same embedding |
| `test_embed_text_different_for_different_text` | Different text = different |
| `test_embed_empty_text` | Zero vector for empty |
| `test_embeddings_are_normalized` | Unit vectors |
| `test_embed_texts_generator` | Multiple texts |
| `test_embed_chunk` | Single chunk |
| `test_embed_chunks` | Multiple chunks |
| `test_embed_chunks_empty_list` | Empty list handling |

#### TestCosineSimilarity (6 tests)

| Test | Vectors | Expected |
|------|---------|----------|
| `test_identical_vectors` | Same | 1.0 |
| `test_orthogonal_vectors` | Perpendicular | 0.0 |
| `test_opposite_vectors` | Opposite | -1.0 |
| `test_similar_vectors` | Similar | > 0.9 |
| `test_zero_vector` | Zero | 0.0 |
| `test_different_dimensions_raises` | Mismatch | ValueError |

#### TestFindSimilarChunks (5 tests)

| Test | Description |
|------|-------------|
| `test_finds_most_similar` | Exact match first |
| `test_respects_top_k` | Limits results |
| `test_respects_threshold` | Filters by similarity |
| `test_empty_chunks_list` | Empty for empty |
| `test_returns_tuples` | (EmbeddedChunk, score) |

#### TestChunkEmbedderInitialization (5 tests)

| Test | Description |
|------|-------------|
| `test_initialization` | No model load initially |
| `test_custom_model_name` | Custom model name |
| `test_get_embedding_dim_without_init` | Returns dim without loading |
| `test_get_embedding_dim_custom_model` | Correct dims for models |

---

## Next Phases

| Phase | Test Files | Status |
|-------|-----------|--------|
| 1. PDF Converter | `test_selector.py`, `test_analyzer.py`, `test_converter.py` | Done |
| 2. Hierarchical Chunking | `test_parser.py`, `test_chunker.py`, `test_metadata.py`, `test_embedder.py` | Done |
| 3. Helix-DB Integration | `test_database.py` | Pending |
| 4. MCP Server | `test_mcp.py` | Pending |
