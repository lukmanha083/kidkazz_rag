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
collected 567 items

tests/test_analyzer.py              ............................           [  5%]
tests/test_chunker.py               ...........................................  [ 12%]
tests/test_cli_commands.py          ....................................   [ 19%]
tests/test_cli_config.py            ............................           [ 24%]
tests/test_cli_integration.py       ....................................   [ 30%]
tests/test_cli_output.py            ..........................             [ 35%]
tests/test_cli_progress.py          .........................              [ 39%]
tests/test_cli_utils.py             .......................                [ 43%]
tests/test_converter.py             ........................               [ 48%]
tests/test_embedder.py              ...........................            [ 52%]
tests/test_metadata.py              .......................................  [ 59%]
tests/test_mcp_config.py            ..............                         [ 62%]
tests/test_mcp_formatters.py        .....................                  [ 65%]
tests/test_mcp_integration.py       .............                          [ 68%]
tests/test_mcp_resources.py         .............                          [ 70%]
tests/test_mcp_server.py            ..ssssssss                             [ 72%]
tests/test_mcp_tools.py             .................................      [ 78%]
tests/test_parser.py                ............................           [ 83%]
tests/test_selector.py              ...........................            [ 88%]
tests/test_storage_converters.py    ........................               [ 92%]
tests/test_storage_integration.py   ...............                        [ 95%]
tests/test_storage_mock.py          ..................................     [ 97%]
tests/test_storage_queries.py       ....................                   [ 99%]
tests/test_storage_schema.py        ............................           [100%]

======================== 557 passed, 10 skipped in 1.10s ================
```

Note: 10 tests are skipped because they require the MCP package to be installed.

### Coverage Report

```text
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/chunker/__init__.py                     5      0   100%
src/chunker/chunker.py                    185     25    86%
src/chunker/embedder.py                   103     31    70%
src/chunker/metadata.py                    93      0   100%
src/chunker/parser.py                     141      0   100%
src/cli/__init__.py                         4      0   100%
src/cli/config.py                         156     12    92%
src/cli/main.py                            32      2    94%
src/cli/output.py                         124      8    94%
src/cli/progress.py                        68      4    94%
src/cli/utils.py                           85      6    93%
src/cli/commands/__init__.py                1      0   100%
src/cli/commands/config_cmd.py             92     12    87%
src/cli/commands/db.py                     78      8    90%
src/cli/commands/docs.py                  142     18    87%
src/cli/commands/ingest.py                168     22    87%
src/cli/commands/search.py                 86     10    88%
src/mcp_server/__init__.py                  8      0   100%
src/mcp_server/__main__.py                 16     16     0%   (entry point)
src/mcp_server/config.py                   74      4    95%
src/mcp_server/formatters.py               42      0   100%
src/mcp_server/resources.py                44      0   100%
src/mcp_server/server.py                   28     10    64%   (requires MCP)
src/mcp_server/tools.py                    85      0   100%
src/pdf_converter/__init__.py               4      0   100%
src/pdf_converter/analyzer.py              61      1    98%
src/pdf_converter/converter.py             86     21    76%
src/pdf_converter/selector.py              32      0   100%
src/storage/__init__.py                    12      0   100%
src/storage/client.py                     298    178    40%   (requires Helix-DB)
src/storage/converters.py                  62      0   100%
src/storage/mock_store.py                 176      0   100%
src/storage/queries.py                    168     72    57%
src/storage/schema.py                      88      4    95%
---------------------------------------------------------------------
TOTAL                                    2842    464    84%
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

## Phase 3: Helix-DB Storage Integration

### 8. test_storage_schema.py - Schema Definitions

**Location:** `tests/test_storage_schema.py`

**Run:**
```bash
python -m pytest tests/test_storage_schema.py -v
```

**Test Classes:**

#### TestSchemaConfig (2 tests)

| Test | Description |
|------|-------------|
| `test_default_values` | Default schema configuration |
| `test_custom_values` | Custom embedding dimensions |

#### TestNodeSchemas (5 tests)

| Test | Description |
|------|-------------|
| `test_document_schema_has_required_fields` | Document node has doc_id, title |
| `test_chunk_schema_has_required_fields` | Chunk node has chunk_id, content |
| `test_chunk_schema_has_relationship_fields` | Chunk has parent_id, child_ids |
| `test_chunk_schema_has_metadata_fields` | Chunk has level, semantic_type |
| `test_vector_schema_has_embedding` | Vector node has embedding field |

#### TestEdgeTypes (4 tests)

| Test | Description |
|------|-------------|
| `test_has_document_to_chunk_edge` | HasChunk edge type |
| `test_has_parent_child_edge` | ParentOf edge type |
| `test_has_sequential_edges` | NextSibling edge type |
| `test_has_embedding_edge` | HasEmbedding edge type |

#### TestValidateChunkNode, TestValidateDocumentNode (10 tests)

| Test | Description |
|------|-------------|
| `test_valid_chunk_returns_empty_list` | Valid chunk passes |
| `test_missing_required_field` | Missing field detected |
| `test_invalid_level_type` | Type validation |
| `test_invalid_level_value` | Value range validation |
| `test_invalid_semantic_type` | Enum validation |
| `test_valid_semantic_types` | All semantic types work |

---

### 9. test_storage_converters.py - Data Conversion

**Location:** `tests/test_storage_converters.py`

**Run:**
```bash
python -m pytest tests/test_storage_converters.py -v
```

**Test Classes:**

#### TestChunkToHelixNode (8 tests)

| Test | Description |
|------|-------------|
| `test_basic_conversion` | Converts chunk to Helix node |
| `test_includes_all_fields` | All fields preserved |
| `test_handles_optional_fields` | None values handled |
| `test_converts_child_ids_to_json` | Lists serialized |

#### TestHelixNodeToChunk (8 tests)

| Test | Description |
|------|-------------|
| `test_basic_conversion` | Converts Helix node to chunk |
| `test_preserves_relationships` | Parent/child links preserved |
| `test_handles_missing_fields` | Defaults for missing |
| `test_preserves_semantic_type` | Metadata preserved |

#### TestMetadataConversion (8 tests)

| Test | Description |
|------|-------------|
| `test_metadata_to_helix` | ChunkMetadata to Helix format |
| `test_helix_to_metadata` | Helix format to ChunkMetadata |
| `test_round_trip` | Convert both ways |

---

### 10. test_storage_mock.py - MockChunkStore

**Location:** `tests/test_storage_mock.py`

**Run:**
```bash
python -m pytest tests/test_storage_mock.py -v
```

**Test Classes:**

#### TestStoreDocument (6 tests)

| Test | Description |
|------|-------------|
| `test_stores_document` | Store document with chunks |
| `test_stores_chunks` | All chunks stored |
| `test_stores_embeddings` | Embeddings preserved |
| `test_stores_metadata` | Metadata preserved |
| `test_updates_existing` | Overwrite existing doc |

#### TestSearchSimilar (8 tests)

| Test | Description |
|------|-------------|
| `test_returns_similar_chunks` | Vector search works |
| `test_respects_top_k` | Limits results |
| `test_respects_threshold` | Filters by similarity |
| `test_filters_by_doc_id` | Document filter |
| `test_filters_by_level` | Level filter |
| `test_filters_by_semantic_type` | Type filter |
| `test_returns_scores` | Similarity scores included |

#### TestSearchKeyword (6 tests)

| Test | Description |
|------|-------------|
| `test_finds_keyword` | Keyword search works |
| `test_case_insensitive` | Default case-insensitive |
| `test_case_sensitive` | Optional case-sensitive |
| `test_filters_by_doc_id` | Document filter |
| `test_no_match_returns_empty` | Empty for no match |

#### TestGraphTraversal (10 tests)

| Test | Description |
|------|-------------|
| `test_get_parent` | Navigate to parent |
| `test_get_children` | Get child chunks |
| `test_get_siblings` | Get sibling chunks |
| `test_get_context_window` | Get surrounding chunks |
| `test_get_chunk` | Get by ID |

#### TestDocumentManagement (4 tests)

| Test | Description |
|------|-------------|
| `test_list_documents` | List all documents |
| `test_get_document_chunks` | Get all chunks from doc |
| `test_get_document_stats` | Get document statistics |
| `test_delete_document` | Remove document |

---

### 11. test_storage_queries.py - HelixQL Queries

**Location:** `tests/test_storage_queries.py`

**Run:**
```bash
python -m pytest tests/test_storage_queries.py -v
```

**Test Classes:**

#### TestSearchSimilarChunks (4 tests)

| Test | Description |
|------|-------------|
| `test_creates_vector_search_payload` | Correct query structure |
| `test_includes_filters` | Filters in payload |
| `test_omits_empty_filters` | No empty filters |

#### TestSearchKeyword (3 tests)

| Test | Description |
|------|-------------|
| `test_creates_keyword_search_payload` | Correct query structure |
| `test_includes_doc_filter` | Document filter |
| `test_includes_case_sensitive` | Case sensitivity flag |

#### TestGetChunk, TestGetDocumentChunks, TestListDocuments (8 tests)

Query construction tests for retrieval operations.

---

### 12. test_storage_integration.py - End-to-End Storage

**Location:** `tests/test_storage_integration.py`

**Run:**
```bash
python -m pytest tests/test_storage_integration.py -v
```

**Test Classes:**

#### TestStoreAndRetrieve (5 tests)

| Test | Description |
|------|-------------|
| `test_store_and_search` | Store then search |
| `test_store_and_traverse` | Store then navigate |
| `test_multiple_documents` | Multiple docs |

#### TestSearchWorkflows (5 tests)

| Test | Description |
|------|-------------|
| `test_semantic_search_workflow` | Vector search flow |
| `test_keyword_search_workflow` | Keyword search flow |
| `test_hybrid_search` | Combined search |

#### TestGraphWorkflows (5 tests)

| Test | Description |
|------|-------------|
| `test_parent_child_navigation` | Navigate hierarchy |
| `test_context_expansion` | Expand context |
| `test_sibling_exploration` | Explore siblings |

---

## Phase 4: MCP Server

### 13. test_mcp_config.py - Configuration

**Location:** `tests/test_mcp_config.py`

**Run:**
```bash
python -m pytest tests/test_mcp_config.py -v
```

**Test Classes:**

#### TestMCPServerConfig (6 tests)

| Test | Description |
|------|-------------|
| `test_default_values` | Default configuration |
| `test_custom_values` | Custom configuration |
| `test_from_env_defaults` | Environment defaults |
| `test_from_env_with_values` | Read from environment |
| `test_create_mock_store` | Factory creates MockChunkStore |
| `test_create_mock_embedder` | Factory creates MockEmbedder |

#### TestLazyEmbedder (4 tests)

| Test | Description |
|------|-------------|
| `test_lazy_initialization` | Not initialized until used |
| `test_embed_text_initializes` | Initializes on first call |
| `test_embed_text_reuses_embedder` | Reuses instance |
| `test_model_name_property` | Returns model name |

#### TestServerState (4 tests)

| Test | Description |
|------|-------------|
| `test_default_config` | Uses from_env if none provided |
| `test_custom_config` | Uses provided config |
| `test_lazy_store` | Store lazy-loaded |
| `test_lazy_embedder` | Embedder lazy-loaded |

---

### 14. test_mcp_formatters.py - Response Formatting

**Location:** `tests/test_mcp_formatters.py`

**Run:**
```bash
python -m pytest tests/test_mcp_formatters.py -v
```

**Test Classes:**

#### TestFormatChunk (6 tests)

| Test | Description |
|------|-------------|
| `test_basic_fields` | chunk_id, content, level |
| `test_word_count` | Word count included |
| `test_relationship_fields` | parent_id, child_ids, prev_id, next_id |
| `test_section_fields` | section_path, source_section |
| `test_embedding_fields` | model_name, embedding_dim |
| `test_embedding_not_included` | Raw embedding excluded |

#### TestFormatSearchResult (5 tests)

| Test | Description |
|------|-------------|
| `test_includes_chunk_fields` | All chunk fields |
| `test_includes_similarity_score` | Score included |
| `test_score_rounding` | Rounded to 4 decimals |
| `test_zero_score` | Handles 0.0 |
| `test_perfect_score` | Handles 1.0 |

#### TestFormatDocument, TestFormatDocumentStats (4 tests)

| Test | Description |
|------|-------------|
| `test_all_fields` | All fields formatted |
| `test_missing_fields` | Defaults for missing |

#### TestFormatChunkList, TestFormatSearchResults, TestFormatOptionalChunk (6 tests)

List and optional formatting tests.

---

### 15. test_mcp_tools.py - Tool Implementations

**Location:** `tests/test_mcp_tools.py`

**Run:**
```bash
python -m pytest tests/test_mcp_tools.py -v
```

**Test Classes:**

#### TestSearchSemantic (6 tests)

| Test | Description |
|------|-------------|
| `test_basic_search` | Basic semantic search |
| `test_search_with_doc_filter` | Filter by document |
| `test_search_with_level_filter` | Filter by level |
| `test_search_with_type_filter` | Filter by semantic type |
| `test_search_with_threshold` | Similarity threshold |
| `test_search_returns_scores` | Scores in results |

#### TestSearchKeyword (4 tests)

| Test | Description |
|------|-------------|
| `test_basic_keyword_search` | Basic keyword search |
| `test_keyword_search_with_doc_filter` | Filter by document |
| `test_case_insensitive_search` | Default case-insensitive |
| `test_case_sensitive_search` | Optional case-sensitive |

#### TestGetChunk (3 tests)

| Test | Description |
|------|-------------|
| `test_get_existing_chunk` | Get by ID |
| `test_get_nonexistent_chunk` | Returns None |
| `test_chunk_has_content` | Content included |

#### TestGetContextWindow (3 tests)

| Test | Description |
|------|-------------|
| `test_get_context_window` | Get with neighbors |
| `test_context_window_default_size` | Default window_size=2 |
| `test_context_window_includes_neighbors` | Neighbors included |

#### TestGetParent, TestGetChildren, TestGetSiblings (8 tests)

| Test | Description |
|------|-------------|
| `test_get_parent_of_child` | Navigate to parent |
| `test_get_parent_of_root` | Returns None for root |
| `test_get_children_of_parent` | Get all children |
| `test_get_children_of_leaf` | Empty for leaf |
| `test_children_have_correct_parent` | Parent ID matches |
| `test_get_siblings` | Get sibling chunks |
| `test_siblings_exclude_self` | Self not included |
| `test_siblings_same_parent` | Same parent |

#### TestListDocuments, TestGetDocumentChunks, TestGetDocumentStats (9 tests)

Document management tool tests.

---

### 16. test_mcp_resources.py - Resource Implementations

**Location:** `tests/test_mcp_resources.py`

**Run:**
```bash
python -m pytest tests/test_mcp_resources.py -v
```

**Test Classes:**

#### TestSchemaInfo (5 tests)

| Test | Description |
|------|-------------|
| `test_schema_has_name` | Name field present |
| `test_schema_has_version` | Version field present |
| `test_schema_has_chunk_levels` | Chunk levels documented |
| `test_schema_has_semantic_types` | Semantic types listed |
| `test_schema_has_tools` | Available tools listed |

#### TestGetSchemaResource (2 tests)

| Test | Description |
|------|-------------|
| `test_returns_json` | Returns valid JSON |
| `test_returns_schema_info` | Contains schema info |

#### TestGetDocumentsResource (2 tests)

| Test | Description |
|------|-------------|
| `test_returns_json` | Returns valid JSON |
| `test_returns_document_list` | Contains document list |

#### TestGetDocumentOverviewResource, TestGetChunkContentResource (4 tests)

| Test | Description |
|------|-------------|
| `test_existing_document` | Returns document overview |
| `test_nonexistent_document` | Returns error |
| `test_existing_chunk` | Returns chunk content |
| `test_nonexistent_chunk` | Returns error |

---

### 17. test_mcp_server.py - Server Initialization

**Location:** `tests/test_mcp_server.py`

**Run:**
```bash
python -m pytest tests/test_mcp_server.py -v
```

**Note:** Most tests require MCP package installed. Tests are skipped if not available.

**Test Classes:**

#### TestCreateServer (4 tests - require MCP)

| Test | Description |
|------|-------------|
| `test_creates_fastmcp_instance` | Returns FastMCP instance |
| `test_uses_default_config` | Uses from_env if none |
| `test_uses_provided_state` | Uses provided state |
| `test_server_has_name` | Server has name attribute |

#### TestCreateServerWithoutMCP (1 test)

| Test | Description |
|------|-------------|
| `test_import_error_without_mcp` | Function exists |

#### TestRunServer (1 test)

| Test | Description |
|------|-------------|
| `test_run_server_calls_run` | Calls mcp.run() |

#### TestServerIntegration (4 tests - require MCP)

| Test | Description |
|------|-------------|
| `test_tools_registered` | Tools registered |
| `test_resources_registered` | Resources registered |
| `test_server_with_mock_store` | Works with mock store |
| `test_server_lazy_loading` | Lazy loading works |

---

### 18. test_mcp_integration.py - End-to-End MCP Workflows

**Location:** `tests/test_mcp_integration.py`

**Run:**
```bash
python -m pytest tests/test_mcp_integration.py -v
```

**Test Classes:**

#### TestSearchWorkflow (2 tests)

| Test | Description |
|------|-------------|
| `test_search_and_expand_context` | Search then expand context |
| `test_search_and_navigate_hierarchy` | Search then navigate |

#### TestDocumentBrowsingWorkflow (2 tests)

| Test | Description |
|------|-------------|
| `test_browse_documents_and_stats` | List docs and get stats |
| `test_browse_document_structure` | Browse hierarchy |

#### TestKeywordSearchWorkflow (1 test)

| Test | Description |
|------|-------------|
| `test_keyword_search_and_retrieve` | Keyword search then get chunk |

#### TestResourceWorkflow (3 tests)

| Test | Description |
|------|-------------|
| `test_schema_discovery` | Discover tools via schema |
| `test_document_discovery` | Discover docs via resource |
| `test_chunk_resource` | Access chunk via resource |

#### TestFilteringWorkflows (2 tests)

| Test | Description |
|------|-------------|
| `test_semantic_type_filtering` | Filter by semantic type |
| `test_level_filtering` | Filter by level |

#### TestEdgeCases (3 tests)

| Test | Description |
|------|-------------|
| `test_nonexistent_chunk` | Handle missing chunk |
| `test_nonexistent_document` | Handle missing document |
| `test_empty_search_results` | Handle no results |

---

## Phase 5: CLI Tool

### 19. test_cli_config.py - Configuration Management

**Location:** `tests/test_cli_config.py`

**Run:**
```bash
python -m pytest tests/test_cli_config.py -v
```

**Test Classes:**

#### TestCLIConfigDefaults (6 tests)

| Test | Description |
|------|-------------|
| `test_default_store_type` | Default is "mock" |
| `test_default_helix_port` | Default is 6969 |
| `test_default_embedder_type` | Default is "fastembed" |
| `test_default_model_name` | Default model name |
| `test_default_chunk_sizes` | Default level sizes |
| `test_all_defaults_set` | All defaults have values |

#### TestCLIConfigFromEnv (8 tests)

| Test | Description |
|------|-------------|
| `test_reads_store_type` | Reads KIDKAZZ_STORE_TYPE |
| `test_reads_helix_port` | Reads KIDKAZZ_HELIX_PORT |
| `test_reads_embedder_type` | Reads KIDKAZZ_EMBEDDER_TYPE |
| `test_reads_model_name` | Reads KIDKAZZ_MODEL_NAME |
| `test_ignores_invalid_port` | Invalid port uses default |
| `test_env_overrides_defaults` | Env vars override defaults |

#### TestCLIConfigLoad (6 tests)

| Test | Description |
|------|-------------|
| `test_loads_defaults` | Loads default config |
| `test_loads_from_project_file` | Loads .kidkazz.toml |
| `test_loads_from_user_file` | Loads user config |
| `test_project_overrides_user` | Project config priority |
| `test_env_overrides_file` | Env overrides file |
| `test_merge_configs` | Config merging works |

#### TestConfigFileParsing (8 tests)

| Test | Description |
|------|-------------|
| `test_parse_valid_toml` | Parses valid TOML |
| `test_parse_storage_section` | Parses [storage] section |
| `test_parse_embedder_section` | Parses [embedder] section |
| `test_parse_chunking_section` | Parses [chunking] section |
| `test_ignores_unknown_keys` | Ignores unknown keys |
| `test_handles_missing_file` | Handles missing file |
| `test_handles_invalid_toml` | Handles invalid TOML |

---

### 20. test_cli_utils.py - Utility Functions

**Location:** `tests/test_cli_utils.py`

**Run:**
```bash
python -m pytest tests/test_cli_utils.py -v
```

**Test Classes:**

#### TestParseChunkSizes (5 tests)

| Test | Description |
|------|-------------|
| `test_valid_sizes` | Parses "2048,512,256" |
| `test_different_values` | Parses different values |
| `test_invalid_format_too_few` | Error on 2 values |
| `test_invalid_format_too_many` | Error on 4 values |
| `test_invalid_non_integer` | Error on non-integers |

#### TestResolveDocId (3 tests)

| Test | Description |
|------|-------------|
| `test_explicit_doc_id` | Explicit ID takes precedence |
| `test_doc_id_from_filename` | Uses filename stem |
| `test_doc_id_from_complex_path` | Handles complex paths |

#### TestResolveTitle (5 tests)

| Test | Description |
|------|-------------|
| `test_explicit_title` | Explicit title takes precedence |
| `test_title_from_h1` | Extracts from markdown H1 |
| `test_title_from_h1_with_spaces` | Handles whitespace |
| `test_title_from_filename_no_h1` | Fallback to filename |
| `test_title_from_filename_empty_content` | Handles empty content |

#### TestFormatFileSize (4 tests)

| Test | Description |
|------|-------------|
| `test_bytes` | Formats bytes |
| `test_kilobytes` | Formats KB |
| `test_megabytes` | Formats MB |
| `test_gigabytes` | Formats GB |

#### TestValidateFileExists (5 tests)

| Test | Description |
|------|-------------|
| `test_existing_file` | Validates existing file |
| `test_nonexistent_file` | Error on missing file |
| `test_directory_not_file` | Error on directory |
| `test_wrong_extension` | Error on wrong extension |
| `test_correct_extension` | Validates correct extension |

---

### 21. test_cli_output.py - Output Formatting

**Location:** `tests/test_cli_output.py`

**Run:**
```bash
python -m pytest tests/test_cli_output.py -v
```

**Test Classes:**

#### TestPrintFunctions (4 tests)

| Test | Description |
|------|-------------|
| `test_print_success_runs` | print_success executes |
| `test_print_error_runs` | print_error executes |
| `test_print_warning_runs` | print_warning executes |
| `test_print_info_runs` | print_info executes |

#### TestPrintSearchResults (5 tests)

| Test | Description |
|------|-------------|
| `test_empty_results` | Handles empty results |
| `test_single_result` | Formats single result |
| `test_multiple_results` | Formats multiple results |
| `test_result_with_context` | Formats with context |
| `test_long_content_truncation` | Truncates long content |

#### TestPrintDocumentList (3 tests)

| Test | Description |
|------|-------------|
| `test_empty_document_list` | Handles empty list |
| `test_single_document` | Formats single document |
| `test_multiple_documents` | Formats multiple documents |

#### TestPrintDocumentStats (3 tests)

| Test | Description |
|------|-------------|
| `test_basic_stats` | Basic statistics display |
| `test_stats_with_levels` | Stats with level breakdown |
| `test_stats_with_types` | Stats with type breakdown |

#### TestPrintDbStatus (2 tests)

| Test | Description |
|------|-------------|
| `test_connected_status` | Connected status display |
| `test_disconnected_status` | Disconnected status display |

#### TestPrintJson (3 tests)

| Test | Description |
|------|-------------|
| `test_simple_dict` | Prints simple dict |
| `test_nested_dict` | Prints nested dict |
| `test_list_of_dicts` | Prints list of dicts |

#### TestPrintChunksPreview, TestPrintIngestionSummary (6 tests)

| Test | Description |
|------|-------------|
| `test_empty_chunks` | Handles empty chunks |
| `test_few_chunks` | Preview with few chunks |
| `test_many_chunks_truncated` | Truncates many chunks |
| `test_basic_summary` | Basic ingestion summary |

---

### 22. test_cli_progress.py - Progress Indicators

**Location:** `tests/test_cli_progress.py`

**Run:**
```bash
python -m pytest tests/test_cli_progress.py -v
```

**Test Classes:**

#### TestCreateProgress (2 tests)

| Test | Description |
|------|-------------|
| `test_create_progress_returns_progress` | Returns Progress instance |
| `test_create_detailed_progress_returns_progress` | Returns detailed Progress |

#### TestIngestionProgress (2 tests)

| Test | Description |
|------|-------------|
| `test_context_manager_yields_progress` | Context manager works |
| `test_can_add_and_update_task` | Can add and update tasks |

#### TestSpinner (1 test)

| Test | Description |
|------|-------------|
| `test_spinner_context_manager` | Spinner context works |

#### TestIngestionTracker (6 tests)

| Test | Description |
|------|-------------|
| `test_add_stage` | Adding a stage |
| `test_update_stage` | Updating a stage |
| `test_set_progress` | Setting absolute progress |
| `test_complete_stage` | Completing a stage |
| `test_update_nonexistent_stage` | Nonexistent stage doesn't raise |
| `test_update_description` | Updating stage description |

#### TestBatchTracker (5 tests)

| Test | Description |
|------|-------------|
| `test_context_manager` | Context manager works |
| `test_advance` | Advancing progress |
| `test_advance_with_error` | Advancing with error |
| `test_get_summary` | Getting summary |
| `test_multiple_errors` | Tracking multiple errors |

---

### 23. test_cli_commands.py - Command Tests

**Location:** `tests/test_cli_commands.py`

**Run:**
```bash
python -m pytest tests/test_cli_commands.py -v
```

**Test Classes:**

#### TestMainApp (3 tests)

| Test | Description |
|------|-------------|
| `test_help_shows_commands` | --help shows all commands |
| `test_version_command` | version command works |
| `test_no_args_shows_help` | No args shows help |

#### TestIngestCommands (4 tests)

| Test | Description |
|------|-------------|
| `test_ingest_help` | ingest --help works |
| `test_ingest_markdown_missing_file` | Error on missing file |
| `test_ingest_markdown_dry_run` | Dry run mode works |
| `test_ingest_pdf_not_implemented` | PDF shows Colab message |

#### TestSearchCommands (4 tests)

| Test | Description |
|------|-------------|
| `test_search_help` | search --help works |
| `test_search_semantic_empty_store` | Empty store returns empty |
| `test_search_keyword_empty_store` | Empty store returns empty |

#### TestDocsCommands (3 tests)

| Test | Description |
|------|-------------|
| `test_docs_help` | docs --help works |
| `test_docs_list_empty` | Empty list on empty store |
| `test_docs_stats_nonexistent` | Error on missing document |

#### TestDbCommands (3 tests)

| Test | Description |
|------|-------------|
| `test_db_help` | db --help works |
| `test_db_status` | db status works |
| `test_db_init` | db init works |

#### TestConfigCommands (4 tests)

| Test | Description |
|------|-------------|
| `test_config_help` | config --help works |
| `test_config_show` | config show works |
| `test_config_show_json` | config show --json works |
| `test_config_set_invalid_key` | Error on invalid key |

#### TestJsonOutput (3 tests)

| Test | Description |
|------|-------------|
| `test_docs_list_json` | docs list --json format |
| `test_db_status_json` | db status --json format |
| `test_config_show_json` | config show --json format |

---

### 24. test_cli_integration.py - End-to-End CLI Workflows

**Location:** `tests/test_cli_integration.py`

**Run:**
```bash
python -m pytest tests/test_cli_integration.py -v
```

**Test Classes:**

#### TestIngestWorkflow (3 tests)

| Test | Description |
|------|-------------|
| `test_ingest_markdown_creates_chunks` | Ingest creates chunks |
| `test_ingest_with_custom_sizes` | Custom chunk sizes work |
| `test_ingest_extracts_title_from_h1` | Title extraction works |

#### TestSearchWorkflow (2 tests)

| Test | Description |
|------|-------------|
| `test_semantic_search_returns_results` | Semantic search after ingest |
| `test_keyword_search_finds_content` | Keyword search finds content |

#### TestDocumentManagement (2 tests)

| Test | Description |
|------|-------------|
| `test_list_shows_ingested_document` | List shows ingested doc |
| `test_stats_shows_chunk_counts` | Stats shows chunk info |

#### TestBatchIngestion (2 tests)

| Test | Description |
|------|-------------|
| `test_batch_processes_directory` | Batch processes directory |
| `test_batch_skips_non_matching` | Skips non-matching files |

#### TestConfigWorkflow (2 tests)

| Test | Description |
|------|-------------|
| `test_show_displays_current_config` | Show displays settings |
| `test_config_init_creates_file` | Init creates config file |

#### TestDatabaseWorkflow (2 tests)

| Test | Description |
|------|-------------|
| `test_init_and_status` | Init followed by status |
| `test_clear_with_force` | Clear with force flag |

#### TestErrorHandling (3 tests)

| Test | Description |
|------|-------------|
| `test_invalid_chunk_sizes` | Error on invalid chunk sizes |
| `test_missing_document_stats` | Error on missing document |
| `test_invalid_config_key` | Error on invalid config key |

---

## Test Summary by Phase

| Phase | Test Files | Tests | Status |
|-------|-----------|-------|--------|
| 1. PDF Converter | `test_selector.py`, `test_analyzer.py`, `test_converter.py` | 51 | ✅ Done |
| 2. Hierarchical Chunking | `test_parser.py`, `test_chunker.py`, `test_metadata.py`, `test_embedder.py` | 172 | ✅ Done |
| 3. Helix-DB Integration | `test_storage_schema.py`, `test_storage_converters.py`, `test_storage_mock.py`, `test_storage_queries.py`, `test_storage_integration.py` | 114 | ✅ Done |
| 4. MCP Server | `test_mcp_config.py`, `test_mcp_formatters.py`, `test_mcp_tools.py`, `test_mcp_resources.py`, `test_mcp_server.py`, `test_mcp_integration.py` | 96 | ✅ Done |
| 5. CLI Tool | `test_cli_config.py`, `test_cli_utils.py`, `test_cli_output.py`, `test_cli_progress.py`, `test_cli_commands.py`, `test_cli_integration.py` | 126 | ✅ Done |

**Total: 557 tests passing, 10 skipped**
