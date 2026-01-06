# Quality Checker Design Document

## Overview

The Quality Checker validates parsed PDF output before ingestion into the knowledge base. It provides comprehensive quality metrics for OCR confidence, content completeness, and structure preservation, with configurable thresholds for different use cases.

## Goals

1. **Automatic validation** - Quality checks run automatically after PDF parsing
2. **Manual verification** - CLI command for checking existing markdown files
3. **Configurable thresholds** - Support for strict, normal, and lenient presets
4. **Block low-quality output** - Prevent ingestion of poor quality content
5. **Actionable feedback** - Clear recommendations for quality issues

## Quality Metrics

### 1. Content Metrics

| Metric | Description | Calculation |
|--------|-------------|-------------|
| `word_count` | Total words in document | Split on whitespace |
| `line_count` | Total lines | Split on newline |
| `estimated_pages` | Estimated page count | word_count / 300 |
| `words_per_page` | Content density | word_count / estimated_pages |
| `heading_count` | Markdown headings | Regex: `^#{1,6}\s+` |
| `table_count` | Markdown tables | Table separator lines |
| `code_block_count` | Fenced code blocks | ``` or ~~~ patterns |
| `list_count` | List items | Bullet or numbered list patterns |
| `special_char_ratio` | OCR artifact indicator | Special chars / total chars |
| `empty_line_ratio` | Content completeness | Empty lines / total lines |

### 2. OCR Metrics (when available)

| Metric | Description | Calculation |
|--------|-------------|-------------|
| `ocr_confidence_avg` | Average word confidence | Sum(confidence) / word_count |
| `low_confidence_word_count` | Words with confidence < 0.7 | Count of low confidence |
| `low_confidence_word_ratio` | Ratio of low confidence words | low_count / total_words |

### 3. Structure Metrics

| Metric | Description | Detection |
|--------|-------------|-----------|
| `broken_table_count` | Tables with inconsistent columns | Column count changes mid-table |

### 4. Chunk Metrics (when using chunked output)

| Metric | Description | Calculation |
|--------|-------------|-------------|
| `chunk_count` | Total chunks | Length of chunks list |
| `empty_chunk_count` | Chunks with no content | Chunks where content.strip() is empty |
| `small_chunk_count` | Chunks below minimum words | Chunks < min_chunk_words |
| `avg_chunk_size` | Average words per chunk | Sum(chunk_words) / chunk_count |

## Quality Thresholds

### Default Thresholds (normal)

```python
@dataclass
class QualityThresholds:
    # OCR confidence
    ocr_confidence_warning: float = 0.8
    ocr_confidence_error: float = 0.6
    low_confidence_word_ratio_warning: float = 0.10
    low_confidence_word_ratio_error: float = 0.25

    # Content
    words_per_page_warning: int = 100
    words_per_page_error: int = 50
    special_char_ratio_warning: float = 0.10
    special_char_ratio_error: float = 0.20
    empty_line_ratio_warning: float = 0.30
    empty_line_ratio_error: float = 0.45

    # Structure
    broken_table_warning: int = 1
    broken_table_error: int = 3

    # Chunks
    empty_chunk_ratio_warning: float = 0.05
    empty_chunk_ratio_error: float = 0.15
    min_chunk_words: int = 10
```

### Preset Configurations

| Preset | Use Case | Key Differences |
|--------|----------|-----------------|
| **strict** | High-quality documents | OCR > 0.9, words/page > 150, special chars < 5% |
| **normal** | Standard documents | Default values above |
| **lenient** | Poor quality sources | OCR > 0.5, words/page > 25, special chars < 30% |

## Architecture

### Class Diagram

```
QualityThresholds (dataclass)
├── Configurable threshold values
├── strict() → QualityThresholds
├── normal() → QualityThresholds (default)
└── lenient() → QualityThresholds

QualityMetrics (dataclass)
├── Content metrics (word_count, line_count, etc.)
├── OCR metrics (confidence, low_confidence_ratio)
├── Structure metrics (broken_table_count)
└── Chunk metrics (chunk_count, empty_chunks, etc.)

QualityIssue (dataclass)
├── code: str (e.g., "LOW_OCR_CONFIDENCE")
├── message: str
├── severity: IssueSeverity (INFO, WARNING, ERROR)
├── metric_name: str
├── actual_value: Any
└── threshold_value: Any

QualityReport (dataclass)
├── status: QualityStatus (PASS, WARNING, FAIL)
├── score: int (0-100)
├── metrics: QualityMetrics
├── issues: list[QualityIssue]
├── recommendation: str
├── passed: bool (property)
├── has_errors: bool (property)
├── to_dict() → dict
└── to_json() → str

ReductoQualityChecker
├── thresholds: QualityThresholds
├── check_markdown(markdown, expected_pages, ocr_data) → QualityReport
├── check_chunks(chunks) → QualityReport
├── check_file(file_path) → QualityReport
├── _collect_metrics() → QualityMetrics
├── _detect_issues() → list[QualityIssue]
├── _calculate_score() → int
├── _determine_status() → QualityStatus
└── _generate_recommendation() → str
```

### Data Flow

```
PDF → Reducto API → ParseResponse
                         ↓
              ReductoQualityChecker.check_markdown()
                         ↓
                  _collect_metrics()
                         ↓
                  _detect_issues()
                         ↓
                  _calculate_score()
                         ↓
                  _determine_status()
                         ↓
                  QualityReport
                    ├─ PASS → Save markdown
                    ├─ WARNING → Save with notice
                    └─ FAIL → Block + show issues
```

## CLI Interface

### Quality Command

```bash
# Check single file
kidkazz inbox quality ~/.kidkazz/output/document.md

# Check all files in output directory
kidkazz inbox quality --all

# Check specific directory
kidkazz inbox quality --dir /path/to/markdown

# JSON output for scripting
kidkazz inbox quality document.md --json

# Verbose metrics breakdown
kidkazz inbox quality document.md --verbose

# Summary of all files
kidkazz inbox quality --all --summary
```

### Parse Command Integration

```bash
# Default: quality check enabled
kidkazz inbox parse

# Disable quality check
kidkazz inbox parse --no-quality-check

# Strict threshold
kidkazz inbox parse --quality-threshold strict

# Lenient threshold
kidkazz inbox parse --quality-threshold lenient
```

## Output Formats

### Human-Readable Output

```
Quality Report: textbook.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Content Stats:
  Words: 4,523 | Pages (est): 15 | Headings: 12
  Tables: 3 | Code blocks: 2 | Lists: 8

Quality Metrics:
  ✓ Content density: 301 words/page (good)
  ✓ Structure: All tables intact
  ✓ Special chars: 2.1% (normal)
  ⚠ Heading hierarchy: Minor gaps detected

Overall Score: 87/100 (PASS)
Recommendation: Safe to ingest
```

### JSON Output

```json
{
  "status": "PASS",
  "score": 87,
  "metrics": {
    "word_count": 4523,
    "line_count": 312,
    "estimated_pages": 15,
    "words_per_page": 301.5,
    "heading_count": 12,
    "table_count": 3,
    "code_block_count": 2,
    "list_count": 8,
    "special_char_ratio": 0.021,
    "empty_line_ratio": 0.15,
    "broken_table_count": 0
  },
  "issues": [
    {
      "code": "HEADING_GAPS",
      "message": "Minor heading hierarchy gaps detected",
      "severity": "warning",
      "metric_name": "heading_hierarchy",
      "actual_value": 2,
      "threshold_value": 1
    }
  ],
  "recommendation": "Safe to ingest with minor warnings"
}
```

## Score Calculation

The quality score (0-100) is calculated using configurable thresholds:

| Factor | Max Deduction | Trigger |
|--------|---------------|---------|
| Words per page | -30 | Below error threshold |
| Special char ratio | -25 | Above error threshold |
| OCR confidence | -20 | Below error threshold |
| Broken tables | -15 | Above error threshold |
| Empty line ratio | -10 | Above error threshold |
| Empty chunk ratio | -10 | Above error threshold |

### Status Determination

| Status | Condition |
|--------|-----------|
| **FAIL** | Any ERROR issue OR score < 50 |
| **WARNING** | Any WARNING issue OR score < 70 |
| **PASS** | No issues AND score >= 70 |

## Files

| File | Purpose |
|------|---------|
| `src/pdf_converter/quality_checker.py` | Core quality checking logic |
| `src/pdf_converter/__init__.py` | Export quality checker classes |
| `src/cli/commands/inbox.py` | CLI quality command and parse integration |
| `tests/test_quality_checker.py` | Unit tests for quality checker |
| `tests/test_quality_cli.py` | CLI integration tests |

## Issue Codes

| Code | Severity | Description |
|------|----------|-------------|
| `LOW_OCR_CONFIDENCE` | WARNING/ERROR | OCR confidence below threshold |
| `HIGH_LOW_CONFIDENCE_WORDS` | WARNING/ERROR | Too many low-confidence words |
| `LOW_WORD_COUNT` | WARNING/ERROR | Words per page below threshold |
| `HIGH_SPECIAL_CHARS` | WARNING/ERROR | Special character ratio above threshold |
| `HIGH_EMPTY_LINES` | WARNING/ERROR | Empty line ratio above threshold |
| `BROKEN_TABLES` | WARNING/ERROR | Tables with inconsistent columns |
| `HIGH_EMPTY_CHUNKS` | WARNING/ERROR | Empty chunk ratio above threshold |
| `HIGH_SMALL_CHUNKS` | WARNING/ERROR | Too many small chunks |

## Testing

### Test Coverage

- **Unit tests**: 22 tests in `test_quality_checker.py`
  - QualityMetrics creation and serialization
  - QualityIssue creation
  - QualityReport status and serialization
  - QualityThresholds presets
  - ReductoQualityChecker analysis
  - Score calculation
  - Integration workflow

- **CLI tests**: 14 tests in `test_quality_cli.py`
  - Single file quality check
  - JSON output
  - Verbose output
  - File not found handling
  - Directory scanning
  - Summary output
  - Exit codes (pass/fail)
  - Parse command integration

### Running Tests

```bash
# Run quality checker tests
pytest tests/test_quality_checker.py -v

# Run CLI tests
pytest tests/test_quality_cli.py -v

# Run all quality-related tests
pytest tests/test_quality*.py -v
```

## Future Enhancements

1. **Heading hierarchy validation** - Detect skipped heading levels (h1 → h3)
2. **Language detection** - Flag unexpected language content
3. **Image reference validation** - Check for broken image links
4. **Semantic coherence** - Use embeddings to detect gibberish
5. **Custom threshold profiles** - User-defined threshold configurations
