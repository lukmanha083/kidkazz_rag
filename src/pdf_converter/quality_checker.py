"""Quality checker for Reducto.ai PDF parsing output.

Provides comprehensive quality metrics and validation for parsed markdown content,
including OCR confidence, content completeness, and structure preservation checks.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class IssueSeverity(Enum):
    """Severity level for quality issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QualityStatus(Enum):
    """Overall quality status."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class QualityMetrics:
    """Metrics collected during quality analysis."""

    # Content metrics
    word_count: int
    estimated_pages: int
    heading_count: int
    table_count: int
    code_block_count: int
    list_count: int
    special_char_ratio: float
    empty_line_ratio: float
    words_per_page: float

    # OCR metrics (optional - only when OCR data available)
    ocr_confidence_avg: Optional[float] = None
    low_confidence_word_count: Optional[int] = None
    low_confidence_word_ratio: Optional[float] = None

    # Structure metrics
    broken_table_count: int = 0

    # Chunk metrics (optional - only for chunked output)
    chunk_count: Optional[int] = None
    empty_chunk_count: Optional[int] = None
    avg_chunk_size: Optional[float] = None


@dataclass
class QualityIssue:
    """A single quality issue detected during analysis."""

    code: str
    message: str
    severity: IssueSeverity
    metric_name: str
    actual_value: Any
    threshold_value: Any


@dataclass
class QualityReport:
    """Complete quality analysis report."""

    status: QualityStatus
    score: int
    metrics: QualityMetrics
    issues: list[QualityIssue]
    recommendation: str

    @property
    def passed(self) -> bool:
        """Check if quality passed (PASS or WARNING status)."""
        return self.status in (QualityStatus.PASS, QualityStatus.WARNING)

    @property
    def has_errors(self) -> bool:
        """Check if any error-level issues exist."""
        return any(issue.severity == IssueSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if any warning-level issues exist."""
        return any(issue.severity == IssueSeverity.WARNING for issue in self.issues)

    def to_json(self) -> str:
        """Convert report to JSON string."""
        data = {
            "status": self.status.value,
            "score": self.score,
            "metrics": asdict(self.metrics),
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity.value,
                    "metric_name": issue.metric_name,
                    "actual_value": issue.actual_value,
                    "threshold_value": issue.threshold_value,
                }
                for issue in self.issues
            ],
            "recommendation": self.recommendation,
        }
        return json.dumps(data, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return json.loads(self.to_json())


@dataclass
class QualityThresholds:
    """Configurable thresholds for quality checks."""

    # OCR confidence thresholds
    ocr_confidence_warning: float = 0.8
    ocr_confidence_error: float = 0.6
    low_confidence_word_ratio_warning: float = 0.10
    low_confidence_word_ratio_error: float = 0.25

    # Content thresholds
    words_per_page_warning: int = 100
    words_per_page_error: int = 50
    special_char_ratio_warning: float = 0.10
    special_char_ratio_error: float = 0.20
    # Markdown naturally has many blank lines for formatting
    empty_line_ratio_warning: float = 0.30
    empty_line_ratio_error: float = 0.45

    # Structure thresholds
    broken_table_warning: int = 1
    broken_table_error: int = 3

    # Chunk thresholds
    empty_chunk_ratio_warning: float = 0.05
    empty_chunk_ratio_error: float = 0.15
    min_chunk_words: int = 10

    @classmethod
    def strict(cls) -> "QualityThresholds":
        """Create strict thresholds for high-quality requirements."""
        return cls(
            ocr_confidence_warning=0.9,
            ocr_confidence_error=0.75,
            low_confidence_word_ratio_warning=0.05,
            low_confidence_word_ratio_error=0.15,
            words_per_page_warning=150,
            words_per_page_error=100,
            special_char_ratio_warning=0.05,
            special_char_ratio_error=0.10,
            empty_line_ratio_warning=0.20,
            empty_line_ratio_error=0.35,
            broken_table_warning=1,
            broken_table_error=2,
            empty_chunk_ratio_warning=0.02,
            empty_chunk_ratio_error=0.10,
        )

    @classmethod
    def lenient(cls) -> "QualityThresholds":
        """Create lenient thresholds for permissive quality checks."""
        return cls(
            ocr_confidence_warning=0.7,
            ocr_confidence_error=0.5,
            low_confidence_word_ratio_warning=0.15,
            low_confidence_word_ratio_error=0.35,
            words_per_page_warning=50,
            words_per_page_error=25,
            special_char_ratio_warning=0.15,
            special_char_ratio_error=0.30,
            empty_line_ratio_warning=0.40,
            empty_line_ratio_error=0.55,
            broken_table_warning=3,
            broken_table_error=5,
            empty_chunk_ratio_warning=0.10,
            empty_chunk_ratio_error=0.25,
        )


class ReductoQualityChecker:
    """Quality checker for Reducto.ai parsed output."""

    # Pattern for detecting special/garbage characters (OCR artifacts)
    SPECIAL_CHAR_PATTERN = re.compile(r"[@#$%^&*|\\~`<>{}[\]]+")

    # Pattern for markdown headings
    HEADING_PATTERN = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

    # Pattern for markdown tables
    TABLE_PATTERN = re.compile(r"^\|.+\|$", re.MULTILINE)
    TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[-:|]+\|$", re.MULTILINE)

    # Pattern for code blocks
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")

    # Pattern for lists
    LIST_PATTERN = re.compile(r"^[\s]*[-*+]\s+.+$|^[\s]*\d+\.\s+.+$", re.MULTILINE)

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """Initialize quality checker with optional custom thresholds."""
        self.thresholds = thresholds or QualityThresholds()

    def check_markdown(
        self,
        markdown: str,
        expected_pages: Optional[int] = None,
        ocr_data: Optional[dict[str, Any]] = None,
    ) -> QualityReport:
        """
        Check quality of markdown content.

        Args:
            markdown: The markdown content to analyze
            expected_pages: Expected number of pages (from Reducto response)
            ocr_data: Optional OCR data with confidence scores

        Returns:
            QualityReport with metrics, issues, and overall status
        """
        metrics = self._collect_metrics(markdown, expected_pages, ocr_data)
        issues = self._detect_issues(metrics)
        score = self._calculate_score(
            words_per_page=metrics.words_per_page,
            special_char_ratio=metrics.special_char_ratio,
            empty_line_ratio=metrics.empty_line_ratio,
            broken_table_count=metrics.broken_table_count,
            ocr_confidence_avg=metrics.ocr_confidence_avg,
            empty_chunk_ratio=(
                metrics.empty_chunk_count / metrics.chunk_count
                if metrics.chunk_count and metrics.chunk_count > 0
                else 0.0
            ),
        )

        status = self._determine_status(issues, score)
        recommendation = self._generate_recommendation(status, issues)

        return QualityReport(
            status=status,
            score=score,
            metrics=metrics,
            issues=issues,
            recommendation=recommendation,
        )

    def check_chunks(self, chunks: list[dict[str, Any]]) -> QualityReport:
        """
        Check quality of chunked output.

        Args:
            chunks: List of chunk dictionaries with 'content' and 'embed' fields

        Returns:
            QualityReport focused on chunk quality metrics
        """
        # Combine chunks for overall metrics
        all_content = "\n\n".join(
            chunk.get("content", "") for chunk in chunks if chunk.get("content")
        )

        # Collect base metrics
        metrics = self._collect_metrics(all_content, expected_pages=None, ocr_data=None)

        # Add chunk-specific metrics
        metrics.chunk_count = len(chunks)
        metrics.empty_chunk_count = sum(
            1
            for chunk in chunks
            if not chunk.get("content") or not chunk.get("content").strip()
        )

        chunk_sizes = [
            len(chunk.get("content", "").split())
            for chunk in chunks
            if chunk.get("content")
        ]
        metrics.avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0.0

        issues = self._detect_issues(metrics)
        score = self._calculate_score(
            words_per_page=metrics.words_per_page,
            special_char_ratio=metrics.special_char_ratio,
            empty_line_ratio=metrics.empty_line_ratio,
            broken_table_count=metrics.broken_table_count,
            ocr_confidence_avg=metrics.ocr_confidence_avg,
            empty_chunk_ratio=(
                metrics.empty_chunk_count / metrics.chunk_count
                if metrics.chunk_count > 0
                else 0.0
            ),
        )

        status = self._determine_status(issues, score)
        recommendation = self._generate_recommendation(status, issues)

        return QualityReport(
            status=status,
            score=score,
            metrics=metrics,
            issues=issues,
            recommendation=recommendation,
        )

    def check_file(self, file_path: Path) -> QualityReport:
        """
        Check quality of a markdown file.

        Args:
            file_path: Path to the markdown file

        Returns:
            QualityReport for the file contents
        """
        content = file_path.read_text(encoding="utf-8")
        return self.check_markdown(content)

    def _collect_metrics(
        self,
        markdown: str,
        expected_pages: Optional[int],
        ocr_data: Optional[dict[str, Any]],
    ) -> QualityMetrics:
        """Collect all quality metrics from content."""
        lines = markdown.split("\n")
        words = markdown.split()
        word_count = len(words)

        # Estimate pages (300 words per page typical)
        estimated_pages = expected_pages or max(1, word_count // 300)

        # Count structural elements
        heading_count = len(self.HEADING_PATTERN.findall(markdown))
        table_lines = self.TABLE_PATTERN.findall(markdown)
        table_separators = self.TABLE_SEPARATOR_PATTERN.findall(markdown)
        table_count = len(table_separators)  # Each table has one separator line
        code_block_count = len(self.CODE_BLOCK_PATTERN.findall(markdown))
        list_count = len(self.LIST_PATTERN.findall(markdown))

        # Calculate ratios
        total_chars = len(markdown)
        special_chars = len("".join(self.SPECIAL_CHAR_PATTERN.findall(markdown)))
        special_char_ratio = special_chars / total_chars if total_chars > 0 else 0.0

        empty_lines = sum(1 for line in lines if not line.strip())
        empty_line_ratio = empty_lines / len(lines) if lines else 0.0

        words_per_page = word_count / estimated_pages if estimated_pages > 0 else 0.0

        # Check for broken tables
        broken_table_count = self._count_broken_tables(markdown)

        # Process OCR data if available
        ocr_confidence_avg = None
        low_confidence_word_count = None
        low_confidence_word_ratio = None

        if ocr_data and "words" in ocr_data:
            confidences = [
                word.get("confidence", 1.0) for word in ocr_data["words"]
            ]
            if confidences:
                ocr_confidence_avg = sum(confidences) / len(confidences)
                low_confidence_word_count = sum(
                    1 for c in confidences if c < 0.7
                )
                low_confidence_word_ratio = (
                    low_confidence_word_count / len(confidences)
                )

        return QualityMetrics(
            word_count=word_count,
            estimated_pages=estimated_pages,
            heading_count=heading_count,
            table_count=table_count,
            code_block_count=code_block_count,
            list_count=list_count,
            special_char_ratio=special_char_ratio,
            empty_line_ratio=empty_line_ratio,
            words_per_page=words_per_page,
            ocr_confidence_avg=ocr_confidence_avg,
            low_confidence_word_count=low_confidence_word_count,
            low_confidence_word_ratio=low_confidence_word_ratio,
            broken_table_count=broken_table_count,
        )

    def _count_broken_tables(self, markdown: str) -> int:
        """Count tables with inconsistent column counts."""
        broken_count = 0
        lines = markdown.split("\n")

        in_table = False
        expected_cols = 0
        table_lines_buffer = []

        for line in lines:
            if line.strip().startswith("|") and line.strip().endswith("|"):
                if not in_table:
                    in_table = True
                    expected_cols = line.count("|") - 1
                    table_lines_buffer = [line]
                else:
                    table_lines_buffer.append(line)
                    # Check column consistency
                    current_cols = line.count("|") - 1
                    if current_cols != expected_cols:
                        broken_count += 1
                        in_table = False
                        table_lines_buffer = []
            else:
                in_table = False
                table_lines_buffer = []

        return broken_count

    def _detect_issues(self, metrics: QualityMetrics) -> list[QualityIssue]:
        """Detect quality issues based on metrics and thresholds."""
        issues = []
        t = self.thresholds

        # Check OCR confidence
        if metrics.ocr_confidence_avg is not None:
            if metrics.ocr_confidence_avg < t.ocr_confidence_error:
                issues.append(
                    QualityIssue(
                        code="LOW_OCR_CONFIDENCE",
                        message=f"OCR confidence ({metrics.ocr_confidence_avg:.2f}) below threshold",
                        severity=IssueSeverity.ERROR,
                        metric_name="ocr_confidence_avg",
                        actual_value=metrics.ocr_confidence_avg,
                        threshold_value=t.ocr_confidence_error,
                    )
                )
            elif metrics.ocr_confidence_avg < t.ocr_confidence_warning:
                issues.append(
                    QualityIssue(
                        code="LOW_OCR_CONFIDENCE",
                        message=f"OCR confidence ({metrics.ocr_confidence_avg:.2f}) is below recommended",
                        severity=IssueSeverity.WARNING,
                        metric_name="ocr_confidence_avg",
                        actual_value=metrics.ocr_confidence_avg,
                        threshold_value=t.ocr_confidence_warning,
                    )
                )

        # Check words per page
        if metrics.words_per_page < t.words_per_page_error:
            issues.append(
                QualityIssue(
                    code="LOW_WORD_COUNT",
                    message=f"Word count per page ({metrics.words_per_page:.0f}) is too low",
                    severity=IssueSeverity.ERROR,
                    metric_name="words_per_page",
                    actual_value=metrics.words_per_page,
                    threshold_value=t.words_per_page_error,
                )
            )
        elif metrics.words_per_page < t.words_per_page_warning:
            issues.append(
                QualityIssue(
                    code="LOW_WORD_COUNT",
                    message=f"Word count per page ({metrics.words_per_page:.0f}) is below recommended",
                    severity=IssueSeverity.WARNING,
                    metric_name="words_per_page",
                    actual_value=metrics.words_per_page,
                    threshold_value=t.words_per_page_warning,
                )
            )

        # Check special character ratio
        if metrics.special_char_ratio > t.special_char_ratio_error:
            issues.append(
                QualityIssue(
                    code="HIGH_SPECIAL_CHARS",
                    message=f"Special character ratio ({metrics.special_char_ratio:.1%}) exceeds limit",
                    severity=IssueSeverity.ERROR,
                    metric_name="special_char_ratio",
                    actual_value=metrics.special_char_ratio,
                    threshold_value=t.special_char_ratio_error,
                )
            )
        elif metrics.special_char_ratio > t.special_char_ratio_warning:
            issues.append(
                QualityIssue(
                    code="HIGH_SPECIAL_CHARS",
                    message=f"Special character ratio ({metrics.special_char_ratio:.1%}) is elevated",
                    severity=IssueSeverity.WARNING,
                    metric_name="special_char_ratio",
                    actual_value=metrics.special_char_ratio,
                    threshold_value=t.special_char_ratio_warning,
                )
            )

        # Check empty line ratio
        if metrics.empty_line_ratio > t.empty_line_ratio_error:
            issues.append(
                QualityIssue(
                    code="HIGH_EMPTY_LINES",
                    message=f"Empty line ratio ({metrics.empty_line_ratio:.1%}) exceeds limit",
                    severity=IssueSeverity.ERROR,
                    metric_name="empty_line_ratio",
                    actual_value=metrics.empty_line_ratio,
                    threshold_value=t.empty_line_ratio_error,
                )
            )

        # Check broken tables
        if metrics.broken_table_count >= t.broken_table_error:
            issues.append(
                QualityIssue(
                    code="BROKEN_TABLES",
                    message=f"{metrics.broken_table_count} broken tables detected",
                    severity=IssueSeverity.ERROR,
                    metric_name="broken_table_count",
                    actual_value=metrics.broken_table_count,
                    threshold_value=t.broken_table_error,
                )
            )
        elif metrics.broken_table_count >= t.broken_table_warning:
            issues.append(
                QualityIssue(
                    code="BROKEN_TABLES",
                    message=f"{metrics.broken_table_count} broken table(s) detected",
                    severity=IssueSeverity.WARNING,
                    metric_name="broken_table_count",
                    actual_value=metrics.broken_table_count,
                    threshold_value=t.broken_table_warning,
                )
            )

        # Check chunk metrics if available
        if metrics.chunk_count is not None and metrics.chunk_count > 0:
            empty_ratio = metrics.empty_chunk_count / metrics.chunk_count
            if empty_ratio > t.empty_chunk_ratio_error:
                issues.append(
                    QualityIssue(
                        code="HIGH_EMPTY_CHUNKS",
                        message=f"Empty chunk ratio ({empty_ratio:.1%}) exceeds limit",
                        severity=IssueSeverity.ERROR,
                        metric_name="empty_chunk_ratio",
                        actual_value=empty_ratio,
                        threshold_value=t.empty_chunk_ratio_error,
                    )
                )
            elif empty_ratio > t.empty_chunk_ratio_warning:
                issues.append(
                    QualityIssue(
                        code="HIGH_EMPTY_CHUNKS",
                        message=f"Empty chunk ratio ({empty_ratio:.1%}) is elevated",
                        severity=IssueSeverity.WARNING,
                        metric_name="empty_chunk_ratio",
                        actual_value=empty_ratio,
                        threshold_value=t.empty_chunk_ratio_warning,
                    )
                )

        return issues

    def _calculate_score(
        self,
        words_per_page: float,
        special_char_ratio: float,
        empty_line_ratio: float,
        broken_table_count: int,
        ocr_confidence_avg: Optional[float],
        empty_chunk_ratio: float,
    ) -> int:
        """Calculate overall quality score (0-100)."""
        score = 100.0

        # Words per page contribution (max -30 points)
        if words_per_page < 50:
            score -= 30
        elif words_per_page < 100:
            score -= 20
        elif words_per_page < 150:
            score -= 10

        # Special character ratio contribution (max -25 points)
        if special_char_ratio > 0.20:
            score -= 25
        elif special_char_ratio > 0.10:
            score -= 15
        elif special_char_ratio > 0.05:
            score -= 5

        # Empty line ratio contribution (max -10 points)
        if empty_line_ratio > 0.20:
            score -= 10
        elif empty_line_ratio > 0.10:
            score -= 5

        # Broken tables contribution (max -15 points)
        score -= min(broken_table_count * 5, 15)

        # OCR confidence contribution (max -20 points)
        if ocr_confidence_avg is not None:
            if ocr_confidence_avg < 0.6:
                score -= 20
            elif ocr_confidence_avg < 0.8:
                score -= 10
            elif ocr_confidence_avg < 0.9:
                score -= 5

        # Empty chunk ratio contribution (max -10 points)
        if empty_chunk_ratio > 0.15:
            score -= 10
        elif empty_chunk_ratio > 0.05:
            score -= 5

        return max(0, min(100, int(score)))

    def _determine_status(
        self, issues: list[QualityIssue], score: int
    ) -> QualityStatus:
        """Determine overall quality status."""
        has_errors = any(issue.severity == IssueSeverity.ERROR for issue in issues)
        has_warnings = any(issue.severity == IssueSeverity.WARNING for issue in issues)

        if has_errors or score < 50:
            return QualityStatus.FAIL
        elif has_warnings or score < 70:
            return QualityStatus.WARNING
        else:
            return QualityStatus.PASS

    def _generate_recommendation(
        self, status: QualityStatus, issues: list[QualityIssue]
    ) -> str:
        """Generate recommendation based on status and issues."""
        if status == QualityStatus.PASS:
            return "Safe to ingest"
        elif status == QualityStatus.WARNING:
            return "Safe to ingest with minor warnings"
        else:
            error_codes = [
                issue.code for issue in issues if issue.severity == IssueSeverity.ERROR
            ]
            return f"Do not ingest - quality issues detected: {', '.join(error_codes)}"
