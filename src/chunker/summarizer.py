"""Document summarization using Instructor + LLM.

This module provides LLM-powered hierarchical summarization of documents
at document, chapter (L1), and section (L2) levels.

Concept extraction is integrated into summarization - each summary LLM call
also extracts key concepts, eliminating the need for separate concept extraction passes.

Features:
- Adaptive strategy selection based on chunk count:
  - < 50 chunks: Sequential (simple, no overhead)
  - 50-500 chunks: Async with rate limiting (parallel, respects limits)
  - > 500 chunks: OpenAI Batch API (50% cheaper, higher limits)
- Checkpointing for resume-from-failure
- Exponential backoff on rate limit errors
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()

# Optional import for instructor
try:
    import instructor

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    instructor = None  # type: ignore
    INSTRUCTOR_AVAILABLE = False

# Optional import for openai (batch API)
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    OPENAI_AVAILABLE = False

# Optional import for table parsing
try:
    from src.chunker.table_parser import parse_markdown_table, ParsedTable

    TABLE_PARSING_AVAILABLE = True
except ImportError:
    parse_markdown_table = None  # type: ignore
    ParsedTable = None  # type: ignore
    TABLE_PARSING_AVAILABLE = False

# Optional import for extractive summarization (section-level)
try:
    from src.chunker.extractive import (
        extractive_summarize,
        extract_keywords,
        SUMY_AVAILABLE,
        YAKE_AVAILABLE,
    )

    EXTRACTIVE_AVAILABLE = SUMY_AVAILABLE or YAKE_AVAILABLE
except ImportError:
    extractive_summarize = None  # type: ignore
    extract_keywords = None  # type: ignore
    SUMY_AVAILABLE = False
    YAKE_AVAILABLE = False
    EXTRACTIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Strategy Enum
# ============================================================================


class SummarizationStrategy(str, Enum):
    """Strategy for processing summarization requests."""

    SEQUENTIAL = "sequential"  # < 50 chunks: simple loop
    ASYNC = "async"  # 50-500 chunks: async with rate limiting
    BATCH = "batch"  # > 500 chunks: OpenAI Batch API


# Thresholds for strategy selection
ASYNC_THRESHOLD = 50  # Use async above this
BATCH_THRESHOLD = 500  # Use batch API above this


# ============================================================================
# Concept Types and Models
# ============================================================================


class ConceptType(str, Enum):
    """Types of concepts that can be extracted from multi-discipline textbooks."""

    # General
    TERM = "term"  # Vocabulary/definitions
    METHOD = "method"  # Techniques/procedures
    PRINCIPLE = "principle"  # Rules/laws/standards
    FORMULA = "formula"  # Calculations/equations

    # Accounting/Business
    ACCOUNT = "account"  # Ledger accounts: Inventory, Cost of Sales

    # STEM/Science
    THEORY = "theory"  # Scientific theories: Evolution, Relativity
    LAW = "law"  # Scientific laws: Newton's Laws, Thermodynamics
    PROCESS = "process"  # Biological/chemical processes: Photosynthesis, Digestion
    ORGANISM = "organism"  # Species/organisms: E. coli, Homo sapiens
    STRUCTURE = "structure"  # Anatomical/chemical structures: Cell membrane, DNA
    SYSTEM = "system"  # Body systems, ecosystems: Circulatory, Nitrogen cycle
    UNIT = "unit"  # Measurement units: Joules, Hectares, PPM

    # Agriculture/Farming/Ranching
    PRACTICE = "practice"  # Farming/ranching practices: Crop rotation, Grazing
    CROP = "crop"  # Crops/plants: Wheat, Corn, Alfalfa
    BREED = "breed"  # Animal breeds: Angus, Holstein, Berkshire
    PEST = "pest"  # Pests/diseases: Aphids, Blight, Mastitis
    TOOL = "tool"  # Equipment/tools: Tractor, Plow, Irrigation
    NUTRIENT = "nutrient"  # Nutrients/fertilizers: Nitrogen, Phosphorus, NPK

    # Computer Science/AI/ML
    ALGORITHM = "algorithm"  # Sorting, Search, Backpropagation, Gradient Descent
    DATA_STRUCTURE = "data_structure"  # Arrays, Trees, Graphs, Hash Tables
    MODEL = "model"  # ML models: CNN, RNN, Transformer, GPT
    ARCHITECTURE = "architecture"  # System/network architectures: MVC, REST, ResNet
    PROTOCOL = "protocol"  # Communication protocols: HTTP, TCP/IP, WebSocket
    LANGUAGE = "language"  # Programming languages/syntax: Python, SQL, Regex
    PARAMETER = "parameter"  # Hyperparameters: Learning rate, Batch size, Epochs
    METRIC = "metric"  # Evaluation metrics: Accuracy, Precision, F1, Loss
    FUNCTION = "function"  # Activation functions, Loss functions: ReLU, Softmax, CrossEntropy
    PATTERN = "pattern"  # Design patterns: Singleton, Factory, Observer
    OPERATOR = "operator"  # Mathematical/logical operators: Convolution, Pooling
    FRAMEWORK = "framework"  # Software frameworks: TensorFlow, PyTorch, Django
    DATASET = "dataset"  # Standard datasets: MNIST, ImageNet, COCO

    # Veterinary/Livestock
    DISEASE = "disease"  # Animal diseases: Foot and Mouth, Avian Flu, Bloat
    SYMPTOM = "symptom"  # Clinical signs: Fever, Lameness, Diarrhea, Lethargy
    TREATMENT = "treatment"  # Medical treatments: Antibiotics, Surgery, Fluid therapy
    VACCINE = "vaccine"  # Vaccines: Blackleg, Brucellosis, Clostridial
    DRUG = "drug"  # Veterinary drugs: Ivermectin, Penicillin, Oxytocin
    PATHOGEN = "pathogen"  # Bacteria, viruses, parasites: E. coli, BVD, Coccidia
    CONDITION = "condition"  # Health conditions: Ketosis, Acidosis, Milk fever
    ANATOMY = "anatomy"  # Animal body parts: Rumen, Udder, Hoof, Abomasum
    PROCEDURE = "procedure"  # Vet procedures: Castration, Dehorning, AI, Hoof trimming
    FEED = "feed"  # Animal nutrition: Silage, TMR, Concentrate, Forage
    BEHAVIOR = "behavior"  # Animal behavior: Estrus signs, Rumination, Nesting


class ExtractedConcept(BaseModel):
    """A concept extracted during summarization."""

    name: str = Field(description="Concept name (e.g., 'FIFO', 'Safety Stock', 'Cost of Goods Sold')")
    concept_type: ConceptType = Field(
        description="Type of concept. General: term, method, principle, formula. "
        "Science: theory, law, process, organism, structure, system, unit. "
        "Agriculture: practice, crop, breed, pest, tool, nutrient. "
        "CS/AI: algorithm, data_structure, model, architecture, protocol, language, parameter, metric, function, pattern, framework. "
        "Veterinary: disease, symptom, treatment, vaccine, drug, pathogen, condition, anatomy, procedure, feed, behavior."
    )
    definition: str = Field(description="1 sentence definition explaining what this concept means")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names or abbreviations (e.g., ['COGS'] for 'Cost of Goods Sold')"
    )

    @field_validator("concept_type", mode="before")
    @classmethod
    def coerce_concept_type(cls, v):
        """Coerce unknown concept types to 'term' (most generic default)."""
        if isinstance(v, ConceptType):
            return v
        if isinstance(v, str):
            v_lower = v.lower().strip().replace("-", "_").replace(" ", "_")
            # Try exact match first
            for ct in ConceptType:
                if ct.value == v_lower:
                    return ct
            # Map common variations to concept types
            type_mapping = {
                # General
                "definition": ConceptType.TERM,
                "concept": ConceptType.TERM,
                "vocabulary": ConceptType.TERM,
                "word": ConceptType.TERM,
                "terminology": ConceptType.TERM,
                "technique": ConceptType.METHOD,
                "approach": ConceptType.METHOD,
                "strategy": ConceptType.METHOD,
                "rule": ConceptType.PRINCIPLE,
                "standard": ConceptType.PRINCIPLE,
                "guideline": ConceptType.PRINCIPLE,
                "calculation": ConceptType.FORMULA,
                "equation": ConceptType.FORMULA,
                "expression": ConceptType.FORMULA,
                # Accounting
                "ledger": ConceptType.ACCOUNT,
                "journal": ConceptType.ACCOUNT,
                # Science
                "hypothesis": ConceptType.THEORY,
                "scientific_law": ConceptType.LAW,
                "reaction": ConceptType.PROCESS,
                "cycle": ConceptType.PROCESS,
                "mechanism": ConceptType.PROCESS,
                "pathway": ConceptType.PROCESS,
                "species": ConceptType.ORGANISM,
                "animal": ConceptType.ORGANISM,
                "plant": ConceptType.ORGANISM,
                "bacteria": ConceptType.ORGANISM,
                "virus": ConceptType.PATHOGEN,
                "microbe": ConceptType.PATHOGEN,
                "parasite": ConceptType.PATHOGEN,
                "organ": ConceptType.STRUCTURE,
                "tissue": ConceptType.STRUCTURE,
                "cell": ConceptType.STRUCTURE,
                "molecule": ConceptType.STRUCTURE,
                "compound": ConceptType.STRUCTURE,
                "ecosystem": ConceptType.SYSTEM,
                "network": ConceptType.SYSTEM,
                "measurement": ConceptType.UNIT,
                # Agriculture
                "farming": ConceptType.PRACTICE,
                "cultivation": ConceptType.PRACTICE,
                "husbandry": ConceptType.PRACTICE,
                "plant": ConceptType.CROP,
                "grain": ConceptType.CROP,
                "vegetable": ConceptType.CROP,
                "livestock": ConceptType.BREED,
                "variety": ConceptType.BREED,
                "insect": ConceptType.PEST,
                "weed": ConceptType.PEST,
                "blight": ConceptType.PEST,
                "equipment": ConceptType.TOOL,
                "machine": ConceptType.TOOL,
                "implement": ConceptType.TOOL,
                "fertilizer": ConceptType.NUTRIENT,
                "mineral": ConceptType.NUTRIENT,
                "supplement": ConceptType.NUTRIENT,
                # Computer Science/AI
                "sorting": ConceptType.ALGORITHM,
                "search": ConceptType.ALGORITHM,
                "optimization": ConceptType.ALGORITHM,
                "heuristic": ConceptType.ALGORITHM,
                "array": ConceptType.DATA_STRUCTURE,
                "tree": ConceptType.DATA_STRUCTURE,
                "graph": ConceptType.DATA_STRUCTURE,
                "list": ConceptType.DATA_STRUCTURE,
                "neural_network": ConceptType.MODEL,
                "classifier": ConceptType.MODEL,
                "regressor": ConceptType.MODEL,
                "transformer": ConceptType.MODEL,
                "cnn": ConceptType.MODEL,
                "rnn": ConceptType.MODEL,
                "design": ConceptType.ARCHITECTURE,
                "topology": ConceptType.ARCHITECTURE,
                "standard": ConceptType.PROTOCOL,
                "specification": ConceptType.PROTOCOL,
                "syntax": ConceptType.LANGUAGE,
                "programming": ConceptType.LANGUAGE,
                "hyperparameter": ConceptType.PARAMETER,
                "config": ConceptType.PARAMETER,
                "setting": ConceptType.PARAMETER,
                "score": ConceptType.METRIC,
                "measure": ConceptType.METRIC,
                "evaluation": ConceptType.METRIC,
                "activation": ConceptType.FUNCTION,
                "loss": ConceptType.FUNCTION,
                "design_pattern": ConceptType.PATTERN,
                "operation": ConceptType.OPERATOR,
                "library": ConceptType.FRAMEWORK,
                "toolkit": ConceptType.FRAMEWORK,
                "platform": ConceptType.FRAMEWORK,
                "corpus": ConceptType.DATASET,
                "benchmark": ConceptType.DATASET,
                # Veterinary
                "illness": ConceptType.DISEASE,
                "infection": ConceptType.DISEASE,
                "disorder": ConceptType.DISEASE,
                "sign": ConceptType.SYMPTOM,
                "clinical_sign": ConceptType.SYMPTOM,
                "indicator": ConceptType.SYMPTOM,
                "therapy": ConceptType.TREATMENT,
                "medication": ConceptType.DRUG,
                "medicine": ConceptType.DRUG,
                "antibiotic": ConceptType.DRUG,
                "pharmaceutical": ConceptType.DRUG,
                "immunization": ConceptType.VACCINE,
                "vaccination": ConceptType.VACCINE,
                "infectious_agent": ConceptType.PATHOGEN,
                "syndrome": ConceptType.CONDITION,
                "state": ConceptType.CONDITION,
                "body_part": ConceptType.ANATOMY,
                "morphology": ConceptType.ANATOMY,
                "surgery": ConceptType.PROCEDURE,
                "operation": ConceptType.PROCEDURE,
                "intervention": ConceptType.PROCEDURE,
                "nutrition": ConceptType.FEED,
                "diet": ConceptType.FEED,
                "ration": ConceptType.FEED,
                "forage": ConceptType.FEED,
                "action": ConceptType.BEHAVIOR,
                "habit": ConceptType.BEHAVIOR,
                "instinct": ConceptType.BEHAVIOR,
            }
            if v_lower in type_mapping:
                return type_mapping[v_lower]
            # Default to TERM for unknown types
            logger.debug(f"Unknown concept type '{v}', defaulting to 'term'")
            return ConceptType.TERM
        return ConceptType.TERM

    @field_validator("aliases", mode="before")
    @classmethod
    def coerce_aliases(cls, v):
        """Ensure aliases is a list of strings."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            # Convert all items to strings, filter out None/empty
            return [str(item) for item in v if item is not None and str(item).strip()]
        return []


# ============================================================================
# Data Classes
# ============================================================================


def _normalize_plural(word: str) -> str:
    """Normalize a single word to singular form for concept deduplication."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        return w[:-1]
    return w


def slugify(name: str) -> str:
    """Convert concept name to URL-safe ID with plural normalization.

    Ensures singular/plural variants (e.g., "cost" vs "costs") produce
    the same concept ID so they merge during aggregation.
    """
    import re
    if not name:
        return ""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    # Normalize each word to singular form
    words = slug.split()
    words = [_normalize_plural(w) for w in words]
    slug = "-".join(words)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


@dataclass
class Concept:
    """A concept extracted during summarization.

    Concepts can appear in multiple summaries/chapters. The occurrences field
    tracks all places where this concept is mentioned.
    """

    concept_id: str  # Slugified name for unique ID
    name: str  # Display name
    concept_type: str  # See ConceptType enum for all valid types (multi-discipline support)
    definition: str  # 1 sentence definition
    aliases: list[str] = field(default_factory=list)
    document_id: str = ""  # Parent document
    occurrences: list[str] = field(default_factory=list)  # List of summary_ids where this concept appears
    occurrence_count: int = 0  # How many times this concept was mentioned


@dataclass
class Summary:
    """A summary of document content at various levels."""

    summary_id: str  # "summary_{source_id}_{level}"
    content: str  # Summary text
    level: str  # "document", "chapter", "section"
    source_id: str  # doc_id or chunk_id
    document_id: str  # For document filtering
    parent_summary_id: Optional[str] = None  # Hierarchy navigation
    key_points: list[str] = field(default_factory=list)  # Key takeaways
    concepts: list[Concept] = field(default_factory=list)  # Extracted concepts
    embedding: Optional[list[float]] = None  # Vector embedding
    word_count: int = 0
    created_at: int = 0

    def __post_init__(self):
        """Calculate word count and timestamp if not set."""
        if self.word_count == 0:
            self.word_count = len(self.content.split())
        if self.created_at == 0:
            self.created_at = int(time.time())


# ============================================================================
# Pydantic Models for LLM Output
# ============================================================================


class SectionSummaryOutput(BaseModel):
    """LLM output for section-level summary."""

    summary: str = Field(
        description="2-3 sentence summary of the section's main points"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="1-3 key takeaways from this section",
    )
    concepts: list[ExtractedConcept] = Field(
        default_factory=list,
        description="Key concepts (terms, methods, principles, formulas) defined or explained in this section. "
                    "Only include concepts that are important enough to be referenced elsewhere.",
    )


class ChapterSummaryOutput(BaseModel):
    """LLM output for chapter-level summary."""

    summary: str = Field(
        description="3-5 sentence summary of the chapter's main topics and themes"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="3-5 key concepts or takeaways from this chapter",
    )
    concepts: list[ExtractedConcept] = Field(
        default_factory=list,
        description="Important concepts covered in this chapter. Include terms, methods, principles, "
                    "formulas, or accounts that readers should understand. Focus on concepts that appear "
                    "multiple times or are central to the chapter's topic.",
    )


class DocumentSummaryOutput(BaseModel):
    """LLM output for document-level summary."""

    summary: str = Field(
        description="5-7 sentence summary of the entire document's scope and purpose"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="5-7 main topics or themes covered in this document",
    )
    concepts: list[ExtractedConcept] = Field(
        default_factory=list,
        description="The most important concepts in this document. These are the key terms, methods, "
                    "principles, and formulas that define the document's subject matter. "
                    "Focus on concepts that appear across multiple chapters.",
    )


# ============================================================================
# Checkpoint Manager
# ============================================================================


@dataclass
class Checkpoint:
    """Tracks progress for resume-from-failure."""

    document_id: str
    completed_sections: list[str] = field(default_factory=list)
    completed_chapters: list[str] = field(default_factory=list)
    section_summaries: dict = field(default_factory=dict)  # chunk_id -> summary dict
    chapter_summaries: list = field(default_factory=list)  # list of summary dicts
    batch_id: Optional[str] = None  # For batch API tracking

    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["Checkpoint"]:
        """Load checkpoint from file if exists."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    @classmethod
    def get_path(cls, document_id: str) -> Path:
        """Get checkpoint file path for a document."""
        return Path.home() / ".kidkazz" / "checkpoints" / f"{document_id}_summarize.json"


# ============================================================================
# DocumentSummarizer Class
# ============================================================================


class DocumentSummarizer:
    """Generate hierarchical summaries for documents using LLM.

    Automatically selects the best strategy based on chunk count:
    - < 50 chunks: Sequential processing
    - 50-500 chunks: Async with rate limiting
    - > 500 chunks: OpenAI Batch API (50% cheaper)
    """

    def __init__(
        self,
        provider: str = "openai/gpt-4o-mini",
        max_retries: int = 3,
        max_tokens_per_summary: int = 500,
        max_concurrent: int = 5,  # For async strategy
        requests_per_minute: int = 500,  # Rate limit for async
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize summarizer.

        Args:
            provider: Instructor provider string (e.g., "openai/gpt-4o-mini")
            max_retries: Number of retries on validation failure
            max_tokens_per_summary: Maximum tokens for each summary
            max_concurrent: Max concurrent requests for async strategy
            requests_per_minute: Rate limit for async strategy
            checkpoint_dir: Directory for checkpoint files

        Raises:
            ImportError: If instructor is not installed
        """
        if not INSTRUCTOR_AVAILABLE or instructor is None:
            raise ImportError(
                "instructor is not installed. "
                "Install with: pip install 'kidkazz[concepts]'"
            )

        self.provider = provider
        self.client = instructor.from_provider(provider)
        self.max_retries = max_retries
        self.max_tokens_per_summary = max_tokens_per_summary
        self.max_concurrent = max_concurrent
        self.requests_per_minute = requests_per_minute
        self.checkpoint_dir = checkpoint_dir or Path.home() / ".kidkazz" / "checkpoints"

        # For async rate limiting
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._rate_limit_delay = 60.0 / requests_per_minute

    def determine_strategy(self, l1_count: int) -> SummarizationStrategy:
        """Determine the best strategy based on L1 (chapter) chunk count.

        Since section-level summaries use local extractive methods (no API calls),
        strategy selection is based only on the number of chapter-level LLM calls.

        Args:
            l1_count: Number of L1 (chapter) chunks requiring LLM calls

        Returns:
            The recommended strategy
        """
        if l1_count < ASYNC_THRESHOLD:
            return SummarizationStrategy.SEQUENTIAL
        elif l1_count < BATCH_THRESHOLD:
            return SummarizationStrategy.ASYNC
        else:
            # Only use batch if OpenAI provider and openai package available
            if "openai" in self.provider.lower() and OPENAI_AVAILABLE:
                return SummarizationStrategy.BATCH
            else:
                return SummarizationStrategy.ASYNC

    # ========================================================================
    # System Prompts
    # ========================================================================

    def _section_system_prompt(self, has_table: bool = False) -> str:
        base_prompt = (
            "You are summarizing a section from a textbook or technical document. "
            "Create a concise summary that captures the main points and key concepts. "
            "Focus on what this section teaches or explains.\n\n"
        )

        if has_table:
            base_prompt += (
                "This section contains a TABLE. Pay special attention to:\n"
                "- Column headers (often contain key terms/concepts)\n"
                "- Relationships and comparisons shown in the table\n"
                "- Key values, patterns, or data points\n"
                "- What the table is comparing or organizing\n\n"
            )

        base_prompt += (
            "Extract important CONCEPTS from this section:\n"
            "- Terms: vocabulary and definitions (e.g., 'Safety Stock', 'COGS')\n"
            "- Methods: techniques and procedures (e.g., 'FIFO', 'ABC Analysis')\n"
            "- Principles: rules and guidelines (e.g., 'Matching Principle')\n"
            "- Formulas: calculations (e.g., 'EOQ Formula')\n"
            "- Accounts: ledger accounts (e.g., 'Inventory', 'Cost of Sales')\n\n"
            "Only include concepts that are important enough to be referenced elsewhere in the document."
        )

        return base_prompt

    def _chapter_system_prompt(self) -> str:
        return (
            "You are summarizing a chapter from a textbook or technical document. "
            "Given the summaries of individual sections, create an overall chapter summary "
            "that captures the main themes and learning objectives. "
            "Synthesize the key concepts into a coherent overview.\n\n"
            "Also extract the most important CONCEPTS from this chapter:\n"
            "- Terms: vocabulary and definitions\n"
            "- Methods: techniques and procedures\n"
            "- Principles: rules and guidelines\n"
            "- Formulas: calculations\n"
            "- Accounts: ledger accounts\n\n"
            "Focus on concepts that appear multiple times across sections or are central to the chapter."
        )

    def _document_system_prompt(self) -> str:
        return (
            "You are summarizing an entire textbook or technical document. "
            "Given the chapter summaries, create a comprehensive document overview "
            "that explains what the document covers, its purpose, and main themes. "
            "This should help readers understand if this document is relevant to their needs.\n\n"
            "Also extract the most important CONCEPTS from this document:\n"
            "- Terms: key vocabulary and definitions\n"
            "- Methods: important techniques and procedures\n"
            "- Principles: core rules and guidelines\n"
            "- Formulas: essential calculations\n"
            "- Accounts: key ledger accounts\n\n"
            "Focus on the concepts that are most central to the document's subject matter "
            "and appear across multiple chapters."
        )

    # ========================================================================
    # Helper: Format table for LLM context
    # ========================================================================

    def _format_table_context(self, chunk_content: str, chunk_id: str) -> Optional[str]:
        """Parse and format table data for better LLM understanding.

        Args:
            chunk_content: Raw chunk content that may contain a table
            chunk_id: Chunk identifier

        Returns:
            Formatted table context string, or None if no table found
        """
        if not TABLE_PARSING_AVAILABLE or parse_markdown_table is None:
            return None

        try:
            parsed = parse_markdown_table(chunk_content, chunk_id)
            if not parsed:
                return None

            # Build structured table context
            lines = [
                "\n--- STRUCTURED TABLE DATA ---",
                f"Columns: {', '.join(parsed.column_names)}",
                f"Column types: {', '.join(parsed.column_types)}",
                f"Rows: {parsed.row_count}",
                "",
                "Table in key-value format (easier to understand):",
                parsed.to_markdown_kv(),
                "--- END TABLE DATA ---\n",
            ]
            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"Table parsing failed for {chunk_id}: {e}")
            return None

    # ========================================================================
    # Helper: Convert LLM output to Summary
    # ========================================================================

    def _result_to_summary(
        self,
        result: BaseModel,
        summary_id: str,
        level: str,
        source_id: str,
        document_id: str,
    ) -> Summary:
        """Convert LLM output to Summary dataclass."""
        concepts = [
            Concept(
                concept_id=slugify(c.name),
                name=c.name,
                concept_type=c.concept_type.value if hasattr(c.concept_type, 'value') else str(c.concept_type),
                definition=c.definition,
                aliases=c.aliases,
                document_id=document_id,
                occurrences=[summary_id],
                occurrence_count=1,
            )
            for c in result.concepts
        ]

        return Summary(
            summary_id=summary_id,
            content=result.summary,
            level=level,
            source_id=source_id,
            document_id=document_id,
            key_points=result.key_points,
            concepts=concepts,
        )

    # ========================================================================
    # Sequential Processing (< 50 chunks)
    # ========================================================================

    def summarize_section(
        self,
        chunk_content: str,
        chunk_id: str,
        document_id: str,
        document_title: str,
        section_path: Optional[list[str]] = None,
        has_table: bool = False,
    ) -> Summary:
        """Generate summary for a section (L2 chunk) using extractive methods.

        Uses local TextRank summarization and YAKE keyword extraction instead
        of LLM API calls. Falls back to naive first-N-sentences if extractive
        libraries are unavailable.

        Args:
            chunk_content: The text content of the chunk
            chunk_id: Unique chunk identifier
            document_id: Parent document ID
            document_title: Document title for context
            section_path: Breadcrumb path for context
            has_table: Whether chunk contains a table

        Returns:
            Summary object for this section
        """
        summary_id = f"summary_{chunk_id}_section"

        try:
            # Extractive summarization (local, no API call)
            if extractive_summarize is not None:
                summary_text = extractive_summarize(chunk_content, sentence_count=3)
            else:
                # Minimal fallback: first 200 chars
                summary_text = chunk_content[:200].strip()
                if len(chunk_content) > 200:
                    summary_text += "..."

            if not summary_text:
                summary_text = f"Section content from {document_title}."

            # Keyword extraction (local, no API call)
            concepts = []
            if extract_keywords is not None:
                keywords = extract_keywords(chunk_content, max_keywords=5)
                for kw in keywords:
                    concepts.append(Concept(
                        concept_id=slugify(kw["name"]),
                        name=kw["name"],
                        concept_type="term",
                        definition="",
                        document_id=document_id,
                        occurrences=[summary_id],
                        occurrence_count=1,
                    ))

            # Build key points from first few keywords
            key_points = [kw["name"] for kw in (keywords if extract_keywords is not None else [])][:3]

            return Summary(
                summary_id=summary_id,
                content=summary_text,
                level="section",
                source_id=chunk_id,
                document_id=document_id,
                key_points=key_points,
                concepts=concepts,
            )

        except Exception as e:
            logger.warning(f"Section extractive summarization failed for {chunk_id}: {e}")
            return Summary(
                summary_id=summary_id,
                content=f"Section content from {document_title}.",
                level="section",
                source_id=chunk_id,
                document_id=document_id,
            )

    def summarize_chapter(
        self,
        chunk_id: str,
        chunk_content: str,
        section_summaries: list[Summary],
        document_id: str,
        document_title: str,
        chapter_title: Optional[str] = None,
    ) -> Summary:
        """Generate summary for a chapter (L1 chunk)."""
        section_context = "\n\n".join(
            f"Section {i+1}:\n{s.content}"
            for i, s in enumerate(section_summaries)
        )

        if not section_context:
            section_context = chunk_content[:2000]

        context = f"Document: {document_title}"
        if chapter_title:
            context += f"\nChapter: {chapter_title}"

        try:
            result = self.client.create(
                response_model=ChapterSummaryOutput,
                messages=[
                    {"role": "system", "content": self._chapter_system_prompt()},
                    {"role": "user", "content": f"{context}\n\nSection Summaries:\n{section_context}"},
                ],
                max_retries=self.max_retries,
                max_tokens=self.max_tokens_per_summary,
            )

            return self._result_to_summary(
                result,
                summary_id=f"summary_{chunk_id}_chapter",
                level="chapter",
                source_id=chunk_id,
                document_id=document_id,
            )

        except Exception as e:
            logger.warning(f"Chapter summarization failed for {chunk_id}: {e}")
            return Summary(
                summary_id=f"summary_{chunk_id}_chapter",
                content=f"Chapter content from {document_title}.",
                level="chapter",
                source_id=chunk_id,
                document_id=document_id,
            )

    def summarize_document(
        self,
        document_id: str,
        document_title: str,
        chapter_summaries: list[Summary],
    ) -> Summary:
        """Generate document-level summary."""
        chapter_context = "\n\n".join(
            f"Chapter {i+1}:\n{s.content}\nKey points: {', '.join(s.key_points)}"
            for i, s in enumerate(chapter_summaries)
        )

        if not chapter_context:
            chapter_context = "No chapter summaries available."

        try:
            # Document-level needs more tokens for summary + concepts across all chapters
            doc_max_tokens = max(self.max_tokens_per_summary, 2000)
            result = self.client.create(
                response_model=DocumentSummaryOutput,
                messages=[
                    {"role": "system", "content": self._document_system_prompt()},
                    {"role": "user", "content": f"Document: {document_title}\n\nChapter Summaries:\n{chapter_context}"},
                ],
                max_retries=self.max_retries,
                max_tokens=doc_max_tokens,
            )

            return self._result_to_summary(
                result,
                summary_id=f"summary_{document_id}_document",
                level="document",
                source_id=document_id,
                document_id=document_id,
            )

        except Exception as e:
            logger.warning(f"Document summarization failed for {document_id}: {e}")
            return Summary(
                summary_id=f"summary_{document_id}_document",
                content=f"Document: {document_title}",
                level="document",
                source_id=document_id,
                document_id=document_id,
            )

    def _generate_sequential(
        self,
        document_id: str,
        document_title: str,
        l1_chunks: list[dict],
        l2_chunks: list[dict],
        checkpoint: Optional[Checkpoint] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[list[Summary], dict[str, Summary]]:
        """Sequential processing for small documents."""
        all_summaries: list[Summary] = []
        section_summaries: dict[str, Summary] = {}

        # Total progress: sections (instant) + chapters (LLM) + 1 doc (LLM)
        total = len(l2_chunks) + len(l1_chunks) + 1
        current = 0

        # Phase 1: Section summaries (extractive, instant - no checkpoint needed)
        if progress_callback:
            progress_callback(0, total, f"Extracting {len(l2_chunks)} sections (local)")

        section_start = time.monotonic()
        logger.info("Extractive section phase: %d sections", len(l2_chunks))

        for i, chunk in enumerate(l2_chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")

            summary = self.summarize_section(
                chunk_content=chunk.get("content", ""),
                chunk_id=chunk_id,
                document_id=document_id,
                document_title=document_title,
                section_path=chunk.get("section_path"),
                has_table=chunk.get("has_table", False),
            )
            section_summaries[chunk_id] = summary
            all_summaries.append(summary)
            current += 1

            # Log progress every 100 sections
            if (i + 1) % 100 == 0:
                elapsed = time.monotonic() - section_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info("  Extractive: %d/%d sections (%.1f/s, %.1fs)",
                            i + 1, len(l2_chunks), rate, elapsed)

            # Yield CPU periodically — CPU affinity handles the main throttling
            if i % 20 == 19:
                time.sleep(0.05)

        elapsed = time.monotonic() - section_start
        logger.info("Extractive section phase done: %d sections in %.1fs", len(l2_chunks), elapsed)

        # Phase 2: Chapter summaries (LLM calls - checkpoint supported)
        chapter_summaries: list[Summary] = []
        for i, chunk in enumerate(l1_chunks):
            chunk_id = chunk.get("id", f"l1_chunk_{i}")

            if checkpoint and chunk_id in checkpoint.completed_chapters:
                current += 1
                continue

            if progress_callback:
                progress_callback(current, total, f"Chapter {i+1}/{len(l1_chunks)} (LLM)")

            child_ids = chunk.get("child_ids", [])
            child_summaries = [
                section_summaries[cid]
                for cid in child_ids
                if cid in section_summaries
            ]

            summary = self.summarize_chapter(
                chunk_id=chunk_id,
                chunk_content=chunk.get("content", ""),
                section_summaries=child_summaries,
                document_id=document_id,
                document_title=document_title,
                chapter_title=chunk.get("source_section"),
            )

            for child_summary in child_summaries:
                child_summary.parent_summary_id = summary.summary_id

            chapter_summaries.append(summary)
            all_summaries.append(summary)
            current += 1

        # Phase 3: Document summary (LLM call)
        if progress_callback:
            progress_callback(current, total, "Document summary (LLM)")

        doc_summary = self.summarize_document(
            document_id=document_id,
            document_title=document_title,
            chapter_summaries=chapter_summaries,
        )

        for chapter_summary in chapter_summaries:
            chapter_summary.parent_summary_id = doc_summary.summary_id

        all_summaries.append(doc_summary)

        return all_summaries, section_summaries

    # ========================================================================
    # Async Processing (50-500 chunks)
    # ========================================================================

    async def _summarize_chapter_async(
        self,
        chunk: dict,
        section_summaries: dict[str, "Summary"],
        document_id: str,
        document_title: str,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, "Summary"]:
        """Async chapter summarization with rate limiting."""
        async with semaphore:
            chunk_id = chunk.get("id", "unknown")

            child_ids = chunk.get("child_ids", [])
            child_summaries = [
                section_summaries[cid]
                for cid in child_ids
                if cid in section_summaries
            ]

            # Run sync LLM call in thread pool
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(
                None,
                lambda: self.summarize_chapter(
                    chunk_id=chunk_id,
                    chunk_content=chunk.get("content", ""),
                    section_summaries=child_summaries,
                    document_id=document_id,
                    document_title=document_title,
                    chapter_title=chunk.get("source_section"),
                )
            )

            # Set parent references
            for child_summary in child_summaries:
                child_summary.parent_summary_id = summary.summary_id

            # Rate limiting delay
            await asyncio.sleep(self._rate_limit_delay)

            return chunk_id, summary

    async def _generate_async(
        self,
        document_id: str,
        document_title: str,
        l1_chunks: list[dict],
        l2_chunks: list[dict],
        checkpoint: Optional[Checkpoint] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[list[Summary], dict[str, Summary]]:
        """Async processing with rate limiting for medium documents.

        Sections are processed locally (extractive, instant).
        Only chapter LLM calls use async with rate limiting.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        all_summaries: list[Summary] = []
        section_summaries: dict[str, Summary] = {}

        total = len(l2_chunks) + len(l1_chunks) + 1

        # Phase 1: Section summaries (extractive, instant - run synchronously)
        if progress_callback:
            progress_callback(0, total, f"Extracting {len(l2_chunks)} sections (local)")

        section_start = time.monotonic()
        logger.info("Async path - extractive section phase: %d sections", len(l2_chunks))

        for i, chunk in enumerate(l2_chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            summary = self.summarize_section(
                chunk_content=chunk.get("content", ""),
                chunk_id=chunk_id,
                document_id=document_id,
                document_title=document_title,
                section_path=chunk.get("section_path"),
                has_table=chunk.get("has_table", False),
            )
            section_summaries[chunk_id] = summary
            all_summaries.append(summary)

            # Log progress every 100 sections
            if (i + 1) % 100 == 0:
                elapsed = time.monotonic() - section_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info("  Async extractive: %d/%d sections (%.1f/s, %.1fs)",
                            i + 1, len(l2_chunks), rate, elapsed)

            # Yield CPU periodically — CPU affinity handles the main throttling
            if i % 20 == 19:
                time.sleep(0.05)

        elapsed = time.monotonic() - section_start
        logger.info("Async extractive done: %d sections in %.1fs", len(l2_chunks), elapsed)

        completed = len(l2_chunks)

        if progress_callback:
            progress_callback(completed, total, f"Processing {len(l1_chunks)} chapters async (LLM)")

        # Phase 2: Async chapter summaries (LLM calls)
        # Filter out already completed chapters from checkpoint
        pending_l1 = [
            c for c in l1_chunks
            if not checkpoint or c.get("id") not in checkpoint.completed_chapters
        ]

        tasks = [
            self._summarize_chapter_async(
                chunk, section_summaries, document_id, document_title, semaphore
            )
            for chunk in pending_l1
        ]

        chapter_summaries: list[Summary] = []
        for coro in asyncio.as_completed(tasks):
            chunk_id, summary = await coro
            chapter_summaries.append(summary)
            all_summaries.append(summary)
            completed += 1
            if progress_callback:
                progress_callback(completed, total, f"Chapters: {completed - len(l2_chunks)}/{len(pending_l1)} (LLM)")

        # Phase 3: Document summary (LLM call)
        if progress_callback:
            progress_callback(completed, total, "Document summary (LLM)")

        doc_summary = self.summarize_document(
            document_id=document_id,
            document_title=document_title,
            chapter_summaries=chapter_summaries,
        )

        for chapter_summary in chapter_summaries:
            chapter_summary.parent_summary_id = doc_summary.summary_id

        all_summaries.append(doc_summary)

        return all_summaries, section_summaries

    # ========================================================================
    # Batch API Processing (> 500 chunks)
    # ========================================================================

    def _create_batch_request(
        self,
        custom_id: str,
        system_prompt: str,
        user_content: str,
        response_format: dict,
    ) -> dict:
        """Create a single batch request in JSONL format."""
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": self.max_tokens_per_summary,
                "response_format": response_format,
            }
        }

    def _generate_batch(
        self,
        document_id: str,
        document_title: str,
        l1_chunks: list[dict],
        l2_chunks: list[dict],
        checkpoint: Optional[Checkpoint] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[list[Summary], dict[str, Summary]]:
        """Batch API processing for large documents (50% cheaper).

        Sections are processed locally (extractive, instant).
        Only chapter LLM calls use the OpenAI Batch API.
        """
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available, falling back to async")
            return asyncio.run(self._generate_async(
                document_id, document_title, l1_chunks, l2_chunks, checkpoint, progress_callback
            ))

        all_summaries: list[Summary] = []
        section_summaries: dict[str, Summary] = {}

        # Phase 1: Section summaries (extractive, instant)
        if progress_callback:
            progress_callback(0, 100, f"Extracting {len(l2_chunks)} sections (local)")

        for i, chunk in enumerate(l2_chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            summary = self.summarize_section(
                chunk_content=chunk.get("content", ""),
                chunk_id=chunk_id,
                document_id=document_id,
                document_title=document_title,
                section_path=chunk.get("section_path"),
                has_table=chunk.get("has_table", False),
            )
            section_summaries[chunk_id] = summary
            all_summaries.append(summary)

        # Phase 2: Chapter summaries via Batch API
        client = openai.OpenAI()
        batch_file_path = self.checkpoint_dir / f"{document_id}_batch.jsonl"
        batch_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if we're resuming a batch
        if checkpoint and checkpoint.batch_id:
            return self._resume_batch(
                client, checkpoint.batch_id, document_id, document_title,
                l1_chunks, l2_chunks, progress_callback
            )

        if progress_callback:
            progress_callback(5, 100, "Creating batch requests for chapters...")

        # Create batch requests for chapters
        requests = []
        for idx, chunk in enumerate(l1_chunks):
            chunk_id = chunk.get("id", f"l1_chunk_{idx}")

            child_ids = chunk.get("child_ids", [])
            child_summaries = [
                section_summaries[cid]
                for cid in child_ids
                if cid in section_summaries
            ]

            section_context = "\n\n".join(
                f"Section {i+1}:\n{s.content}"
                for i, s in enumerate(child_summaries)
            )
            if not section_context:
                section_context = chunk.get("content", "")[:2000]

            context = f"Document: {document_title}"
            chapter_title = chunk.get("source_section")
            if chapter_title:
                context += f"\nChapter: {chapter_title}"

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "chapter_summary",
                    "schema": ChapterSummaryOutput.model_json_schema(),
                }
            }

            requests.append(self._create_batch_request(
                custom_id=f"chapter_{idx}_{chunk_id}",
                system_prompt=self._chapter_system_prompt(),
                user_content=f"{context}\n\nSection Summaries:\n{section_context}",
                response_format=response_format,
            ))

        # Write batch file
        with open(batch_file_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        if progress_callback:
            progress_callback(10, 100, "Uploading batch file...")

        # Upload batch file
        with open(batch_file_path, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")

        if progress_callback:
            progress_callback(20, 100, "Creating batch job...")

        # Create batch
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"document_id": document_id}
        )

        # Save checkpoint with batch ID
        if not checkpoint:
            checkpoint = Checkpoint(
                document_id=document_id,
                completed_sections=[],
                completed_chapters=[],
                section_summaries={},
                chapter_summaries=[],
            )
        checkpoint.batch_id = batch.id
        checkpoint.save(Checkpoint.get_path(document_id))

        logger.info(f"Batch job created: {batch.id}")
        logger.info(f"Status: {batch.status}")
        logger.info("Use 'kidkazz summarize status {document_id}' to check progress")

        # Poll and wait
        return self._poll_batch(
            client, batch.id, document_id, document_title,
            l1_chunks, l2_chunks, progress_callback
        )

    def _poll_batch(
        self,
        client,
        batch_id: str,
        document_id: str,
        document_title: str,
        l1_chunks: list[dict],
        l2_chunks: list[dict],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[list[Summary], dict[str, Summary]]:
        """Poll batch job until completion.

        Batch now contains chapter requests (not sections).
        Sections are already processed extractively before batch submission.
        """
        while True:
            batch = client.batches.retrieve(batch_id)

            if progress_callback:
                completed = batch.request_counts.completed if batch.request_counts else 0
                total = batch.request_counts.total if batch.request_counts else len(l1_chunks)
                progress_callback(
                    int(20 + (completed / max(total, 1)) * 60),
                    100,
                    f"Batch {batch.status}: {completed}/{total} chapters"
                )

            if batch.status == "completed":
                break
            elif batch.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch job {batch.status}: {batch.errors}")

            time.sleep(30)  # Poll every 30 seconds

        # Download results
        if progress_callback:
            progress_callback(80, 100, "Downloading results...")

        output_file = client.files.content(batch.output_file_id)
        results = [json.loads(line) for line in output_file.text.strip().split("\n")]

        # Re-generate section summaries (extractive, instant)
        section_summaries: dict[str, Summary] = {}
        all_summaries: list[Summary] = []

        for i, chunk in enumerate(l2_chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            summary = self.summarize_section(
                chunk_content=chunk.get("content", ""),
                chunk_id=chunk_id,
                document_id=document_id,
                document_title=document_title,
                section_path=chunk.get("section_path"),
                has_table=chunk.get("has_table", False),
            )
            section_summaries[chunk_id] = summary
            all_summaries.append(summary)

        # Parse chapter batch results
        chapter_summaries: list[Summary] = []
        parse_errors = 0
        for result in results:
            custom_id = result["custom_id"]
            # custom_id format: chapter_{idx}_{chunk_id}
            parts = custom_id.split("_", 2)
            chunk_id = parts[2] if len(parts) >= 3 else custom_id.replace("chapter_", "")

            if result.get("error"):
                logger.warning(f"Batch error for {chunk_id}: {result['error']}")
                continue

            try:
                response = result["response"]["body"]
                raw_content = response["choices"][0]["message"]["content"]
                content = json.loads(raw_content)

                output = ChapterSummaryOutput(**content)
                summary = self._result_to_summary(
                    output,
                    summary_id=f"summary_{chunk_id}_chapter",
                    level="chapter",
                    source_id=chunk_id,
                    document_id=document_id,
                )

                # Set parent references on child sections
                chunk_dict = next((c for c in l1_chunks if c.get("id") == chunk_id), None)
                if chunk_dict:
                    for cid in chunk_dict.get("child_ids", []):
                        if cid in section_summaries:
                            section_summaries[cid].parent_summary_id = summary.summary_id

                chapter_summaries.append(summary)
                all_summaries.append(summary)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                parse_errors += 1
                logger.warning(f"Failed to parse result for {chunk_id}: {e}")
                continue
            except Exception as e:
                parse_errors += 1
                logger.warning(f"Validation error for {chunk_id}: {e}")
                continue

        if parse_errors > 0:
            logger.warning(f"Skipped {parse_errors} results due to parse errors")

        if progress_callback:
            progress_callback(95, 100, "Document summary (LLM)...")

        # Document summary (single LLM call)
        doc_summary = self.summarize_document(
            document_id=document_id,
            document_title=document_title,
            chapter_summaries=chapter_summaries,
        )

        for chapter_summary in chapter_summaries:
            chapter_summary.parent_summary_id = doc_summary.summary_id

        all_summaries.append(doc_summary)

        # Clean up checkpoint
        checkpoint_path = Checkpoint.get_path(document_id)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        return all_summaries, section_summaries

    def _resume_batch(
        self,
        client,
        batch_id: str,
        document_id: str,
        document_title: str,
        l1_chunks: list[dict],
        l2_chunks: list[dict],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[list[Summary], dict[str, Summary]]:
        """Resume a previously started batch job."""
        logger.info(f"Resuming batch job: {batch_id}")
        return self._poll_batch(
            client, batch_id, document_id, document_title,
            l1_chunks, l2_chunks, progress_callback
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _dict_to_summary(self, data: dict) -> Summary:
        """Convert dict to Summary (for checkpoint restore)."""
        concepts = [
            Concept(**c) if isinstance(c, dict) else c
            for c in data.get("concepts", [])
        ]
        return Summary(
            summary_id=data["summary_id"],
            content=data["content"],
            level=data["level"],
            source_id=data["source_id"],
            document_id=data["document_id"],
            parent_summary_id=data.get("parent_summary_id"),
            key_points=data.get("key_points", []),
            concepts=concepts,
            word_count=data.get("word_count", 0),
            created_at=data.get("created_at", 0),
        )

    def _summary_to_dict(self, summary: Summary) -> dict:
        """Convert Summary to dict (for checkpoint save)."""
        return {
            "summary_id": summary.summary_id,
            "content": summary.content,
            "level": summary.level,
            "source_id": summary.source_id,
            "document_id": summary.document_id,
            "parent_summary_id": summary.parent_summary_id,
            "key_points": summary.key_points,
            "concepts": [asdict(c) for c in summary.concepts],
            "word_count": summary.word_count,
            "created_at": summary.created_at,
        }

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def generate_all_summaries(
        self,
        document_id: str,
        document_title: str,
        chunks: list[dict],
        strategy: Optional[SummarizationStrategy] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[list[Summary], list[Concept]]:
        """
        Generate hierarchical summaries for an entire document.

        Automatically selects the best processing strategy based on chunk count,
        or uses the specified strategy.

        Args:
            document_id: Document identifier
            document_title: Document title
            chunks: List of chunk dicts with 'id', 'content', 'level', 'section_path'
            strategy: Force a specific strategy (auto-detect if None)
            progress_callback: Optional callback(current, total, message) for progress

        Returns:
            Tuple of:
            - List of all summaries (section, chapter, document)
            - List of deduplicated concepts with occurrence tracking
        """
        # Group chunks by level
        l1_chunks = [c for c in chunks if c.get("level") == 1]
        l2_chunks = [c for c in chunks if c.get("level") == 2]

        # Determine strategy based on L1 count (sections are extractive, no API calls)
        if strategy is None:
            strategy = self.determine_strategy(len(l1_chunks))

        api_calls = len(l1_chunks) + 1  # chapters + document
        logger.info(
            f"Generating summaries: {len(l2_chunks)} sections (extractive), "
            f"{len(l1_chunks)} chapters (LLM), 1 document (LLM)"
        )
        logger.info(f"Using strategy: {strategy.value} (~{api_calls} API calls)")

        # Load checkpoint if exists
        checkpoint_path = Checkpoint.get_path(document_id)
        checkpoint = Checkpoint.load(checkpoint_path)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {len(checkpoint.completed_sections)} sections completed")

        # Execute based on strategy
        if strategy == SummarizationStrategy.SEQUENTIAL:
            all_summaries, section_summaries = self._generate_sequential(
                document_id, document_title, l1_chunks, l2_chunks, checkpoint, progress_callback
            )
        elif strategy == SummarizationStrategy.ASYNC:
            all_summaries, section_summaries = asyncio.run(self._generate_async(
                document_id, document_title, l1_chunks, l2_chunks, checkpoint, progress_callback
            ))
        elif strategy == SummarizationStrategy.BATCH:
            all_summaries, section_summaries = self._generate_batch(
                document_id, document_title, l1_chunks, l2_chunks, checkpoint, progress_callback
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Aggregate concepts
        logger.info("Aggregating concepts across summaries")
        all_concepts = self._aggregate_concepts(all_summaries, document_id)

        logger.info(f"Generated {len(all_summaries)} summaries and {len(all_concepts)} unique concepts")
        return all_summaries, all_concepts

    def _aggregate_concepts(
        self,
        summaries: list[Summary],
        document_id: str,
    ) -> list[Concept]:
        """Aggregate and deduplicate concepts from all summaries."""
        concept_map: dict[str, Concept] = {}

        for summary in summaries:
            for concept in summary.concepts:
                concept_id = concept.concept_id

                if concept_id not in concept_map:
                    concept_map[concept_id] = Concept(
                        concept_id=concept_id,
                        name=concept.name,
                        concept_type=concept.concept_type,
                        definition=concept.definition,
                        aliases=list(concept.aliases),
                        document_id=document_id,
                        occurrences=list(concept.occurrences),
                        occurrence_count=1,
                    )
                else:
                    existing = concept_map[concept_id]

                    for occ in concept.occurrences:
                        if occ not in existing.occurrences:
                            existing.occurrences.append(occ)

                    existing.occurrence_count += 1

                    if len(concept.definition) > len(existing.definition):
                        existing.definition = concept.definition

                    for alias in concept.aliases:
                        if alias not in existing.aliases:
                            existing.aliases.append(alias)

        return sorted(
            concept_map.values(),
            key=lambda c: c.occurrence_count,
            reverse=True,
        )
