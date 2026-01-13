"""Document summarization using Instructor + LLM.

This module provides LLM-powered hierarchical summarization of documents
at document, chapter (L1), and section (L2) levels.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

# Optional import for instructor
try:
    import instructor

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    instructor = None  # type: ignore
    INSTRUCTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


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


class ChapterSummaryOutput(BaseModel):
    """LLM output for chapter-level summary."""

    summary: str = Field(
        description="3-5 sentence summary of the chapter's main topics and themes"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="3-5 key concepts or takeaways from this chapter",
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


# ============================================================================
# DocumentSummarizer Class
# ============================================================================


class DocumentSummarizer:
    """Generate hierarchical summaries for documents using LLM."""

    def __init__(
        self,
        provider: str = "anthropic/claude-sonnet-4-20250514",
        max_retries: int = 2,
        max_tokens_per_summary: int = 500,
    ):
        """
        Initialize summarizer.

        Args:
            provider: Instructor provider string
            max_retries: Number of retries on validation failure
            max_tokens_per_summary: Maximum tokens for each summary

        Raises:
            ImportError: If instructor is not installed
        """
        if not INSTRUCTOR_AVAILABLE or instructor is None:
            raise ImportError(
                "instructor is not installed. "
                "Install with: pip install 'kidkazz[concepts]'"
            )

        self.client = instructor.from_provider(provider)
        self.max_retries = max_retries
        self.max_tokens_per_summary = max_tokens_per_summary

    def summarize_section(
        self,
        chunk_content: str,
        chunk_id: str,
        document_id: str,
        document_title: str,
        section_path: Optional[list[str]] = None,
    ) -> Summary:
        """
        Generate summary for a section (L2 chunk).

        Args:
            chunk_content: The text content of the chunk
            chunk_id: Unique chunk identifier
            document_id: Parent document ID
            document_title: Document title for context
            section_path: Breadcrumb path for context

        Returns:
            Summary object for this section
        """
        context = f"Document: {document_title}"
        if section_path:
            context += f"\nSection: {' > '.join(section_path)}"

        system_prompt = (
            "You are summarizing a section from a textbook or technical document. "
            "Create a concise summary that captures the main points and key concepts. "
            "Focus on what this section teaches or explains."
        )

        try:
            result = self.client.create(
                response_model=SectionSummaryOutput,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nContent:\n{chunk_content}"},
                ],
                max_retries=self.max_retries,
            )

            summary_id = f"summary_{chunk_id}_section"
            return Summary(
                summary_id=summary_id,
                content=result.summary,
                level="section",
                source_id=chunk_id,
                document_id=document_id,
                key_points=result.key_points,
            )

        except Exception as e:
            logger.warning(f"Section summarization failed for {chunk_id}: {e}")
            # Return minimal summary on failure
            return Summary(
                summary_id=f"summary_{chunk_id}_section",
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
        """
        Generate summary for a chapter (L1 chunk) from its section summaries.

        Args:
            chunk_id: L1 chunk identifier
            chunk_content: L1 chunk content (for context)
            section_summaries: List of section summaries under this chapter
            document_id: Parent document ID
            document_title: Document title for context
            chapter_title: Chapter title if available

        Returns:
            Summary object for this chapter
        """
        # Build context from section summaries
        section_context = "\n\n".join(
            f"Section {i+1}:\n{s.content}"
            for i, s in enumerate(section_summaries)
        )

        if not section_context:
            # Fall back to chunk content if no section summaries
            section_context = chunk_content[:2000]  # Limit context

        context = f"Document: {document_title}"
        if chapter_title:
            context += f"\nChapter: {chapter_title}"

        system_prompt = (
            "You are summarizing a chapter from a textbook or technical document. "
            "Given the summaries of individual sections, create an overall chapter summary "
            "that captures the main themes and learning objectives. "
            "Synthesize the key concepts into a coherent overview."
        )

        try:
            result = self.client.create(
                response_model=ChapterSummaryOutput,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nSection Summaries:\n{section_context}"},
                ],
                max_retries=self.max_retries,
            )

            summary_id = f"summary_{chunk_id}_chapter"
            return Summary(
                summary_id=summary_id,
                content=result.summary,
                level="chapter",
                source_id=chunk_id,
                document_id=document_id,
                key_points=result.key_points,
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
        """
        Generate document-level summary from chapter summaries.

        Args:
            document_id: Document identifier
            document_title: Document title
            chapter_summaries: List of chapter summaries

        Returns:
            Summary object for the entire document
        """
        # Build context from chapter summaries
        chapter_context = "\n\n".join(
            f"Chapter {i+1}:\n{s.content}\nKey points: {', '.join(s.key_points)}"
            for i, s in enumerate(chapter_summaries)
        )

        if not chapter_context:
            chapter_context = "No chapter summaries available."

        system_prompt = (
            "You are summarizing an entire textbook or technical document. "
            "Given the chapter summaries, create a comprehensive document overview "
            "that explains what the document covers, its purpose, and main themes. "
            "This should help readers understand if this document is relevant to their needs."
        )

        try:
            result = self.client.create(
                response_model=DocumentSummaryOutput,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document: {document_title}\n\nChapter Summaries:\n{chapter_context}"},
                ],
                max_retries=self.max_retries,
            )

            summary_id = f"summary_{document_id}_document"
            return Summary(
                summary_id=summary_id,
                content=result.summary,
                level="document",
                source_id=document_id,
                document_id=document_id,
                key_points=result.key_points,
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

    def generate_all_summaries(
        self,
        document_id: str,
        document_title: str,
        chunks: list[dict],
    ) -> list[Summary]:
        """
        Generate hierarchical summaries for an entire document.

        Args:
            document_id: Document identifier
            document_title: Document title
            chunks: List of chunk dicts with 'id', 'content', 'level', 'section_path'

        Returns:
            List of all summaries (section, chapter, document)
        """
        all_summaries: list[Summary] = []

        # Group chunks by level
        l1_chunks = [c for c in chunks if c.get("level") == 1]
        l2_chunks = [c for c in chunks if c.get("level") == 2]

        logger.info(f"Generating summaries: {len(l2_chunks)} sections, {len(l1_chunks)} chapters")

        # Phase 1: Generate section (L2) summaries
        section_summaries: dict[str, Summary] = {}  # chunk_id -> summary
        for i, chunk in enumerate(l2_chunks):
            logger.info(f"Summarizing section {i+1}/{len(l2_chunks)}")
            summary = self.summarize_section(
                chunk_content=chunk.get("content", ""),
                chunk_id=chunk.get("id", f"chunk_{i}"),
                document_id=document_id,
                document_title=document_title,
                section_path=chunk.get("section_path"),
            )
            section_summaries[chunk.get("id", f"chunk_{i}")] = summary
            all_summaries.append(summary)

        # Phase 2: Generate chapter (L1) summaries
        chapter_summaries: list[Summary] = []
        for i, chunk in enumerate(l1_chunks):
            logger.info(f"Summarizing chapter {i+1}/{len(l1_chunks)}")

            # Find child sections for this chapter
            child_ids = chunk.get("child_ids", [])
            child_summaries = [
                section_summaries[cid]
                for cid in child_ids
                if cid in section_summaries
            ]

            summary = self.summarize_chapter(
                chunk_id=chunk.get("id", f"l1_chunk_{i}"),
                chunk_content=chunk.get("content", ""),
                section_summaries=child_summaries,
                document_id=document_id,
                document_title=document_title,
                chapter_title=chunk.get("source_section"),
            )

            # Link section summaries to chapter
            for child_summary in child_summaries:
                child_summary.parent_summary_id = summary.summary_id

            chapter_summaries.append(summary)
            all_summaries.append(summary)

        # Phase 3: Generate document summary
        logger.info("Generating document summary")
        doc_summary = self.summarize_document(
            document_id=document_id,
            document_title=document_title,
            chapter_summaries=chapter_summaries,
        )

        # Link chapter summaries to document
        for chapter_summary in chapter_summaries:
            chapter_summary.parent_summary_id = doc_summary.summary_id

        all_summaries.append(doc_summary)

        logger.info(f"Generated {len(all_summaries)} summaries total")
        return all_summaries
