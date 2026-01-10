// KidKazz RAG Queries

// ========== MUTATION QUERIES ==========

// Add a new document
QUERY AddDocument(doc_id: String, title: String, tags: String, chunk_count: U32, created_at: I64) =>
    doc <- AddN<Document>({
        doc_id: doc_id,
        title: title,
        tags: tags,
        created_at: created_at,
        chunk_count: chunk_count
    })
    RETURN doc

// Add a chunk
QUERY AddChunk(
    chunk_id: String,
    content: String,
    level: U32,
    token_count: U32,
    word_count: U32,
    document_id: String,
    semantic_type: String,
    topic_tags: String,
    section_path: String,
    source_section: String,
    sequence_position: U32,
    parent_id: String,
    child_ids: String,
    sibling_ids: String,
    prev_id: String,
    next_id: String,
    has_table: U32,
    has_code: U32,
    has_math: U32,
    has_list: U32
) =>
    chunk <- AddN<Chunk>({
        chunk_id: chunk_id,
        content: content,
        level: level,
        token_count: token_count,
        word_count: word_count,
        document_id: document_id,
        semantic_type: semantic_type,
        topic_tags: topic_tags,
        section_path: section_path,
        source_section: source_section,
        sequence_position: sequence_position,
        parent_id: parent_id,
        child_ids: child_ids,
        sibling_ids: sibling_ids,
        prev_id: prev_id,
        next_id: next_id,
        has_table: has_table,
        has_code: has_code,
        has_math: has_math,
        has_list: has_list
    })
    RETURN chunk

// Link document to chunk
QUERY LinkDocumentChunk(doc_id: ID, chunk_id: ID) =>
    doc <- N<Document>(doc_id)
    chunk <- N<Chunk>(chunk_id)
    AddE<HasChunk>::From(doc)::To(chunk)
    RETURN doc

// Add parent-child relationship
QUERY AddParentChild(parent_id: ID, child_id: ID) =>
    parent <- N<Chunk>(parent_id)
    child <- N<Chunk>(child_id)
    AddE<ParentOf>::From(parent)::To(child)
    RETURN parent

// Add next sibling relationship
QUERY AddNextSibling(chunk_id: ID, next_id: ID) =>
    chunk <- N<Chunk>(chunk_id)
    next <- N<Chunk>(next_id)
    AddE<NextSibling>::From(chunk)::To(next)
    RETURN chunk

// ========== READ QUERIES (String-based lookups) ==========

// List all documents
QUERY ListDocuments() =>
    docs <- N<Document>
    RETURN docs

// Get document by user-facing doc_id string
QUERY GetDocumentByDocId(doc_id: String) =>
    docs <- N<Document>::WHERE(_::{doc_id}::EQ(doc_id))
    RETURN docs

// Get chunk by user-facing chunk_id string
QUERY GetChunkByChunkId(chunk_id: String) =>
    chunks <- N<Chunk>::WHERE(_::{chunk_id}::EQ(chunk_id))
    RETURN chunks

// Get all chunks for a document by document_id string
QUERY GetChunksByDocumentId(document_id: String) =>
    chunks <- N<Chunk>::WHERE(_::{document_id}::EQ(document_id))
    RETURN chunks

// Get chunks by level
QUERY GetChunksByLevel(level: U32) =>
    chunks <- N<Chunk>::WHERE(_::{level}::EQ(level))
    RETURN chunks

// Get chunks by document AND level
QUERY GetChunksByDocAndLevel(document_id: String, level: U32) =>
    chunks <- N<Chunk>::WHERE(
        AND(
            _::{document_id}::EQ(document_id),
            _::{level}::EQ(level)
        )
    )
    RETURN chunks

// Get chunks by parent_id (for finding children)
QUERY GetChunksByParentId(parent_id: String) =>
    chunks <- N<Chunk>::WHERE(_::{parent_id}::EQ(parent_id))
    RETURN chunks

// Delete document by doc_id string (returns doc for client to then DROP)
QUERY DeleteDocumentByDocId(doc_id: String) =>
    doc <- N<Document>::WHERE(_::{doc_id}::EQ(doc_id))
    RETURN doc

// ========== DELETE QUERIES (using internal IDs) ==========

// Delete a chunk by internal ID (also removes connected edges and vectors)
QUERY DropChunk(chunk_id: ID) =>
    DROP N<Chunk>(chunk_id)
    RETURN "Removed chunk"

// Delete a document by internal ID
QUERY DropDocument(doc_id: ID) =>
    DROP N<Document>(doc_id)
    RETURN "Removed document"

// Delete HasChunk edges from document (useful before dropping document)
QUERY DropDocumentChunkEdges(doc_id: ID) =>
    DROP N<Document>(doc_id)::OutE<HasChunk>
    RETURN "Removed document chunk edges"

// Delete HasEmbedding edge from chunk
QUERY DropChunkEmbeddingEdge(chunk_id: ID) =>
    DROP N<Chunk>(chunk_id)::OutE<HasEmbedding>
    RETURN "Removed chunk embedding edge"

// ========== UPDATE QUERIES ==========

// Update chunk content
QUERY UpdateChunkContent(chunk_id: ID, content: String, word_count: U32) =>
    updated <- N<Chunk>(chunk_id)::UPDATE({
        content: content,
        word_count: word_count
    })
    RETURN updated

// ========== READ QUERIES (Internal ID-based - for edge traversals) ==========

// Get a document by internal ID
QUERY GetDocument(doc_id: ID) =>
    doc <- N<Document>(doc_id)
    RETURN doc

// Get a chunk by internal ID
QUERY GetChunk(chunk_id: ID) =>
    chunk <- N<Chunk>(chunk_id)
    RETURN chunk

// Get all chunks for a document via edge traversal
QUERY GetDocumentChunks(doc_id: ID) =>
    chunks <- N<Document>(doc_id)::Out<HasChunk>
    RETURN chunks

// Get parent chunk via edge
QUERY GetParentChunk(chunk_id: ID) =>
    parent <- N<Chunk>(chunk_id)::In<ParentOf>
    RETURN parent

// Get child chunks via edge
QUERY GetChildChunks(chunk_id: ID) =>
    children <- N<Chunk>(chunk_id)::Out<ParentOf>
    RETURN children

// Get next chunk in sequence via edge
QUERY GetNextChunk(chunk_id: ID) =>
    next <- N<Chunk>(chunk_id)::Out<NextSibling>
    RETURN next

// Get previous chunk via edge
QUERY GetPrevChunk(chunk_id: ID) =>
    prev <- N<Chunk>(chunk_id)::In<NextSibling>
    RETURN prev

// Get sibling chunks via edge
QUERY GetSiblingChunks(chunk_id: ID) =>
    siblings <- N<Chunk>(chunk_id)::Out<SiblingOf>
    RETURN siblings

// ========== VECTOR QUERIES ==========

// Add vector embedding for a chunk
// Note: AddV expects (vector_data, {properties}) - vector first, then metadata
QUERY AddChunkVector(embedding: [F64], model_name: String, embedding_dim: U32) =>
    vec <- AddV<ChunkVector>(embedding, {
        model_name: model_name,
        embedding_dim: embedding_dim
    })
    RETURN vec

// Link chunk to its embedding
QUERY LinkChunkVector(chunk_id: ID, vector_id: ID) =>
    chunk <- N<Chunk>(chunk_id)
    vec <- V<ChunkVector>(vector_id)
    AddE<HasEmbedding>::From(chunk)::To(vec)
    RETURN chunk

// ========== SEARCH QUERIES ==========

// Vector similarity search (filtering done in Python post-processing)
QUERY SearchSimilar(query_vec: [F64], top_k: U32) =>
    results <- SearchV<ChunkVector>(query_vec, top_k)
    RETURN results

// Vector search with MMR reranking
QUERY SearchSimilarMMR(query_vec: [F64], top_k: U32, lambda: F64) =>
    results <- SearchV<ChunkVector>(query_vec, top_k)::RerankMMR(lambda: lambda)
    RETURN results

// BM25 keyword search
QUERY SearchKeywordBM25(keyword: String, limit: U32) =>
    results <- SearchBM25<Chunk>(keyword, limit)
    RETURN results


// ========== CONCEPT MUTATIONS ==========

// Add a new concept
QUERY AddConcept(
    concept_id: String,
    name: String,
    definition: String,
    concept_type: String,
    source_documents: String,
    aliases: String
) =>
    concept <- AddN<Concept>({
        concept_id: concept_id,
        name: name,
        definition: definition,
        concept_type: concept_type,
        source_documents: source_documents,
        aliases: aliases
    })
    RETURN concept

// Link chunk to concept it defines
QUERY LinkChunkDefinesConcept(chunk_id: ID, concept_id: ID) =>
    chunk <- N<Chunk>(chunk_id)
    concept <- N<Concept>(concept_id)
    AddE<DefinesConcept>::From(chunk)::To(concept)
    RETURN chunk

// Link chunk to concept it mentions
QUERY LinkChunkMentionsConcept(chunk_id: ID, concept_id: ID) =>
    chunk <- N<Chunk>(chunk_id)
    concept <- N<Concept>(concept_id)
    AddE<MentionsConcept>::From(chunk)::To(concept)
    RETURN chunk

// Link two concepts with a relationship
QUERY LinkConceptRelatesTo(from_id: ID, to_id: ID) =>
    from_concept <- N<Concept>(from_id)
    to_concept <- N<Concept>(to_id)
    AddE<RelatesTo>::From(from_concept)::To(to_concept)
    RETURN from_concept

// ========== CONCEPT QUERIES ==========

// Get concept by name
QUERY GetConceptByName(name: String) =>
    concepts <- N<Concept>::WHERE(_::{name}::EQ(name))
    RETURN concepts

// Get concept by concept_id
QUERY GetConceptById(concept_id: String) =>
    concepts <- N<Concept>::WHERE(_::{concept_id}::EQ(concept_id))
    RETURN concepts

// List all concepts
QUERY ListConcepts() =>
    concepts <- N<Concept>
    RETURN concepts

// List concepts from a specific document
QUERY ListDocumentConcepts(document_id: String) =>
    chunks <- N<Chunk>::WHERE(_::{document_id}::EQ(document_id))
    concepts <- chunks::Out<DefinesConcept>
    RETURN concepts

// ========== CONCEPT TRAVERSALS ==========

// Get chunks that define a concept (for citations)
QUERY GetConceptDefinitionChunks(concept_id: ID) =>
    chunks <- N<Concept>(concept_id)::In<DefinesConcept>
    RETURN chunks

// Get chunks that mention a concept
QUERY GetConceptMentionChunks(concept_id: ID) =>
    chunks <- N<Concept>(concept_id)::In<MentionsConcept>
    RETURN chunks

// Get related concepts (one hop out)
QUERY GetRelatedConcepts(concept_id: ID) =>
    related <- N<Concept>(concept_id)::Out<RelatesTo>
    RETURN related

// Get concepts that relate TO this one (reverse)
QUERY GetConceptDependents(concept_id: ID) =>
    dependents <- N<Concept>(concept_id)::In<RelatesTo>
    RETURN dependents

// ========== CONCEPT UPDATE ==========

// Update concept source_documents and aliases (for cross-document merging)
QUERY UpdateConcept(concept_id: String, source_documents: String, aliases: String) =>
    concept <- N<Concept>::WHERE(_::{concept_id}::EQ(concept_id))::UPDATE({
        source_documents: source_documents,
        aliases: aliases
    })
    RETURN concept

// Update concept with definition
QUERY UpdateConceptWithDefinition(concept_id: String, source_documents: String, aliases: String, definition: String) =>
    concept <- N<Concept>::WHERE(_::{concept_id}::EQ(concept_id))::UPDATE({
        source_documents: source_documents,
        aliases: aliases,
        definition: definition
    })
    RETURN concept

// ========== CONCEPT DELETION ==========

// Delete a concept
QUERY DropConcept(concept_id: ID) =>
    DROP N<Concept>(concept_id)
    RETURN "Removed concept"
