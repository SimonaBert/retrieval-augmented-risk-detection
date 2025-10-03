from cat.mad_hatter.decorators import hook
import os

from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    QuantizationSearchParams,
)

## Hook used to replaces the default text splitter with a markdown-aware recursive splitter
@hook (priority=0)
def rabbithole_instantiate_splitter(text_splitter: TextSplitter, cat) -> TextSplitter:
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=2000,
        chunk_overlap=400,

    )

## Hook used to assign metadata
@hook 
def before_rabbithole_insert_memory(doc, cat):
    source_file = doc.metadata['source']
    filename_without_extension = os.path.splitext(source_file)[0]
    
    parts = filename_without_extension.split('-', 1)
    category = parts[0]
    sub_category = parts[1]

    cat.send_ws_message("Category of the document: " + category.lower())
    cat.send_ws_message("Sub_category of the document: " + sub_category.lower())

    doc.metadata['category'] = category.lower()
    doc.metadata['sub_category'] = sub_category.lower()

    return doc

## Hook used to disable the default retrieval 
@hook
def before_cat_recalls_declarative_memories(default_declarative_recall_config, cat):
    default_declarative_recall_config["k"] = 0
    return default_declarative_recall_config

## Hook used to personalize retrieved document
@hook
def after_cat_recalls_memories(cat) -> None:
    k = 30 
    threshold = 0.67

    if "majority_label" in cat.working_memory:
        majority_label = cat.working_memory.majority_label

        # Get all points from declarative memory to determine sub_categories
        all_points, _ = cat.memory.vectors.declarative.get_all_points()

        sub_categories = set()
        for point in all_points:
            metadata = point.payload.get('metadata', {})
            if (metadata.get('category') == majority_label and
                'sub_category' in metadata):
                sub_categories.add(metadata['sub_category'])
        
        # Embed list of tweets
        user_message = cat.working_memory.user_message_json.text
        user_message_embedding = cat.embedder.embed_query(user_message)

        # Retrieve documents for each sub_category 
        for sub_cat in sub_categories:

            memories = cat.memory.vectors.vector_db.search(
                collection_name='declarative',
                query_vector=user_message_embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.category",
                            match=MatchValue(value=majority_label)
                        ),
                        FieldCondition(
                            key="metadata.sub_category", 
                            match=MatchValue(value=sub_cat)
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=True,
                limit=k,
                score_threshold=threshold,
                search_params=SearchParams(
                    quantization=QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
            )

            if memories:
                langchain_docs_from_points = []
                for point in memories:
                    langchain_docs_from_points.append(
                        (
                            Document(
                                page_content=point.payload.get("page_content"),
                                metadata=point.payload.get("metadata") or {},
                            ),
                            point.score,
                            point.vector,
                            point.id,
                        )
                    )
                
                cat.working_memory.declarative_memories.extend(langchain_docs_from_points)
            
        # Reordering
        if hasattr(cat.working_memory, 'declarative_memories') and cat.working_memory.declarative_memories:
            cat.working_memory.declarative_memories = sorted(
                cat.working_memory.declarative_memories, 
                key=lambda x: x[1], 
                reverse=True
            ) 
   
