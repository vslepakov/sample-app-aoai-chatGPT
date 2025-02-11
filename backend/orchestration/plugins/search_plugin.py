import logging
from typing import Annotated, List, Optional
from pydantic import BaseModel
from openai import AsyncAzureOpenAI
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    VectorQuery,
    QueryType,
    VectorizedQuery
)
from semantic_kernel.functions.kernel_function_decorator import kernel_function

class Document(BaseModel):
    id: Optional[str] = None
    parent_id: Optional[str] = None
    content: Optional[str] = None
    title: Optional[str] = None
    score: Optional[float] = None
    reranker_score: Optional[float] = None
    

class AzureAISearchPlugin:
    def __init__(self, openai_client: AsyncAzureOpenAI, search_client: SearchClient, embedding_model: str, **kwargs):
        self.search_client = search_client
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.top = kwargs.get("top", 5)
        self.use_text_search = kwargs.get("use_text_search", True)
        self.use_vector_search = kwargs.get("use_vector_search", False)
        self.use_semantic_ranker = kwargs.get("use_semantic_ranker", False)
        self.use_semantic_captions = kwargs.get("use_semantic_captions", False)
        self.minimum_search_score = kwargs.get("minimum_search_score", 0.0)
        self.minimum_reranker_score = kwargs.get("minimum_reranker_score", 0.0)

    @kernel_function(
        name="search",
        description="Answers user's search queries and questions based on the data provided. Does not handle support ticket creation, only reponsible for answering questions."
    )
    async def search(
        self, 
        query: Annotated[
            str,
            "Original, full and unchanged user query, input or question used to run the search query.",
        ]
    ) -> Annotated[
            List[Document], 
            "List of search results as Documents."
        ]:
        """
        Executes a search query using Azure AI Search.

        :param query: The search query string.
        :return: A list of search results as Documents.
        """
        logging.info(f"Searching for: {query}")
        
        vectors: list[VectorQuery] = []
        if self.use_vector_search:
            logging.info("Generating embeddings for the query.")
            vectors.append(await self.__compute_text_embedding(query))
        
        results = await self.__search_internal(
            self.top, 
            query, 
            None, 
            vectors, 
            self.use_text_search, 
            self.use_vector_search, 
            self.use_semantic_ranker, 
            self.use_semantic_captions,
            self.minimum_search_score,
            self.minimum_reranker_score
        )
        
        return results
    
    async def __search_internal(
        self,
        top: int,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
        use_text_search: bool,
        use_vector_search: bool,
        use_semantic_ranker: bool,
        use_semantic_captions: bool,
        minimum_search_score: Optional[float],
        minimum_reranker_score: Optional[float]
    ) -> List[Document]:
        
        search_text = query_text if use_text_search else ""
        search_vectors = vectors if use_vector_search else []
        
        if use_semantic_ranker:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                vector_queries=search_vectors,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="default",
                semantic_query=query_text,
            )
        else:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                vector_queries=search_vectors,
            )
            
        documents = []
        async for page in results.by_page():
            async for document in page:
                documents.append(
                    Document(
                        id=document.get("id"),
                        parent_id=document.get("parent_id"),
                        content=document.get("content"),
                        title=document.get("title"),
                        score=document.get("@search.score"),
                        reranker_score=document.get("@search.reranker_score"),
                    )
                )

            qualified_documents = [
                doc
                for doc in documents
                if (
                    (doc.score or 0) >= (minimum_search_score or 0)
                    and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
                )
            ]

        return qualified_documents
    
    async def __compute_text_embedding(self, query: str):
        embedding = await self.openai_client.embeddings.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.embedding_model,
            input=query
        )
        query_vector = embedding.data[0].embedding
        return VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="contentVector")