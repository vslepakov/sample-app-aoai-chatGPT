import logging
from typing import Annotated, List, Optional
from pydantic import BaseModel
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from backend.search.aisearchservice import AiSearchService

class Document(BaseModel):
    id: Optional[str] = None
    parent_id: Optional[str] = None
    content: Optional[str] = None
    title: Optional[str] = None
    score: Optional[float] = None
    reranker_score: Optional[float] = None
    

class AzureAISearchPlugin:
    def __init__(self, search_service: AiSearchService, **kwargs):
        self.__search_service = search_service
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
        
        vectors = [VectorizableTextQuery(kind="text", text=query, k_nearest_neighbors=50, fields="text_vector")]
        results = await self.__search_service.search_knowledge(query_text=query, filter=None, vectors=vectors)
        
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
                    (doc.score or 0) >= (self.minimum_search_score or 0)
                    and (doc.reranker_score or 0) >= (self.minimum_reranker_score or 0)
                )
            ]

        return qualified_documents