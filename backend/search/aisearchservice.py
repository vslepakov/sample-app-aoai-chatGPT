from typing import Dict, List, Optional
from azure.search.documents.aio import SearchClient, AsyncSearchItemPaged
from azure.search.documents.models import (
    VectorQuery,
    QueryType,
    QueryCaptionType,
)
from azure.identity.aio import DefaultAzureCredential
from backend.settings import AppSettings

class AiSearchService:
    def __init__(self, azure_credential: DefaultAzureCredential, app_settings: AppSettings, **kwargs):
        self.__knowledge_search_client = SearchClient(
                endpoint=app_settings.datasource.endpoint,
                index_name=app_settings.datasource.index,
                credential=azure_credential,
        )

        # TODO switch to app_settings
        self.top = kwargs.get("top", 5)
        self.use_text_search = kwargs.get("use_text_search", True)
        self.use_vector_search = kwargs.get("use_vector_search", False)
        self.use_semantic_ranker = kwargs.get("use_semantic_ranker", False)
        self.use_semantic_captions = kwargs.get("use_semantic_captions", True)


    async def search_knowledge(
        self,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
    ) -> AsyncSearchItemPaged[Dict]:
        return await self.__search_internal(self.__knowledge_search_client, query_text, filter, vectors)
    
    
    async def __search_internal( 
        self,
        search_client: SearchClient,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
    ) -> AsyncSearchItemPaged[Dict]:
        
        search_text = query_text if self.use_text_search else ""
        search_vectors = vectors if self.use_vector_search else []
        
        if self.use_semantic_ranker:
            results = await search_client.search(
                search_text=search_text,
                filter=filter,
                top=self.top,
                query_caption=QueryCaptionType.EXTRACTIVE if self.use_semantic_captions else None,
                vector_queries=search_vectors,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="default",
                semantic_query=query_text,
            )
        else:
            results = await search_client.search(
                search_text=search_text,
                filter=filter,
                top=self.top,
                vector_queries=search_vectors,
            )

        return results
        
    
    def dispose(self):
        self.__knowledge_search_client.close()