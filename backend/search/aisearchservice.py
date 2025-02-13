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
        
        self.__template_search_client = SearchClient(
            endpoint=app_settings.datasource.endpoint,
            index_name= app_settings.datasource.template_index,
            credential=azure_credential,
        )

        self.top = app_settings.datasource.top_k
        self.use_text_search = app_settings.datasource.use_text_search
        self.use_vector_search = app_settings.datasource.use_vector_search
        self.use_semantic_search = app_settings.datasource.use_semantic_search
        self.use_semantic_captions = app_settings.datasource.use_semantic_captions


    async def search_knowledge(
        self,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
    ) -> AsyncSearchItemPaged[Dict]:
        return await self.__search_internal(self.__knowledge_search_client, query_text, filter, vectors)
    
    async def search_templates(
        self,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
    ) -> AsyncSearchItemPaged[Dict]:
        return await self.__search_internal(self.__template_search_client, query_text, filter, vectors)
    
    async def __search_internal( 
        self,
        search_client: SearchClient,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
    ) -> AsyncSearchItemPaged[Dict]:
        
        search_text = query_text if self.use_text_search else ""
        search_vectors = vectors if self.use_vector_search else []
        
        if self.use_semantic_search:
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
        self.__template_search_client.close()