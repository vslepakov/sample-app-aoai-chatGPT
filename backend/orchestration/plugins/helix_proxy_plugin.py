import json
import logging
from typing import Annotated, List, Optional
from azure.search.documents.models import VectorizableTextQuery
from pydantic import BaseModel
from semantic_kernel.functions.kernel_function_decorator import kernel_function
import requests
from backend.search.aisearchservice import AiSearchService

class Document(BaseModel):
    id: Optional[str] = None
    template_name: Optional[str] = None
    category_tier1: Optional[str] = None
    category_tier2: Optional[str] = None
    category_tier3: Optional[str] = None
    description: Optional[str] = None
    detailed_description: Optional[str] = None
    priority: Optional[str] = None
    urgency: Optional[str] = None
    assigned_group: Optional[str] = None
    assigned_group_id: Optional[str] = None
    score: Optional[float] = None
    reranker_score: Optional[float] = None

class HelixProxyPlugin:
    def __init__(self, search_service: AiSearchService, **kwargs):
        self.__search_service = search_service
        self.minimum_search_score = kwargs.get("minimum_search_score", 0.0)
        self.minimum_reranker_score = kwargs.get("minimum_reranker_score", 0.0)
        
    @kernel_function(
        name="get_ticket_templates",
        description="Retrieves one or more templates for creating support tickets in the Helix system.",
    )
    async def get_ticket_templates(
        self,
        category: Annotated[
            str,
            "The category of the incident as provided by the user.",
        ],
        description: Annotated[
            str,
            "Description of the incident or an issue as provided by the user.",
        ],
    ) -> Annotated[
        List[Document], "Returns one or more templates to be considered for support ticket creation."
    ]:
        logging.info(f"Searching for: {description}")
        
        vectors = [VectorizableTextQuery(kind="text", text=description, k_nearest_neighbors=50, fields="text_vector")]
        results = await self.__search_service.search_templates(query_text=description, filter=f"Template_Category_Tier_2 eq '{category}'", vectors=vectors)
        
        documents = []
        async for page in results.by_page():
            async for document in page:
                documents.append(
                    Document(
                        id=document.get("HPD_Template_ID"),
                        template_name=document.get("chunk"),
                        category_tier1=document.get("Template_Category_Tier_1"),
                        category_tier2=document.get("Template_Category_Tier_2"),
                        category_tier3=document.get("Template_Category_Tier_3"),
                        description=document.get("Description"),
                        detailed_description=document.get("Detailed_Decription"),
                        priority=document.get("Priority"),
                        urgency=document.get("Urgency"),
                        assigned_group=document.get("Assigned_Group"),
                        assigned_group_id=document.get("Assigned_Group_ID"),
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


    @kernel_function(
        name="create_ticket",
        description="Creates a support ticket in the Helix system based on the template name and the detailed description.",
    )
    def create_ticket(
        self,
        template_name: Annotated[
            str,
            "The name of the template to be used for support ticket creation.",
        ],
        detailed_description: Annotated[
            str,
            "The comprehensive issue detailed description with all placeholders replaced by user-provided values.",
        ],
    ) -> Annotated[
        str, "Returns the response from the Helix API after creating the ticket."
    ]:
        """
        Creates a support ticket using the Helix API.

        Args:
            template_name (str): Name of the template to use.
            detailed_description (str): Detailed description of the issue.
        Returns:
            dict: The response from the Helix API.
        """
        headers = {
            "Authorization": "Bearer TODO",
            "Content-Type": "application/json",
        }

        payload = {
            "description": template_name,
        }

        try:
            # response = requests.post(API_URL, json=payload, headers=headers)
            # response.raise_for_status()  # Raise an exception for HTTP errors

            logging.log(
                logging.INFO,
                f"Creating ticket with template: {template_name} and description: {detailed_description}",
            )

            return json.dumps({"ticket_id": 12345, "status": "success"})
        except requests.exceptions.RequestException as e:
            return json.dumps({"status": "error"})
