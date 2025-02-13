import json
import logging
from typing import Annotated, List
from semantic_kernel.functions.kernel_function_decorator import kernel_function
import requests


class HelixProxyPlugin:
    def __init__(self):
        # TODO
        pass
        
    @kernel_function(
        name="get_ticket_templates",
        description="Retrieves one or more templates for creating support tickets in the Helix system.",
    )
    def create_ticket(
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
        List[str], "Returns one or more templates to be considered for support ticket creation."
    ]:
        logging.log(logging.INFO, f"Getting ticket templates for category: {category} and description: {description}")
        return ["template1", "template2", "template3"]


    @kernel_function(
        name="create_ticket",
        description="Creates a support ticket in the Helix system based on the provided detailed description and template name.",
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
