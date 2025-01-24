import json
import logging
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
import requests

class HelixProxyPlugin:
    @kernel_function(
        name="create_ticket",
        description="Creates a support ticket in the Helix system based on the provided description and user details.",
    )
    def create_ticket(
        self,
        ticket_description: Annotated[
            str,
            "The text description of the support ticket to be created.",
        ],
        user_name: Annotated[
            str,
            "Name of the user who creates the ticket.",
        ],
        user_email: Annotated[
            str,
            "Email address of the user who creates the ticket.",
        ]
    ) -> Annotated[
        str, "Returns the response from the Helix API after creating the ticket."
    ]:
        """
        Creates a support ticket using the Helix API.

        Args:
            ticket_description (str): Description of the ticket.
        Returns:
            dict: The response from the Helix API.
        """
        headers = {
            "Authorization": "Bearer TODO",
            "Content-Type": "application/json",
        }

        payload = {
            "description": ticket_description,
        }

        try:
            # response = requests.post(API_URL, json=payload, headers=headers)
            # response.raise_for_status()  # Raise an exception for HTTP errors
            
            logging.log(logging.INFO, f"Creating ticket with description: {ticket_description}")
            
            return json.dumps({
                "ticket_id": 12345,
                "status": "success"
            })
        except requests.exceptions.RequestException as e:
            return json.dumps({
                "status": "error"
            })
