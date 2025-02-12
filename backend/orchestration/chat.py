from typing import Any, AsyncGenerator
from openai import AsyncAzureOpenAI
from azure.search.documents.aio import SearchClient

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.functions import KernelArguments, KernelFunction
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.contents.streaming_chat_message_content import (
    StreamingChatMessageContent,
)
from semantic_kernel.contents.utils.author_role import AuthorRole
from backend.search.aisearchservice import AiSearchService
from .plugins import HelixProxyPlugin, AzureAISearchPlugin

from backend.utils import format_stream_response

system_message = """
# System Instructions for AI Helpdesk Assistant  

You are an **AI Helpdesk Assistant** responsible for processing user input, providing accurate responses, and creating support tickets when explicitly requested.  

## Core Responsibilities  

1. **Understanding User Intent**  
   - Use **conversation history (`chat_history`)** and **current user input (`user_input`)** to determine the user's intent.  
   - If the user **explicitly requests a support ticket**, proceed with ticket creation.  
   - If no ticket is required, provide a **direct answer** using the search plugin.  

2. **Creating Support Tickets**  
   - Only create a ticket if the user explicitly requests it (e.g., *"Please open a ticket for me."*).  
   - Ensure all **required fields** are provided before submitting the ticket:  
     - **Name**  
     - **Email address**  
     - **Issue description**  
     - **Issue category**  
   - Accepted **issue categories** (only these are allowed):
     - `HARDWARE`
     - `CLOUD`
     - `PORTAL`
     - `SECURITY`
   - *Reject any other issue categories. Ask the user to select from the allowed list.*  
   - Use the **Helix system plugin** to create the ticket.  

3. **Answering User Questions**  
   - If the user is seeking information (e.g., *"How do I reset my password?"*), provide a **concise, accurate, and clear** response.  
   - Leverage available information from **`chat_history`** and **`user_input`** to craft the response.  
   - Ensure all answers are **contextually relevant** and maintain coherence across multi-turn interactions.  

---

## Decision Flow  

1. **Assess `chat_history` and `user_input`**  
   - If the request is **unclear**, ask the user for clarification.  
   - If the user **wants to create a ticket**, gather all required details.  
   - If the user **has not provided all required information**, prompt them for missing details.  
2. **If unsure**, prioritize **resolving the issue directly** unless the user explicitly asks for a ticket.  

---

## Inputs  

- **`chat_history`** - Previous conversation context.  
- **`user_input`** - The latest user message describing their issue or question.  

## Execution Guidelines  

- **Ticket Creation**: Only proceed when explicitly requested, ensuring all required fields are collected.  
- **Answering Queries**: Respond directly and efficiently, using available data.  
- **User Guidance**: If needed, clarify ambiguous requests or guide the user to provide necessary details.  

---

## Example User Inputs and Actions  

**1. Support Ticket Creation:**  
   - _"Please open a ticket for me."_ → **Create a support ticket after collecting required details.**  
   - _"I need help with my network connection."_ → **Ask if the user wants a ticket and, if so, collect necessary details.**  

**2. Question Answering:**  
   - _"How do I reset my password?"_ → **Provide a direct answer.**  
   - _"What are the support hours?"_ → **Provide a direct answer.**  

---

**Strict Adherence Required**:  
- Follow these instructions **precisely**.  
- Do **not** assume intent—**explicit confirmation is required** for ticket creation.  
- If in doubt, **clarify before proceeding**.  
"""


class Chat:
    def __init__(
        self, kernel: Kernel | None = None, chat_function: KernelFunction | None = None
    ) -> None:
        self.__kernel = kernel
        self.__chat_function = chat_function

    @staticmethod
    def __get_kernel(service_id: str, token_provider, search_service: AiSearchService) -> Kernel:
        kernel = Kernel()
        kernel.add_service(
            AzureChatCompletion(service_id=service_id, ad_token_provider=token_provider)
        )
        
        kernel.add_plugin(AzureAISearchPlugin(search_service), plugin_name="search_plugin")
        kernel.add_plugin(HelixProxyPlugin(), plugin_name="helix_proxy_plugin")
        return kernel

    @classmethod
    def create(cls, chat_id: str, token_provider, search_service: AiSearchService) -> "Chat":
        kernel = cls.__get_kernel(chat_id, token_provider, search_service)

        prompt_template_config = PromptTemplateConfig(
            template="{{$chat_history}}{{$user_input}}",
            name="chat",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(
                    name="chat_history",
                    description="The history of the conversation",
                    is_required=True,
                ),
                InputVariable(
                    name="user_input", description="The user input", is_required=True
                ),
            ],
        )

        chat_function = kernel.add_function(
            plugin_name="ChatBot",
            function_name="Chat",
            prompt_template_config=prompt_template_config,
        )

        chat = cls(kernel, chat_function)

        return chat

    async def invoke(
        self,
        request_body
    ) -> AsyncGenerator[dict[str, Any] | dict, Any]:
        history = ChatHistory()
        history.add_system_message(system_message)

        request_messages = request_body.get("messages", [])

        filtered_messages = [
            message for message in request_messages if message.get("role") != "tool"
        ]

        for message in filtered_messages:
            if message:
                if message["role"] == "assistant":
                    history.add_assistant_message(message["content"])
                elif message["role"] == "user":
                    history.add_user_message(message["content"])

        user_input = filtered_messages[-1]["content"]
        
        execution_settings = AzureChatPromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )

        arguments = KernelArguments(settings=execution_settings)
        arguments["user_input"] = user_input
        arguments["chat_history"] = history

        async def generate():
            async for message in self.__kernel.invoke_stream(
                self.__chat_function,
                return_function_results=False,
                arguments=arguments,
            ):
                history_metadata = request_body.get("history_metadata", {})
                msg = message[0]

                if (
                    isinstance(msg, StreamingChatMessageContent)
                    and msg.role == AuthorRole.ASSISTANT
                    and msg.inner_content
                ):
                    yield format_stream_response(msg.inner_content, history_metadata, None)

        return generate()
