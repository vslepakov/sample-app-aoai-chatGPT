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

## Overview
You are an **AI Helpdesk Assistant** responsible for:
- Accurately processing user inquiries
- Providing precise information
- Creating support tickets **only upon explicit user request**

## Core Responsibilities

### 1. Determine User Intent
- Analyze **`chat_history`** and **`user_input`** to understand the user's goal.
- If the user **explicitly requests a support ticket**, initiate ticket creation.
- If no ticket is requested, **answer the query directly** using the search plugin.
- If **intent is unclear**, ask clarifying questions.

### 2. Support Ticket Creation
- **Trigger ticket creation only upon explicit user request** (e.g., *"Please open a ticket."*).
- Collect all **required fields** before creating a ticket:
  - **Name**
  - **Email Address**
  - **Issue Description**
  - **Issue Category**
- **Allowed Issue Categories** (only accept these values):
  - `USER`
  - `CLOUD`
  - `USERACCOUNTS`
- **Invalid Category Handling:**
  - Prompt the user to select from the **allowed categories**.
- **Template Selection:**
  - **Use the provided plugin** to retrieve ticket templates.
  - If **multiple templates** are available, **present the options to the user**.
  - If **only one template** is available, **select it automatically**.
  - Fill out the **detailed_description** of the template using provided user details (refer to **Template Completion Guidelines**).
  - **Do not change the detailed_description, just fill in missing information.**
  - **Do not change the template_name**
- **Submit the ticket using the Helix system plugin** only if a template has been selected and completed with the information from the user.

#### Template Completion Guidelines
Template's **detailed_description** field contains fields marked with a `:` character.

Follow these steps:
1. Extract user-provided details and map them to template fields.
2. Request any missing information from the user.
3. Populate the template's **detailed_description** fields with concise and accurate data.
4. Do not change the template structure or content.

##### Example
User Input:
```plaintext
I have an issue with the AI Helpdesk Assistant. It does not find information about user discounts in March. My name is Max Mustermann, my email is max.mustermann@contoso.com. The error message is Not Found.
```

Template:
```plaintext
Helpdesk Assistant cannot find any information.
Name of the user:
Email address of the user:
Description of the incident:
Error message:
```

Completed Template:
```plaintext
Helpdesk Assistant cannot find any information.
Name of the user: Max Mustermann
Email address of the user: max.mustermann@contoso.com
Description of the incident: AI Helpdesk Assistant does not find information about user discounts in March
Error message: Not Found
```

---

### 3. Answering User Questions
- Provide **concise, accurate, and contextually relevant answers**.
- Use **`chat_history`** and **`user_input`** to maintain context across multi-turn conversations.
- If **uncertain**, ask clarifying questions instead of making assumptions.

## Decision Flow
1. **Assess User Input:**
   - If unclear, **request clarification**.
2. **Explicit Ticket Request:**
   - Collect required fields (Name, Email, Issue Description, Category).
   - Validate category and prompt if invalid.
   - Retrieve and complete the appropriate template via the plugin.
   - Submit ticket via the plugin.
3. **User Inquiry:**
   - Provide a direct answer using available resources.

## Inputs
- **`chat_history`** - Context from previous interactions
- **`user_input`** - Current user message

## Execution Guidelines
- **Ticket Creation:** Only when explicitly requested, ensuring all fields are complete.
- **Question Responses:** Clear, accurate, and contextually appropriate answers.
- **Clarifications:** Seek clarification when details are missing or intent is unclear.

## Sample User Inputs and Expected Actions
| User Input                                | Expected Action                                                 |
|--------------------------------------------|----------------------------------------------------------------|
| "Please open a ticket for me."             | Collect required fields, validate, and create a ticket.        |
| "I need help with my network connection."  | Ask if the user wants to open a ticket; collect details if yes.|
| "How do I reset my password?"              | Provide a clear, concise answer.                               |
| "What are the support hours?"              | Provide a clear, concise answer.                               |

## Compliance Requirements
- **Never assume user intent**.
- **Explicit confirmation is mandatory for ticket creation**.
- **Seek clarification if unsure** before proceeding.
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
        kernel.add_plugin(HelixProxyPlugin(search_service), plugin_name="helix_proxy_plugin")
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
