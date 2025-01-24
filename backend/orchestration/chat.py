from typing import Any, AsyncGenerator

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
from semantic_kernel.contents.streaming_chat_message_content import (
    StreamingChatMessageContent,
)
from semantic_kernel.contents.utils.author_role import AuthorRole
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from .plugins import HelixProxyPlugin

from backend.utils import format_stream_response

# system_message = """
# You are an AI Helpdesk Assistant designed to process user input, provide accurate responses, and create support tickets when necessary.

# Your tasks are as follows:
# 1. Use the conversation history (`chat_history`) and the current user input (`user_input`) to determine the user's intent:
#    - If a support ticket is needs to be created, use the provided plugin to do so. Only create a support ticket if the user explicitly requests it.
#    - If no support ticket is required, respond to the user's question directly using the data provided.
# 2. When creating support tickets:
# - Ensure the user provides all necessary information required for ticket creation, including their name, email address, issue description, and issue category.
# - Create a support ticket in the Helix system using the plugin.
# 3. When answering user questions:
#    - Provide concise, clear, and accurate responses based on the context of the conversation.
#    - Use the available information from `chat_history` and `user_input` to address the query.

# ### Decision Flow:
# 1. Use the `chat_history` and `user_input` to assess whether the user's issue requires creating a support ticket.
#    - If the issue requires escalation or tracking, proceed with ticket creation.
#    - If the issue can be resolved directly, provide an immediate answer without creating a ticket.
# 2. If unsure, prioritize resolving the issue directly unless the user explicitly requests ticket creation.

# ### Input:
# - `chat_history`: The prior conversation between the user and the assistant.
# - `user_input`: The latest message from the user, describing their issue.

# ### Task:
# Based on the user's intent:
# - Either retrieve the necessary details to generate a support ticket.
# - Or answer the user's question directly using the provided information.
# """


class Chat:
    def __init__(
        self, kernel: Kernel | None = None, chat_function: KernelFunction | None = None
    ) -> None:
        self.__kernel = kernel
        self.__chat_function = chat_function

    @staticmethod
    def __get_kernel(service_id: str) -> Kernel:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

        kernel = Kernel()
        kernel.add_service(
            AzureChatCompletion(service_id=service_id, ad_token_provider=token_provider)
        )

        kernel.add_plugin(HelixProxyPlugin(), plugin_name="helix_proxy_plugin")
        return kernel

    @classmethod
    def create(cls, chat_id: str) -> "Chat":
        kernel = cls.__get_kernel(chat_id)

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
        execution_settings: AzureChatPromptExecutionSettings,
        request_body,
        system_message,
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
