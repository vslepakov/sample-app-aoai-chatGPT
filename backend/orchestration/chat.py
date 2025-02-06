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
