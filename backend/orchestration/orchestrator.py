import logging
from typing import Any, AsyncGenerator, Coroutine
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

from .intent_detection import Intent, UserIntentDetector
from .chat import Chat

system_message = """
You are an AI assistant that communicates with users through a chat interface. Your goals are:
1. **Create Support Tickets** on the user's behalf if they explicitly request it (or strongly imply they want to file a ticket) like in 'Create Ticket'. 
2. **Gather Additional Information** if required to complete a ticket or clarify ambiguous requests.

### Important Behaviors:

- **Ask Clarifying Questions**: 
  - If the request is ambiguous, politely ask the user for clarification.  
  - If they want a ticket but haven't provided all required details, prompt the user for any missing pieces of information.

- **Required Fields for a Ticket**:
  - Short summary of the issue (e.g., "I can't log in").
  - User name.
  - User email address

- **Ticket Creation Flow**:
  1. If the user indicates they want to open a ticket (e.g., "Please open a ticket for me"), begin gathering the required fields.
  2. Confirm the user's details. For example: "Your email address is max.mustermann@contoso.com, correct?"
  3. Once all required data is collected, finalize the ticket creation process by invoking the Helix API plugin and returning the ticket id and status to the user.

- **Edge Cases**:
  - If the user cancels the ticket request, confirm the cancellation.

### Implementation Notes:

- You do **not** need to reveal internal processes or mention "I am calling an API" to the user. 
- You **may** ask them politely for required fields or clarifications to help them.

### Objective:

Use this single chat interface to create a support ticket—by applying the guidelines above. 
Always strive to produce clear, coherent, and contextually relevant responses, and handle multi-turn interactions gracefully.

"""


class Orchestrator:   
    async def run(
        self,
        request_body: Any,
        user_intent_detector: UserIntentDetector,
        run_aoai_on_your_data_fn: Coroutine[
            Any, Any, AsyncGenerator[dict[str, Any] | dict, Any]
        ],
    ) -> AsyncGenerator[dict[str, Any] | dict, Any]:
        # Spearating Function Calling and Azure OpenAI on Your Data due to this limitation:
        # https://github.com/Azure/azure-sdk-for-net/blob/main/sdk/openai/Azure.AI.OpenAI/README.md#use-your-own-data-with-azure-openai
        # Consider using Azure AI Search directly with Semantic Kernel
        try:
            messages = request_body.get("messages", [])
            intent = await user_intent_detector.get_user_intent(messages)
            
            if intent == Intent.CREATE_TICKET or intent == Intent.GET_TICKET_STATUS:
                chat = Chat.create("helpdesk_assistant")

                execution_settings = AzureChatPromptExecutionSettings(
                    function_choice_behavior=FunctionChoiceBehavior.Auto()
                )
                return await chat.invoke(execution_settings, request_body, system_message)
            else:
                return await run_aoai_on_your_data_fn()
        except Exception as ex:
            logging.error(f"Orchestrator failed with error: {ex}. User intent: {intent}")
            raise ex
