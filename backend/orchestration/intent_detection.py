import json
from enum import Enum 
import logging

from openai import AsyncAzureOpenAI

class Intent(str, Enum):
    ANSWER_QUESTION = "ANSWER_QUESTION"
    CREATE_TICKET = "CREATE_TICKET"
    GET_TICKET_STATUS = "GET_TICKET_STATUS"
    
system_instructions = f"""
You are an AI assistant that identifies the user's intent from their request.
You have three possible intents:
1) ANSWER_QUESTION
2) CREATE_TICKET
3) GET_TICKET_STATUS

Where <intent_value> MUST be one of:
- {{{Intent.ANSWER_QUESTION.value}}}
- {{{Intent.CREATE_TICKET.value}}}
- {{{Intent.GET_TICKET_STATUS.value}}}
If uncertain, interpret the request as best as possible within these three categories.
Only respond with the JSON (no additional text, no explanation).
"""

class UserIntentDetector:
    def __init__(
        self, azure_openai_client: AsyncAzureOpenAI, model: str) -> None:
        self.__azure_openai_client = azure_openai_client
        self.__model = model
        
    async def get_user_intent(self, request_messages: list) -> Intent:
        """
        Determine user intent by calling an LLM model.

        :param model_args: Arguments for the Azure OpenAI chat completion.
        :param azure_openai_client: An initialized Azure OpenAI client with appropriate credentials.
        :return: An Intent enum value indicating the detected intent.
        """

        try:
            
            messages = [
                {
                    "role": "system",
                    "content": system_instructions
                }
            ]

            for message in request_messages:
                if message:
                    if message["role"] == "assistant" and "context" in message:
                        context_obj = json.loads(message["context"])
                        messages.append(
                            {
                                "role": message["role"],
                                "content": message["content"],
                                "context": context_obj
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": message["role"],
                                "content": message["content"]
                            }
                        )
                        
            model_args = {
                "messages": messages,
                "temperature": 0.0,
                "stream": False,
                "model": self.__model
            }
            

            raw_response = (
                await self.__azure_openai_client.chat.completions.with_raw_response.create(**model_args)
            )
            response = raw_response.parse()
            assistant_content = response.choices[0].message.content.strip()
            
            as_json = json.loads(assistant_content)
            intent = Intent(as_json["intent"])
        except Exception as e:
            # If the API call fails at all, return a default error response
            logging.error(f"Error in get_user_intent: {e}")
            return Intent.ANSWER_QUESTION

        return intent
