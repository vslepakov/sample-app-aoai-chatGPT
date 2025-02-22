import os
from pydantic import (
    BaseModel,
    confloat,
    Field,
    field_validator,
    model_validator,
    PrivateAttr,
    ValidationError,
)
from pydantic.alias_generators import to_snake
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Literal, Optional
from typing_extensions import Self
from quart import Request
from backend.utils import parse_multi_columns

DOTENV_PATH = os.environ.get(
    "DOTENV_PATH",
    os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)
        ),
        ".env"
    )
)
AZURE_OPENAI_API_VERSION = "2024-09-01-preview"

class _UiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UI_",
        env_file=DOTENV_PATH,
        extra="ignore",
        env_ignore_empty=True
    )

    title: str = "Contoso"
    logo: Optional[str] = None
    chat_logo: Optional[str] = None
    chat_title: str = "Start chatting"
    chat_description: str = "This chatbot is configured to answer your questions"
    favicon: str = "/favicon.ico"
    show_share_button: bool = True
    show_chat_history_button: bool = True


class _ChatHistorySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AZURE_COSMOSDB_",
        env_file=DOTENV_PATH,
        extra="ignore",
        env_ignore_empty=True
    )

    database: str
    account: str
    account_key: Optional[str] = None
    conversations_container: str
    enable_feedback: bool = False

class _AzureOpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        env_file=DOTENV_PATH,
        extra='ignore',
        env_ignore_empty=True
    )
    
    model: str
    resource: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0
    top_p: float = 0
    max_tokens: int = 1000
    stop_sequence: Optional[List[str]] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    presence_penalty: Optional[confloat(ge=-2.0, le=2.0)] = 0.0
    frequency_penalty: Optional[confloat(ge=-2.0, le=2.0)] = 0.0
    api_version: str = AZURE_OPENAI_API_VERSION
    embedding_name: Optional[str] = None
        
    @field_validator('stop_sequence', mode='before')
    @classmethod
    def split_contexts(cls, comma_separated_string: str) -> List[str]:
        if isinstance(comma_separated_string, str) and len(comma_separated_string) > 0:
            return parse_multi_columns(comma_separated_string)
        
        return None
    
    @model_validator(mode="after")
    def ensure_endpoint(self) -> Self:
        if self.endpoint:
            return Self
        
        elif self.resource:
            self.endpoint = f"https://{self.resource}.openai.azure.com"
            return Self
        
        raise ValidationError("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE is required")


class _AzureSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AZURE_SEARCH_",
        env_file=DOTENV_PATH,
        extra="ignore",
        env_ignore_empty=True
    )
    _type: Literal["azure_search"] = PrivateAttr(default="azure_search")
    top_k: int = Field(default=5, serialization_alias="top_n_documents")
    service: str = Field(exclude=True)
    endpoint_suffix: str = Field(default="search.windows.net", exclude=True)
    index: str = Field(serialization_alias="index_name")
    template_index: str = Field(serialization_alias="template_index_name")
    use_text_search: bool = Field(default=True, exclude=True)
    use_vector_search: bool = Field(default=True, exclude=True)
    use_semantic_search: bool = Field(default=False, exclude=True)
    use_semantic_captions: bool = Field(default=False, exclude=True)
    
    # Constructed fields
    endpoint: Optional[str] = None
    
    @model_validator(mode="after")
    def set_endpoint(self) -> Self:
        self.endpoint = f"https://{self.service}.{self.endpoint_suffix}"
        return self
        
class _BaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        extra="ignore",
        arbitrary_types_allowed=True,
        env_ignore_empty=True
    )
    datasource_type: Optional[str] = None
    auth_enabled: bool = True
    sanitize_answer: bool = False
    use_promptflow: bool = False


class AppSettings(BaseModel):
    base_settings: _BaseSettings = _BaseSettings()
    azure_openai: _AzureOpenAISettings = _AzureOpenAISettings()
    datasource: _AzureSearchSettings = _AzureSearchSettings()
    ui: Optional[_UiSettings] = _UiSettings()
    
    # Constructed properties
    chat_history: Optional[_ChatHistorySettings] = None
    
    @model_validator(mode="after")
    def set_chat_history_settings(self) -> Self:
        try:
            self.chat_history = _ChatHistorySettings()
        
        except ValidationError:
            self.chat_history = None
        
        return self

app_settings = AppSettings()
