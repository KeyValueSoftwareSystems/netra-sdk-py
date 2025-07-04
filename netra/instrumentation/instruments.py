from enum import Enum
from typing import Any

from traceloop.sdk import Instruments


class CustomInstruments(Enum):
    AIOHTTP = "aiohttp"
    COHEREAI = "cohere_ai"
    HTTPX = "httpx"
    MISTRALAI = "mistral_ai"
    QDRANTDB = "qdrant_db"
    WEAVIATEDB = "weaviate_db"
    GOOGLE_GENERATIVEAI = "google_genai"
    FASTAPI = "fastapi"


class NetraInstrumentSet(Enum):
    """A custom enum that stores the original enum class in an 'origin' attribute."""

    def __new__(cls: Any, value: Any, origin: Any = None) -> Any:
        member = object.__new__(cls)
        member._value_ = value
        member.origin = origin
        return member


merged_members = {}

for member in Instruments:
    merged_members[member.name] = (member.value, Instruments)

for member in CustomInstruments:
    merged_members[member.name] = (member.value, CustomInstruments)

NetraInstruments = NetraInstrumentSet("NetraInstruments", merged_members)


#####################################################################################
"""
NetraInstruments follows the given structure. Refer this for usage within Netra SDK:

class NetraInstruments(OriginEnum):
    AIOHTTP = "aiohttp"
    ALEPHALPHA = "alephalpha"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    CHROMA = "chroma"
    COHEREAI = "cohere_ai"
    CREW = "crew"
    FASTAPI = "fastapi"
    GOOGLE_GENERATIVEAI = "google_genai"
    GROQ = "groq"
    HAYSTACK = "haystack"
    HTTPX = "httpx"
    LANCEDB = "lancedb"
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"
    MARQO = "marqo"
    MCP = "mcp"
    MILVUS = "milvus"
    MISTRALAI = "mistral_ai"
    OLLAMA = "ollama"
    OPENAI = "openai"
    PINECONE = "pinecone"
    PYMYSQL = "pymysql"
    QDRANTDB = "qdrant_db"
    REDIS = "redis"
    REPLICATE = "replicate"
    REQUESTS = "requests"
    SAGEMAKER = "sagemaker"
    TOGETHER = "together"
    TRANSFORMERS = "transformers"
    URLLIB3 = "urllib3"
    VERTEXAI = "vertexai"
    WATSONX = "watsonx"
    WEAVIATEDB = "weaviate_db"
"""
