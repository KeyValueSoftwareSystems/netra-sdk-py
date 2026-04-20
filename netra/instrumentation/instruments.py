from enum import Enum
from typing import Any

from traceloop.sdk import Instruments


class CustomInstruments(Enum):
    AIOHTTP = "aiohttp"
    COHEREAI = "cohere_ai"
    DSPY = "dspy"
    HTTPX = "httpx"
    LITELLM = "litellm"
    MISTRALAI = "mistral_ai"
    OPENAI = "openai"
    PYDANTIC_AI = "pydantic_ai"
    QDRANTDB = "qdrant_db"
    WEAVIATEDB = "weaviate_db"
    GOOGLE_GENERATIVEAI = "google_genai"
    FASTAPI = "fastapi"
    ADK = "google_adk"
    AIO_PIKA = "aio_pika"
    AIOHTTP_SERVER = "aiohttp_server"
    AIOKAFKA = "aiokafka"
    AIOPG = "aiopg"
    ASGI = "asgi"
    ASYNCCLICK = "asyncclick"
    ASYNCIO = "asyncio"
    ASYNCPG = "asyncpg"
    AWS_LAMBDA = "aws_lambda"
    BOTO = "boto"
    BOTO3SQS = "boto3sqs"
    BOTOCORE = "botocore"
    CASSANDRA = "cassandra"
    CELERY = "celery"
    CLICK = "click"
    CONFLUENT_KAFKA = "confluent_kafka"
    DBAPI = "dbapi"
    DJANGO = "django"
    ELASTICSEARCH = "elasticsearch"
    FALCON = "falcon"
    FLASK = "flask"
    GRPC = "grpc"
    GROQ = "groq"
    JINJA2 = "jinja2"
    KAFKA_PYTHON = "kafka_python"
    LOGGING = "logging"
    MYSQL = "mysql"
    MYSQLCLIENT = "mysqlclient"
    PIKA = "pika"
    PSYCOPG = "psycopg"
    PSYCOPG2 = "psycopg2"
    PYMEMCACHE = "pymemcache"
    PYMONGO = "pymongo"
    PYMSSQL = "pymssql"
    PYMYSQL = "pymysql"
    PYRAMID = "pyramid"
    REDIS = "redis"
    REMOULADE = "remoulade"
    REQUESTS = "requests"
    SQLALCHEMY = "sqlalchemy"
    SQLITE3 = "sqlite3"
    STARLETTE = "starlette"
    SYSTEM_METRICS = "system_metrics"
    THREADING = "threading"
    TORNADO = "tornado"
    TORTOISEORM = "tortoiseorm"
    URLLIB = "urllib"
    URLLIB3 = "urllib3"
    WSGI = "wsgi"
    CEREBRAS = "cerebras"
    DEEPGRAM = "deepgram"
    CARTESIA = "cartesia"
    ELEVENLABS = "elevenlabs"
    CLAUDE_AGENT_SDK = "claude_agent_sdk"


class InstrumentSet(Enum):
    """Custom enum that stores the original enum class in an 'origin' attribute."""

    def __new__(cls: Any, value: Any, origin: Any = None) -> Any:
        member = object.__new__(cls)
        member._value_ = value
        member.origin = origin
        return member

    ADK = ("adk", CustomInstruments)
    AIOHTTP = ("aiohttp", CustomInstruments)
    AIOHTTP_SERVER = ("aiohttp_server", CustomInstruments)
    AIO_PIKA = ("aio_pika", CustomInstruments)
    AIOKAFKA = ("aiokafka", CustomInstruments)
    AIOPG = ("aiopg", CustomInstruments)
    ALEPHALPHA = ("alephalpha", Instruments)
    ANTHROPIC = ("anthropic", Instruments)
    ASGI = ("asgi", CustomInstruments)
    ASYNCCLICK = ("asyncclick", CustomInstruments)
    ASYNCIO = ("asyncio", CustomInstruments)
    ASYNCPG = ("asyncpg", CustomInstruments)
    AWS_LAMBDA = ("aws_lambda", CustomInstruments)
    BEDROCK = ("bedrock", Instruments)
    BOTO = ("boto", CustomInstruments)
    BOTO3SQS = ("boto3sqs", CustomInstruments)
    BOTOCORE = ("botocore", CustomInstruments)
    CARTESIA = ("cartesia", CustomInstruments)
    CASSANDRA = ("cassandra", CustomInstruments)
    CEREBRAS = ("cerebras", CustomInstruments)
    CELERY = ("celery", CustomInstruments)
    CHROMA = ("chroma", Instruments)
    CLICK = ("click", CustomInstruments)
    COHEREAI = ("cohere_ai", CustomInstruments)
    CONFLUENT_KAFKA = ("confluent_kafka", CustomInstruments)
    CREWAI = ("crewai", Instruments)
    DEEPGRAM = ("deepgram", CustomInstruments)
    DBAPI = ("dbapi", CustomInstruments)
    DJANGO = ("django", CustomInstruments)
    DSPY = ("dspy", CustomInstruments)
    ELASTICSEARCH = ("elasticsearch", CustomInstruments)
    ELEVENLABS = ("elevenlabs", CustomInstruments)
    FALCON = ("falcon", CustomInstruments)
    FASTAPI = ("fastapi", CustomInstruments)
    FLASK = ("flask", CustomInstruments)
    GOOGLE_GENERATIVEAI = ("google_genai", CustomInstruments)
    GROQ = ("groq", CustomInstruments)
    GRPC = ("grpc", CustomInstruments)
    HAYSTACK = ("haystack", Instruments)
    HTTPX = ("httpx", CustomInstruments)
    JINJA2 = ("jinja2", CustomInstruments)
    KAFKA_PYTHON = ("kafka_python", CustomInstruments)
    LANCEDB = ("lancedb", Instruments)
    LANGCHAIN = ("langchain", Instruments)
    LITELLM = ("litellm", CustomInstruments)
    LLAMA_INDEX = ("llama_index", Instruments)
    LOGGING = ("logging", CustomInstruments)
    MARQO = ("marqo", Instruments)
    MCP = ("mcp", Instruments)
    MILVUS = ("milvus", Instruments)
    MISTRALAI = ("mistral_ai", CustomInstruments)
    MYSQL = ("mysql", CustomInstruments)
    MYSQLCLIENT = ("mysqlclient", CustomInstruments)
    OLLAMA = ("ollama", Instruments)
    OPENAI = ("openai", CustomInstruments)
    OPENAI_AGENTS = ("openai_agents", Instruments)
    PIKA = ("pika", CustomInstruments)
    PINECONE = ("pinecone", Instruments)
    PSYCOPG = ("psycopg", CustomInstruments)
    PSYCOPG2 = ("psycopg2", CustomInstruments)
    PYDANTIC_AI = ("pydantic_ai", CustomInstruments)
    PYMEMCACHE = ("pymemcache", CustomInstruments)
    PYMONGO = ("pymongo", CustomInstruments)
    PYMSSQL = ("pymssql", CustomInstruments)
    PYMYSQL = ("pymysql", CustomInstruments)
    PYRAMID = ("pyramid", CustomInstruments)
    QDRANTDB = ("qdrant_db", CustomInstruments)
    REDIS = ("redis", CustomInstruments)
    REMOULADE = ("remoulade", CustomInstruments)
    REPLICATE = ("replicate", Instruments)
    REQUESTS = ("requests", CustomInstruments)
    SAGEMAKER = ("sagemaker", Instruments)
    SQLALCHEMY = ("sqlalchemy", CustomInstruments)
    SQLITE3 = ("sqlite3", CustomInstruments)
    STARLETTE = ("starlette", CustomInstruments)
    SYSTEM_METRICS = ("system_metrics", CustomInstruments)
    THREADING = ("threading", CustomInstruments)
    TOGETHER = ("together", Instruments)
    TORNADO = ("tornado", CustomInstruments)
    TORTOISEORM = ("tortoiseorm", CustomInstruments)
    TRANSFORMERS = ("transformers", Instruments)
    URLLIB = ("urllib", CustomInstruments)
    URLLIB3 = ("urllib3", CustomInstruments)
    VERTEXAI = ("vertexai", Instruments)
    WATSONX = ("watsonx", Instruments)
    WEAVIATEDB = ("weaviate_db", CustomInstruments)
    WRITER = ("writer", Instruments)
    WSGI = ("wsgi", CustomInstruments)


NetraInstruments = InstrumentSet


# Curated default instrument set used for root_instruments when the user does
# not pass an explicit value. Covers core LLM/AI providers and frameworks.
DEFAULT_INSTRUMENTS_FOR_ROOT = {
    InstrumentSet.ANTHROPIC,
    InstrumentSet.CARTESIA,
    InstrumentSet.COHEREAI,
    InstrumentSet.CREWAI,
    InstrumentSet.DEEPGRAM,
    InstrumentSet.ELEVENLABS,
    InstrumentSet.GOOGLE_GENERATIVEAI,
    InstrumentSet.ADK,
    InstrumentSet.GROQ,
    InstrumentSet.LANGCHAIN,
    InstrumentSet.LITELLM,
    InstrumentSet.CEREBRAS,
    InstrumentSet.MISTRALAI,
    InstrumentSet.OPENAI,
    InstrumentSet.OLLAMA,
    InstrumentSet.VERTEXAI,
    InstrumentSet.LLAMA_INDEX,
    InstrumentSet.PYDANTIC_AI,
    InstrumentSet.DSPY,
    InstrumentSet.HAYSTACK,
    InstrumentSet.BEDROCK,
    InstrumentSet.TOGETHER,
    InstrumentSet.REPLICATE,
    InstrumentSet.ALEPHALPHA,
    InstrumentSet.WATSONX,
}

# Broader default instrument set used for the ``instruments`` parameter when
# the user does not pass an explicit value. Includes the root defaults plus
# common vector DBs, HTTP client/server, and database ORM/client libraries.
DEFAULT_INSTRUMENTS = DEFAULT_INSTRUMENTS_FOR_ROOT.union(
    {
        InstrumentSet.PINECONE,
        InstrumentSet.CHROMA,
        InstrumentSet.WEAVIATEDB,
        InstrumentSet.QDRANTDB,
        InstrumentSet.MILVUS,
        InstrumentSet.LANCEDB,
        InstrumentSet.MARQO,
        InstrumentSet.PYMYSQL,
        InstrumentSet.REQUESTS,
        InstrumentSet.SQLALCHEMY,
        InstrumentSet.HTTPX,
    }
)


#####################################################################################
"""
NetraInstruments follows the given structure. Refer this for usage within Netra SDK:

class InstrumentSet(Enum):
    ADK = "google_adk"
    AIOHTTP = "aiohttp"
    AIO_PIKA = "aio_pika"
    AIOKAFKA = "aiokafka"
    AIOPG = "aiopg"
    ALEPHALPHA = "alephalpha"
    ANTHROPIC = "anthropic"
    ASYNCCLICK = "asyncclick"
    ASYNCIO = "asyncio"
    ASYNCPG = "asyncpg"
    AWS_LAMBDA = "aws_lambda"
    BEDROCK = "bedrock"
    BOTO = "boto"
    BOTO3SQS = "boto3sqs"
    BOTOCORE = "botocore"
    CARTESIA = "cartesia"
    CASSANDRA = "cassandra"
    CELERY = "celery"
    CHROMA = "chroma"
    CLICK = "click"
    COHEREAI = "cohere_ai"
    CONFLUENT_KAFKA = "confluent_kafka"
    CREW = "crew"
    DEEPGRAM = "deepgram"
    DBAPI = "dbapi"
    DJANGO = "django"
    ELASTICSEARCH = "elasticsearch"
    ELEVENLABS = "elevenlabs"
    FALCON = "falcon"
    FASTAPI = "fastapi"
    FLASK = "flask"
    GOOGLE_GENERATIVEAI = "google_genai"
    GROQ = "groq"
    GRPC = "grpc"
    HAYSTACK = "haystack"
    HTTPX = "httpx"
    JINJA2 = "jinja2"
    KAFKA_PYTHON = "kafka_python"
    LANCEDB = "lancedb"
    LANGCHAIN = "langchain"
    LITELLM = "litellm"
    LLAMA_INDEX = "llama_index"
    LOGGING = "logging"
    MARQO = "marqo"
    MCP = "mcp"
    MILVUS = "milvus"
    MISTRALAI = "mistral_ai"
    MYSQL = "mysql"
    MYSQLCLIENT = "mysqlclient"
    OLLAMA = "ollama"
    OPENAI = "openai"
    PIKA = "pika"
    PINECONE = "pinecone"
    PSYCOPG = "psycopg"
    PSYCOPG2 = "psycopg2"
    PYMEMCACHE = "pymemcache"
    PYMONGO = "pymongo"
    PYMSSQL = "pymssql"
    PYMYSQL = "pymysql"
    QDRANTDB = "qdrant_db"
    REDIS = "redis"
    REMOULADE = "remoulade"
    REPLICATE = "replicate"
    REQUESTS = "requests"
    SAGEMAKER = "sagemaker"
    SQLALCHEMY = "sqlalchemy"
    SQLITE3 = "sqlite3"
    STARLETTE = "starlette"
    SYSTEM_METRICS = "system_metrics"
    THREADING = "threading"
    TOGETHER = "together"
    TORNADO = "tornado"
    TORTOISEORM = "tortoiseorm"
    TRANSFORMERS = "transformers"
    URLLIB = "urllib"
    URLLIB3 = "urllib3"
    VERTEXAI = "vertexai"
    WATSONX = "watsonx"
    WEAVIATEDB = "weaviate_db"
"""
