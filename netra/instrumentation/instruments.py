from enum import Enum
from typing import Any, Optional, Type

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
    AGNO = "agno"
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
    HERMES_AGENT = "hermes_agent"
    HONCHO = "honcho"


class InstrumentSet(Enum):
    """Custom enum that stores the original enum class in an 'origin' attribute.

    Every member carries an ``origin`` attribute that identifies which
    underlying enum (``CustomInstruments`` or ``Instruments``) provides the
    actual instrumentor.  The special ``ALL`` member has ``origin=None`` and
    acts as a sentinel: when included in the ``instruments`` or
    ``root_instruments`` sets passed to ``Netra.init()``, it restores the
    legacy behaviour where **every** instrumentation available in the user's
    environment is enabled automatically — no curated default list is applied.
    """

    origin: Optional[Type[Enum]]

    def __new__(cls, value: Any, origin: Optional[Type[Enum]] = None) -> "InstrumentSet":
        member = object.__new__(cls)
        member._value_ = value
        member.origin = origin
        return member

    ALL = ("__all__", None)

    ADK = ("google_adk", CustomInstruments)
    AGNO = ("agno", CustomInstruments)
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
    CLAUDE_AGENT_SDK = ("claude_agent_sdk", CustomInstruments)
    CLICK = ("click", CustomInstruments)
    COHEREAI = ("cohere_ai", CustomInstruments)
    CONFLUENT_KAFKA = ("confluent_kafka", CustomInstruments)
    CREW = ("crew", Instruments)
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
    HERMES_AGENT = ("hermes_agent", CustomInstruments)
    HONCHO = ("honcho", CustomInstruments)
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
    VOYAGEAI = ("voyageai", Instruments)
    WATSONX = ("watsonx", Instruments)
    WEAVIATEDB = ("weaviate_db", CustomInstruments)
    WRITER = ("writer", Instruments)
    WSGI = ("wsgi", CustomInstruments)


# Public alias — same class, not a copy, so identity/membership checks
# (e.g. ``InstrumentSet.ALL in some_set``) work correctly.
NetraInstruments = InstrumentSet


# Default instrument sets

# These sets are intentionally independent.  Removing an
# instrument from the root allow-list must NOT prevent it from being
# installed — it should still create spans, but those spans will be
# filtered when they appear at the root of a trace.

# Full set of instrumentations installed by default.
DEFAULT_INSTRUMENTS: frozenset[InstrumentSet] = frozenset(
    {
        # LLM / AI providers and agent frameworks
        InstrumentSet.ANTHROPIC,
        InstrumentSet.CARTESIA,
        InstrumentSet.COHEREAI,
        InstrumentSet.CREW,
        InstrumentSet.DEEPGRAM,
        InstrumentSet.ELEVENLABS,
        InstrumentSet.GOOGLE_GENERATIVEAI,
        InstrumentSet.ADK,
        InstrumentSet.AGNO,
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
        InstrumentSet.MCP,
        InstrumentSet.CLAUDE_AGENT_SDK,
        InstrumentSet.HERMES_AGENT,
        # Web frameworks
        InstrumentSet.FASTAPI,
        # Vector DBs
        InstrumentSet.PINECONE,
        InstrumentSet.CHROMA,
        InstrumentSet.WEAVIATEDB,
        InstrumentSet.QDRANTDB,
        InstrumentSet.MILVUS,
        InstrumentSet.LANCEDB,
        InstrumentSet.MARQO,
        # Memory
        InstrumentSet.HONCHO,
        # HTTP clients and database libraries
        InstrumentSet.HTTPX,
        InstrumentSet.REQUESTS,
        InstrumentSet.PYMYSQL,
        InstrumentSet.SQLALCHEMY,
        # Concurrency: propagate OTel trace context across thread boundaries
        # (threading.Thread / ThreadPoolExecutor) so spans created in worker
        # threads attach to the parent workflow trace instead of becoming roots.
        InstrumentSet.THREADING,
    }
)

# Subset of DEFAULT_INSTRUMENTS allowed to produce root-level spans.
DEFAULT_INSTRUMENTS_FOR_ROOT: frozenset[InstrumentSet] = frozenset(
    {
        InstrumentSet.ANTHROPIC,
        InstrumentSet.CARTESIA,
        InstrumentSet.COHEREAI,
        InstrumentSet.CREW,
        InstrumentSet.DEEPGRAM,
        InstrumentSet.ELEVENLABS,
        InstrumentSet.GOOGLE_GENERATIVEAI,
        InstrumentSet.ADK,
        InstrumentSet.AGNO,
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
        InstrumentSet.FASTAPI,
        InstrumentSet.MCP,
        InstrumentSet.CLAUDE_AGENT_SDK,
        InstrumentSet.HERMES_AGENT,
    }
)
