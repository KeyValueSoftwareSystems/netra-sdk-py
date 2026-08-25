"""The instrumentations the SDK knows about, and the sets enabled by default.

This module is pure data: enum members, the default instrument sets and the
scope aliases the span processors key off.  It deliberately imports nothing
beyond the standard library — every consumer of ``netra`` reaches it, including
processes that never call ``Netra.init()``.
"""

from enum import Enum
from typing import Any, Optional


class CustomInstruments(Enum):
    """Instrumentations Netra provides itself rather than delegating to traceloop.

    Retained as public API.  Activation is keyed on :class:`InstrumentSet`
    (see ``netra.instrumentation.registry``), so nothing inside the SDK reads
    this enum any more.
    """

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


class _Origin(Enum):
    """Which family of instrumentors backs an :class:`InstrumentSet` member.

    A plain sentinel rather than the concrete enum class it used to tag.
    ``traceloop.sdk`` costs ~620 ms to import, and tagging members with
    ``traceloop.sdk.Instruments`` forced that cost onto every ``import netra``
    — including processes that never call ``Netra.init()``.  The traceloop
    member itself is still resolved by name, but only at activation time.
    """

    TRACELOOP = "traceloop"
    CUSTOM = "custom"


class InstrumentSet(Enum):
    """Every instrumentation that can be enabled, tagged with its origin.

    Each member carries an ``origin`` (an :class:`_Origin`) naming the family
    that provides the actual instrumentor.  ``ALL`` is a sentinel with
    ``origin=None``: passing it in the ``instruments`` or ``root_instruments``
    set given to ``Netra.init()`` restores the legacy behaviour where **every**
    instrumentation available in the environment is enabled, bypassing the
    curated default lists below.
    """

    origin: Optional[_Origin]

    def __new__(cls, value: Any, origin: Optional[_Origin] = None) -> "InstrumentSet":
        member = object.__new__(cls)
        member._value_ = value
        member.origin = origin
        return member

    ALL = ("__all__", None)

    ADK = ("google_adk", _Origin.CUSTOM)
    AGNO = ("agno", _Origin.CUSTOM)
    AIOHTTP = ("aiohttp", _Origin.CUSTOM)
    AIOHTTP_SERVER = ("aiohttp_server", _Origin.CUSTOM)
    AIO_PIKA = ("aio_pika", _Origin.CUSTOM)
    AIOKAFKA = ("aiokafka", _Origin.CUSTOM)
    AIOPG = ("aiopg", _Origin.CUSTOM)
    ALEPHALPHA = ("alephalpha", _Origin.TRACELOOP)
    ANTHROPIC = ("anthropic", _Origin.TRACELOOP)
    ASGI = ("asgi", _Origin.CUSTOM)
    ASYNCCLICK = ("asyncclick", _Origin.CUSTOM)
    ASYNCIO = ("asyncio", _Origin.CUSTOM)
    ASYNCPG = ("asyncpg", _Origin.CUSTOM)
    AWS_LAMBDA = ("aws_lambda", _Origin.CUSTOM)
    BEDROCK = ("bedrock", _Origin.TRACELOOP)
    BOTO = ("boto", _Origin.CUSTOM)
    BOTO3SQS = ("boto3sqs", _Origin.CUSTOM)
    BOTOCORE = ("botocore", _Origin.CUSTOM)
    CARTESIA = ("cartesia", _Origin.CUSTOM)
    CASSANDRA = ("cassandra", _Origin.CUSTOM)
    CEREBRAS = ("cerebras", _Origin.CUSTOM)
    CELERY = ("celery", _Origin.CUSTOM)
    CHROMA = ("chroma", _Origin.TRACELOOP)
    CLAUDE_AGENT_SDK = ("claude_agent_sdk", _Origin.CUSTOM)
    CLICK = ("click", _Origin.CUSTOM)
    COHEREAI = ("cohere_ai", _Origin.CUSTOM)
    CONFLUENT_KAFKA = ("confluent_kafka", _Origin.CUSTOM)
    CREW = ("crew", _Origin.TRACELOOP)
    DEEPGRAM = ("deepgram", _Origin.CUSTOM)
    DBAPI = ("dbapi", _Origin.CUSTOM)
    DJANGO = ("django", _Origin.CUSTOM)
    DSPY = ("dspy", _Origin.CUSTOM)
    ELASTICSEARCH = ("elasticsearch", _Origin.CUSTOM)
    ELEVENLABS = ("elevenlabs", _Origin.CUSTOM)
    FALCON = ("falcon", _Origin.CUSTOM)
    FASTAPI = ("fastapi", _Origin.CUSTOM)
    FLASK = ("flask", _Origin.CUSTOM)
    GOOGLE_GENERATIVEAI = ("google_genai", _Origin.CUSTOM)
    GROQ = ("groq", _Origin.CUSTOM)
    GRPC = ("grpc", _Origin.CUSTOM)
    HAYSTACK = ("haystack", _Origin.TRACELOOP)
    HERMES_AGENT = ("hermes_agent", _Origin.CUSTOM)
    HONCHO = ("honcho", _Origin.CUSTOM)
    HTTPX = ("httpx", _Origin.CUSTOM)
    JINJA2 = ("jinja2", _Origin.CUSTOM)
    KAFKA_PYTHON = ("kafka_python", _Origin.CUSTOM)
    LANCEDB = ("lancedb", _Origin.TRACELOOP)
    LANGCHAIN = ("langchain", _Origin.TRACELOOP)
    LITELLM = ("litellm", _Origin.CUSTOM)
    LLAMA_INDEX = ("llama_index", _Origin.TRACELOOP)
    LOGGING = ("logging", _Origin.CUSTOM)
    MARQO = ("marqo", _Origin.TRACELOOP)
    MCP = ("mcp", _Origin.TRACELOOP)
    MILVUS = ("milvus", _Origin.TRACELOOP)
    MISTRALAI = ("mistral_ai", _Origin.CUSTOM)
    MYSQL = ("mysql", _Origin.CUSTOM)
    MYSQLCLIENT = ("mysqlclient", _Origin.CUSTOM)
    OLLAMA = ("ollama", _Origin.TRACELOOP)
    OPENAI = ("openai", _Origin.CUSTOM)
    OPENAI_AGENTS = ("openai_agents", _Origin.TRACELOOP)
    PIKA = ("pika", _Origin.CUSTOM)
    PINECONE = ("pinecone", _Origin.TRACELOOP)
    PSYCOPG = ("psycopg", _Origin.CUSTOM)
    PSYCOPG2 = ("psycopg2", _Origin.CUSTOM)
    PYDANTIC_AI = ("pydantic_ai", _Origin.CUSTOM)
    PYMEMCACHE = ("pymemcache", _Origin.CUSTOM)
    PYMONGO = ("pymongo", _Origin.CUSTOM)
    PYMSSQL = ("pymssql", _Origin.CUSTOM)
    PYMYSQL = ("pymysql", _Origin.CUSTOM)
    PYRAMID = ("pyramid", _Origin.CUSTOM)
    QDRANTDB = ("qdrant_db", _Origin.CUSTOM)
    REDIS = ("redis", _Origin.CUSTOM)
    REMOULADE = ("remoulade", _Origin.CUSTOM)
    REPLICATE = ("replicate", _Origin.TRACELOOP)
    REQUESTS = ("requests", _Origin.CUSTOM)
    SAGEMAKER = ("sagemaker", _Origin.TRACELOOP)
    SQLALCHEMY = ("sqlalchemy", _Origin.CUSTOM)
    SQLITE3 = ("sqlite3", _Origin.CUSTOM)
    STARLETTE = ("starlette", _Origin.CUSTOM)
    SYSTEM_METRICS = ("system_metrics", _Origin.CUSTOM)
    THREADING = ("threading", _Origin.CUSTOM)
    TOGETHER = ("together", _Origin.TRACELOOP)
    TORNADO = ("tornado", _Origin.CUSTOM)
    TORTOISEORM = ("tortoiseorm", _Origin.CUSTOM)
    TRANSFORMERS = ("transformers", _Origin.TRACELOOP)
    URLLIB = ("urllib", _Origin.CUSTOM)
    URLLIB3 = ("urllib3", _Origin.CUSTOM)
    VERTEXAI = ("vertexai", _Origin.TRACELOOP)
    VOYAGEAI = ("voyageai", _Origin.TRACELOOP)
    WATSONX = ("watsonx", _Origin.TRACELOOP)
    WEAVIATEDB = ("weaviate_db", _Origin.CUSTOM)
    WRITER = ("writer", _Origin.TRACELOOP)
    WSGI = ("wsgi", _Origin.CUSTOM)


# Public alias — same class, not a copy, so identity/membership checks
# (e.g. ``InstrumentSet.ALL in some_set``) work correctly.
NetraInstruments = InstrumentSet

# Every real instrumentation, i.e. every member except the ``ALL`` sentinel.
# This is what ``InstrumentSet.ALL`` expands to.
ALL_INSTRUMENTS: frozenset[InstrumentSet] = frozenset(
    member for member in InstrumentSet if member is not InstrumentSet.ALL
)


# Default instrument sets
#
# These two sets are intentionally independent.  Removing an instrument from
# the root allow-list must NOT prevent it from being installed — it should
# still create spans, but those spans are filtered when they appear at the root
# of a trace.

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
