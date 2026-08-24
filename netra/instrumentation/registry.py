"""The instrumentors Netra applies itself, as data.

Every entry answers the same three questions — which distributions must be
installed, which module holds the instrumentor, and what it is called — so a
single generic activator (``netra.instrumentation.activation``) can apply any
of them.  Adding an instrumentation means adding a row here, an
``InstrumentSet`` member, and a trigger module in
``netra.instrumentation.triggers``.

Module paths are strings rather than imported symbols on purpose: importing an
instrumentor imports the library it patches, and deferring that import is the
whole point of lazy activation.

Insertion order is activation order.  Callers iterate this table rather than
their own set, so activation stays deterministic regardless of set iteration
order.  Instrumentations absent from this table are never activated.
"""

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from netra.instrumentation.instruments import InstrumentSet

logger = logging.getLogger(__name__)

_NO_KWARGS: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class InstrumentorSpec:
    """How to build one instrumentor, and the distributions it needs.

    Attributes:
        required_distributions: Distribution names that must **all** be
            installed for this instrumentor to apply.  Empty means it always
            applies — used for instrumentors that patch the standard library.
        module: Import path of the module holding the instrumentor class.
        class_name: Name of the instrumentor class within ``module``.
        constructor_kwargs: Keyword arguments for the instrumentor's
            constructor.  ``Any`` because each instrumentor defines its own.
    """

    required_distributions: tuple[str, ...]
    module: str
    class_name: str
    constructor_kwargs: Mapping[str, Any] = field(default=_NO_KWARGS)


def _log_mistral_wrapper_error(exception: Exception) -> None:
    """Report an error raised inside a Mistral instrumentor wrapper.

    Passed to ``MistralAiInstrumentor`` as its ``exception_logger``: without
    one, an error inside a patched Mistral call is swallowed silently.

    Args:
        exception: The exception raised by the wrapper.
    """
    logger.error("Error in Mistral instrumentor", exc_info=exception)


# Instrumentations Netra applies itself, in activation order.
#
# The value is a tuple of candidates and the first one whose distributions are
# all installed wins, which is how a library that ships under more than one
# distribution name is handled (DSPy, Pydantic AI).
CUSTOM_INSTRUMENTORS: dict[InstrumentSet, tuple[InstrumentorSpec, ...]] = {
    # LLM / AI providers and agent frameworks
    InstrumentSet.GROQ: (InstrumentorSpec(("groq",), "netra.instrumentation.groq", "NetraGroqInstrumentor"),),
    InstrumentSet.GOOGLE_GENERATIVEAI: (
        InstrumentorSpec(("google-genai",), "netra.instrumentation.google_genai", "NetraGoogleGenAiInstrumentor"),
    ),
    InstrumentSet.FASTAPI: (
        InstrumentorSpec(("fastapi",), "netra.instrumentation.fastapi", "NetraFastAPIInstrumentor"),
    ),
    InstrumentSet.QDRANTDB: (
        InstrumentorSpec(("qdrant-client",), "opentelemetry.instrumentation.qdrant", "QdrantInstrumentor"),
    ),
    InstrumentSet.WEAVIATEDB: (
        InstrumentorSpec(("weaviate-client",), "netra.instrumentation.weaviate", "WeaviateInstrumentor"),
    ),
    InstrumentSet.HTTPX: (InstrumentorSpec(("httpx",), "netra.instrumentation.httpx", "HTTPXInstrumentor"),),
    InstrumentSet.AIOHTTP: (
        InstrumentorSpec(
            ("aiohttp",),
            "opentelemetry.instrumentation.aiohttp_client",
            "AioHttpClientInstrumentor",
        ),
    ),
    InstrumentSet.COHEREAI: (InstrumentorSpec(("cohere",), "netra.instrumentation.cohere", "CohereInstrumentor"),),
    InstrumentSet.MISTRALAI: (
        InstrumentorSpec(
            ("mistralai",),
            "netra.instrumentation.mistralai",
            "MistralAiInstrumentor",
            {"exception_logger": _log_mistral_wrapper_error},
        ),
    ),
    InstrumentSet.LITELLM: (InstrumentorSpec(("litellm",), "netra.instrumentation.litellm", "LiteLLMInstrumentor"),),
    # DSPy renamed its distribution from dspy-ai to dspy in v3.0.
    InstrumentSet.DSPY: (
        InstrumentorSpec(("dspy-ai",), "netra.instrumentation.dspy", "NetraDSPyInstrumentor"),
        InstrumentorSpec(("dspy",), "netra.instrumentation.dspy", "NetraDSPyInstrumentor"),
    ),
    InstrumentSet.OPENAI: (InstrumentorSpec(("openai",), "netra.instrumentation.openai", "NetraOpenAIInstrumentor"),),
    InstrumentSet.DEEPGRAM: (
        InstrumentorSpec(("deepgram-sdk",), "netra.instrumentation.deepgram", "NetraDeepgramInstrumentor"),
    ),
    InstrumentSet.LIVEKIT: (
        InstrumentorSpec(("livekit-agents",), "netra.instrumentation.livekit", "NetraLiveKitInstrumentor"),
    ),
    InstrumentSet.ADK: (
        InstrumentorSpec(("google-adk",), "netra.instrumentation.google_adk", "NetraGoogleADKInstrumentor"),
    ),
    InstrumentSet.AGNO: (InstrumentorSpec(("agno",), "netra.instrumentation.agno", "NetraAgnoInstrumentor"),),
    # pydantic-ai-slim is the same library without the optional extras, and
    # needs its own instrumentor.  The full distribution wins when both are
    # installed, since it depends on the slim one.
    InstrumentSet.PYDANTIC_AI: (
        InstrumentorSpec(("pydantic-ai",), "netra.instrumentation.pydantic_ai", "NetraPydanticAIInstrumentor"),
        InstrumentorSpec(
            ("pydantic-ai-slim",),
            "netra.instrumentation.pydantic_ai_slim",
            "NetraPydanticAISlimInstrumentor",
        ),
    ),
    # Queues, brokers and task runners
    InstrumentSet.AIO_PIKA: (
        InstrumentorSpec(("aio_pika",), "opentelemetry.instrumentation.aio_pika", "AioPikaInstrumentor"),
    ),
    InstrumentSet.AIOKAFKA: (
        InstrumentorSpec(("aiokafka",), "opentelemetry.instrumentation.aiokafka", "AIOKafkaInstrumentor"),
    ),
    InstrumentSet.AIOPG: (InstrumentorSpec(("aiopg",), "opentelemetry.instrumentation.aiopg", "AiopgInstrumentor"),),
    InstrumentSet.ASYNCCLICK: (
        InstrumentorSpec(("asyncclick",), "opentelemetry.instrumentation.asyncclick", "AsyncClickInstrumentor"),
    ),
    InstrumentSet.ASYNCIO: (InstrumentorSpec((), "opentelemetry.instrumentation.asyncio", "AsyncioInstrumentor"),),
    InstrumentSet.ASYNCPG: (
        InstrumentorSpec(("asyncpg",), "opentelemetry.instrumentation.asyncpg", "AsyncPGInstrumentor"),
    ),
    # No distribution gate: the AWS Lambda instrumentor ships with the SDK and
    # decides for itself whether it is running inside a Lambda runtime.
    InstrumentSet.AWS_LAMBDA: (
        InstrumentorSpec((), "opentelemetry.instrumentation.aws_lambda", "AwsLambdaInstrumentor"),
    ),
    InstrumentSet.BOTO3SQS: (
        InstrumentorSpec(("boto3",), "opentelemetry.instrumentation.boto3sqs", "Boto3SQSInstrumentor"),
    ),
    InstrumentSet.BOTOCORE: (
        InstrumentorSpec(("botocore",), "opentelemetry.instrumentation.botocore", "BotocoreInstrumentor"),
    ),
    # NOTE: both distributions are required, carried over verbatim from the
    # per-library gate this table replaced.  cassandra-driver and scylla-driver
    # are alternatives, so this most likely wants to be two candidate rows the
    # way DSPy is — a behaviour change, deliberately not made here.
    InstrumentSet.CASSANDRA: (
        InstrumentorSpec(
            ("cassandra-driver", "scylla-driver"),
            "opentelemetry.instrumentation.cassandra",
            "CassandraInstrumentor",
        ),
    ),
    InstrumentSet.CELERY: (
        InstrumentorSpec(("celery",), "opentelemetry.instrumentation.celery", "CeleryInstrumentor"),
    ),
    InstrumentSet.CLICK: (InstrumentorSpec(("click",), "opentelemetry.instrumentation.click", "ClickInstrumentor"),),
    InstrumentSet.CONFLUENT_KAFKA: (
        InstrumentorSpec(
            ("confluent-kafka",),
            "opentelemetry.instrumentation.confluent_kafka",
            "ConfluentKafkaInstrumentor",
        ),
    ),
    # Web frameworks
    InstrumentSet.DJANGO: (
        InstrumentorSpec(("django",), "opentelemetry.instrumentation.django", "DjangoInstrumentor"),
    ),
    InstrumentSet.ELASTICSEARCH: (
        InstrumentorSpec(
            ("elasticsearch",),
            "opentelemetry.instrumentation.elasticsearch",
            "ElasticsearchInstrumentor",
        ),
    ),
    InstrumentSet.FALCON: (
        InstrumentorSpec(("falcon",), "opentelemetry.instrumentation.falcon", "FalconInstrumentor"),
    ),
    InstrumentSet.FLASK: (InstrumentorSpec(("flask",), "opentelemetry.instrumentation.flask", "FlaskInstrumentor"),),
    InstrumentSet.GRPC: (
        InstrumentorSpec(("grpcio",), "opentelemetry.instrumentation.grpc", "GrpcInstrumentorClient"),
    ),
    InstrumentSet.JINJA2: (
        InstrumentorSpec(("jinja2",), "opentelemetry.instrumentation.jinja2", "Jinja2Instrumentor"),
    ),
    InstrumentSet.KAFKA_PYTHON: (
        InstrumentorSpec(("kafka-python",), "opentelemetry.instrumentation.kafka", "KafkaInstrumentor"),
    ),
    # Standard library: no distribution to gate on, so these always apply.
    InstrumentSet.LOGGING: (InstrumentorSpec((), "opentelemetry.instrumentation.logging", "LoggingInstrumentor"),),
    # Databases, caches and ORMs
    InstrumentSet.MYSQL: (
        InstrumentorSpec(("mysql-connector-python",), "opentelemetry.instrumentation.mysql", "MySQLInstrumentor"),
    ),
    InstrumentSet.MYSQLCLIENT: (
        InstrumentorSpec(
            ("mysqlclient",),
            "opentelemetry.instrumentation.mysqlclient",
            "MySQLClientInstrumentor",
        ),
    ),
    InstrumentSet.PIKA: (InstrumentorSpec(("pika",), "opentelemetry.instrumentation.pika", "PikaInstrumentor"),),
    InstrumentSet.PSYCOPG: (
        InstrumentorSpec(("psycopg",), "opentelemetry.instrumentation.psycopg", "PsycopgInstrumentor"),
    ),
    InstrumentSet.PSYCOPG2: (
        InstrumentorSpec(("psycopg2",), "opentelemetry.instrumentation.psycopg2", "Psycopg2Instrumentor"),
    ),
    InstrumentSet.PYMEMCACHE: (
        InstrumentorSpec(("pymemcache",), "opentelemetry.instrumentation.pymemcache", "PymemcacheInstrumentor"),
    ),
    InstrumentSet.PYMONGO: (
        InstrumentorSpec(("pymongo",), "opentelemetry.instrumentation.pymongo", "PymongoInstrumentor"),
    ),
    InstrumentSet.PYMSSQL: (
        InstrumentorSpec(("pymssql",), "opentelemetry.instrumentation.pymssql", "PyMSSQLInstrumentor"),
    ),
    InstrumentSet.PYMYSQL: (
        InstrumentorSpec(("PyMySQL",), "opentelemetry.instrumentation.pymysql", "PyMySQLInstrumentor"),
    ),
    InstrumentSet.REDIS: (InstrumentorSpec(("redis",), "opentelemetry.instrumentation.redis", "RedisInstrumentor"),),
    InstrumentSet.REMOULADE: (
        InstrumentorSpec(("remoulade",), "opentelemetry.instrumentation.remoulade", "RemouladeInstrumentor"),
    ),
    # HTTP clients
    InstrumentSet.REQUESTS: (
        InstrumentorSpec(("requests",), "netra.instrumentation.requests", "RequestsInstrumentor"),
    ),
    InstrumentSet.SQLALCHEMY: (
        InstrumentorSpec(("sqlalchemy",), "opentelemetry.instrumentation.sqlalchemy", "SQLAlchemyInstrumentor"),
    ),
    InstrumentSet.SQLITE3: (InstrumentorSpec((), "opentelemetry.instrumentation.sqlite3", "SQLite3Instrumentor"),),
    InstrumentSet.STARLETTE: (
        InstrumentorSpec(("starlette",), "opentelemetry.instrumentation.starlette", "StarletteInstrumentor"),
    ),
    InstrumentSet.SYSTEM_METRICS: (
        InstrumentorSpec(("psutil",), "opentelemetry.instrumentation.system_metrics", "SystemMetricsInstrumentor"),
    ),
    InstrumentSet.THREADING: (
        InstrumentorSpec((), "opentelemetry.instrumentation.threading", "ThreadingInstrumentor"),
    ),
    InstrumentSet.TORNADO: (
        InstrumentorSpec(("tornado",), "opentelemetry.instrumentation.tornado", "TornadoInstrumentor"),
    ),
    InstrumentSet.TORTOISEORM: (
        InstrumentorSpec(("tortoise-orm",), "opentelemetry.instrumentation.tortoiseorm", "TortoiseORMInstrumentor"),
    ),
    InstrumentSet.URLLIB: (InstrumentorSpec((), "opentelemetry.instrumentation.urllib", "URLLibInstrumentor"),),
    InstrumentSet.URLLIB3: (
        InstrumentorSpec(("urllib3",), "opentelemetry.instrumentation.urllib3", "URLLib3Instrumentor"),
    ),
    # Speech, agent and memory SDKs
    InstrumentSet.CEREBRAS: (
        InstrumentorSpec(("cerebras_cloud_sdk",), "netra.instrumentation.cerebras", "NetraCerebrasInstrumentor"),
    ),
    InstrumentSet.CARTESIA: (
        InstrumentorSpec(("cartesia",), "netra.instrumentation.cartesia", "NetraCartesiaInstrumentor"),
    ),
    InstrumentSet.ELEVENLABS: (
        InstrumentorSpec(("elevenlabs",), "netra.instrumentation.elevenlabs", "NetraElevenlabsInstrumentor"),
    ),
    InstrumentSet.CLAUDE_AGENT_SDK: (
        InstrumentorSpec(
            ("claude-agent-sdk",),
            "netra.instrumentation.claude_agent_sdk",
            "NetraClaudeAgentSDKInstrumentor",
        ),
    ),
    InstrumentSet.HERMES_AGENT: (
        InstrumentorSpec(("hermes-agent",), "netra.instrumentation.hermes_agent", "NetraHermesAgentInstrumentor"),
    ),
    InstrumentSet.HONCHO: (
        InstrumentorSpec(("honcho-ai",), "netra.instrumentation.honcho", "NetraHonchoInstrumentor"),
    ),
}


# Subprocess context propagation is not selectable: ``Netra.init()`` always
# applies it, so it has no ``InstrumentSet`` member and no trigger module.
SUBPROCESS_INSTRUMENTOR = InstrumentorSpec((), "netra.instrumentation.subprocess", "NetraSubprocessInstrumentor")
