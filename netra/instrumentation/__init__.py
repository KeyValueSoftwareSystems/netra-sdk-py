import logging
import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import AbstractSet, Any, Callable, Optional

from traceloop.sdk import Instruments
from traceloop.sdk.utils.package_check import is_package_installed

from netra.instrumentation.instruments import DEFAULT_INSTRUMENTS, CustomInstruments, InstrumentSet, NetraInstruments

_ALL_SENTINEL = InstrumentSet.ALL

# Traceloop instrumentors that Netra always replaces with its own custom implementations
# Built with getattr() so the SDK stays importable even when the installed
# traceloop-sdk version lacks some of these enum members.
_TRACELOOP_ALWAYS_BLOCKED_NAMES: tuple[str, ...] = (
    "AGNO",
    "WEAVIATE",
    "QDRANT",
    "GOOGLE_GENERATIVEAI",
    "MISTRAL",
    "OPENAI",
    "GROQ",
    "REDIS",
    "PYMYSQL",
    "REQUESTS",
    "URLLIB3",
    "COHERE",
)
_TRACELOOP_ALWAYS_BLOCKED: frozenset[Instruments] = frozenset(
    member for name in _TRACELOOP_ALWAYS_BLOCKED_NAMES if (member := getattr(Instruments, name, None)) is not None
)


def _contains_all(instrument_set: Optional[AbstractSet[NetraInstruments]]) -> bool:
    """Return ``True`` if the set contains the ``InstrumentSet.ALL`` sentinel.

    Args:
        instrument_set: The set of instruments to check.

    Returns:
        True if the set contains ``InstrumentSet.ALL``, False otherwise.
    """
    return instrument_set is not None and _ALL_SENTINEL in instrument_set


def _classify(
    members: AbstractSet[NetraInstruments],
) -> tuple[set[Instruments], set[CustomInstruments]]:
    """Partition ``InstrumentSet`` members into traceloop and custom buckets.

    Each member's ``origin`` attribute determines whether it maps to a
    ``traceloop.sdk.Instruments`` member or a ``CustomInstruments`` member.
    The ``ALL`` sentinel is silently skipped.

    Args:
        members: Set of ``InstrumentSet`` members to classify.

    Returns:
        A ``(traceloop, custom)`` tuple of concrete enum member sets.
    """
    traceloop: set[Instruments] = set()
    custom: set[CustomInstruments] = set()
    for instrument in members:
        if instrument is _ALL_SENTINEL:
            continue
        if instrument.origin == CustomInstruments:
            member = getattr(CustomInstruments, instrument.name, None)
            if member is not None:
                custom.add(member)
            else:
                logging.warning("Unknown custom instrument: %s", instrument.name)
        else:
            member = getattr(Instruments, instrument.name, None)
            if member is not None:
                traceloop.add(member)
            else:
                logging.warning("Unknown traceloop instrument: %s", instrument.name)
    return traceloop, custom


def init_instrumentations(
    should_enrich_metrics: bool,
    base64_image_uploader: Optional[Callable[[str, str, str], str]],
    instruments: Optional[AbstractSet[NetraInstruments]] = None,
    block_instruments: Optional[AbstractSet[NetraInstruments]] = None,
) -> None:
    """Initialize all requested instrumentations.

    When ``InstrumentSet.ALL`` is present in *instruments*, the curated
    default list is bypassed and every instrumentation available in the
    user's environment is enabled — matching the legacy behaviour before
    default instrument lists were introduced.

    Args:
        should_enrich_metrics: Whether to enrich metrics.
        base64_image_uploader: Optional callback for image uploads.
        instruments: Set of instruments to enable. ``None`` falls back to the
            curated default; a set containing ``InstrumentSet.ALL`` enables
            everything available.
        block_instruments: Set of instruments to explicitly block.  A set
            containing ``InstrumentSet.ALL`` blocks every instrumentation.
    """
    from traceloop.sdk.tracing.tracing import init_instrumentations

    enable_all = _contains_all(instruments)
    block_all = _contains_all(block_instruments)

    resolved = None if enable_all else (instruments or DEFAULT_INSTRUMENTS)

    # Classify user-requested instruments into traceloop / custom buckets
    if resolved is not None:
        traceloop_instruments, netra_custom_instruments = _classify(resolved)
    else:
        all_members = frozenset(m for m in InstrumentSet if m is not _ALL_SENTINEL)
        traceloop_instruments, netra_custom_instruments = _classify(all_members)

    # Classify blocked instruments
    if block_all:
        traceloop_block, netra_custom_block = set(Instruments), set(CustomInstruments)
    elif block_instruments:
        traceloop_block, netra_custom_block = _classify(block_instruments)
    else:
        traceloop_block, netra_custom_block = set(), set()

    # When user specified instruments but none mapped to a category,
    # block that entire category to avoid silently enabling everything.
    if resolved is not None:
        if not traceloop_instruments and not traceloop_block:
            traceloop_block = set(Instruments)
        if not netra_custom_instruments and not netra_custom_block:
            netra_custom_block = set(CustomInstruments)

    # Apply blocking and prepare traceloop arguments
    traceloop_instruments -= traceloop_block
    traceloop_block |= _TRACELOOP_ALWAYS_BLOCKED

    os.environ["TRACELOOP_TELEMETRY"] = "false"
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        init_instrumentations(
            should_enrich_metrics=should_enrich_metrics,
            base64_image_uploader=base64_image_uploader,
            instruments=traceloop_instruments or None,
            block_instruments=traceloop_block,
        )

    # When ALL is requested, default to every custom instrument if none
    # were explicitly classified.
    if enable_all:
        netra_custom_instruments = netra_custom_instruments or set(CustomInstruments)
    netra_custom_instruments -= netra_custom_block

    # Initialize Groq instrumentation.
    if CustomInstruments.GROQ in netra_custom_instruments:
        init_groq_instrumentation()

    # Initialize Google GenAI instrumentation.
    if CustomInstruments.GOOGLE_GENERATIVEAI in netra_custom_instruments:
        init_google_genai_instrumentation()

    # Initialize FastAPI instrumentation.
    if CustomInstruments.FASTAPI in netra_custom_instruments:
        init_fastapi_instrumentation()

    # Initialize Qdrant instrumentation.
    if CustomInstruments.QDRANTDB in netra_custom_instruments:
        init_qdrant_instrumentation()

    # Initialize Weaviate instrumentation.
    if CustomInstruments.WEAVIATEDB in netra_custom_instruments:
        init_weviate_instrumentation()

    # Initialize HTTPX instrumentation.
    if CustomInstruments.HTTPX in netra_custom_instruments:
        init_httpx_instrumentation()

    # Initialize Cohere instrumentation.
    if CustomInstruments.COHEREAI in netra_custom_instruments:
        init_cohere_instrumentation()

    if CustomInstruments.MISTRALAI in netra_custom_instruments:
        init_mistral_instrumentor()

    # Initialize LiteLLM instrumentation.
    if CustomInstruments.LITELLM in netra_custom_instruments:
        init_litellm_instrumentation()

    if CustomInstruments.DSPY in netra_custom_instruments:
        init_dspy_instrumentation()

    # Initialize OpenAI instrumentation.
    if CustomInstruments.OPENAI in netra_custom_instruments:
        init_openai_instrumentation()

    if CustomInstruments.DEEPGRAM in netra_custom_instruments:
        init_deepgram_instrumentation()

    # Initialize ADK instrumentation.
    if CustomInstruments.ADK in netra_custom_instruments:
        init_adk_instrumentation()

    # Initialize Agno instrumentation.
    if CustomInstruments.AGNO in netra_custom_instruments:
        init_agno_instrumentation()

    # Initialize Pydantic AI instrumentation.
    if CustomInstruments.PYDANTIC_AI in netra_custom_instruments:
        init_pydantic_ai_instrumentation()

    # Initialize aio_pika instrumentation.
    if CustomInstruments.AIO_PIKA in netra_custom_instruments:
        init_aio_pika_instrumentation()

    # Initialize aiokafka instrumentation.
    if CustomInstruments.AIOKAFKA in netra_custom_instruments:
        init_aiokafka_instrumentation()

    # Initialize aiopg instrumentation.
    if CustomInstruments.AIOPG in netra_custom_instruments:
        init_aiopg_instrumentation()

    # Initialize asyncclick instrumentation.
    if CustomInstruments.ASYNCCLICK in netra_custom_instruments:
        init_asyncclick_instrumentation()

    # Initialize asyncio instrumentation.
    if CustomInstruments.ASYNCIO in netra_custom_instruments:
        init_asyncio_instrumentation()

    # Initialize asyncpg instrumentation.
    if CustomInstruments.ASYNCPG in netra_custom_instruments:
        init_asyncpg_instrumentation()

    # Initialize aws_lambda instrumentation.
    if CustomInstruments.AWS_LAMBDA in netra_custom_instruments:
        init_aws_lambda_instrumentation()

    # Initialize boto3sqs instrumentation.
    if CustomInstruments.BOTO3SQS in netra_custom_instruments:
        init_boto3sqs_instrumentation()

    # Initialize botocore instrumentation.
    if CustomInstruments.BOTOCORE in netra_custom_instruments:
        init_botocore_instrumentation()

    # Initialize cassandra instrumentation.
    if CustomInstruments.CASSANDRA in netra_custom_instruments:
        init_cassandra_instrumentation()

    # Initialize celery instrumentation.
    if CustomInstruments.CELERY in netra_custom_instruments:
        init_celery_instrumentation()

    # Initialize click instrumentation.
    if CustomInstruments.CLICK in netra_custom_instruments:
        init_click_instrumentation()

    # Initialize confluent_kafka instrumentation.
    if CustomInstruments.CONFLUENT_KAFKA in netra_custom_instruments:
        init_confluent_kafka_instrumentation()

    # Initialize django instrumentation.
    if CustomInstruments.DJANGO in netra_custom_instruments:
        init_django_instrumentation()

    # Initialize elasticsearch instrumentation.
    if CustomInstruments.ELASTICSEARCH in netra_custom_instruments:
        init_elasticsearch_instrumentation()

    # Initialize falcon instrumentation.
    if CustomInstruments.FALCON in netra_custom_instruments:
        init_falcon_instrumentation()

    # Initialize flask instrumentation.
    if CustomInstruments.FLASK in netra_custom_instruments:
        init_flask_instrumentation()

    # Initialize grpc instrumentation.
    if CustomInstruments.GRPC in netra_custom_instruments:
        init_grpc_instrumentation()

    # Initialize jinja2 instrumentation.
    if CustomInstruments.JINJA2 in netra_custom_instruments:
        init_jinja2_instrumentation()

    # Initialize kafka_python instrumentation.
    if CustomInstruments.KAFKA_PYTHON in netra_custom_instruments:
        init_kafka_python_instrumentation()

    # Initialize logging instrumentation.
    if CustomInstruments.LOGGING in netra_custom_instruments:
        init_logging_instrumentation()

    # Initialize mysql instrumentation.
    if CustomInstruments.MYSQL in netra_custom_instruments:
        init_mysql_instrumentation()

    # Initialize mysqlclient instrumentation.
    if CustomInstruments.MYSQLCLIENT in netra_custom_instruments:
        init_mysqlclient_instrumentation()

    # Initialize pika instrumentation.
    if CustomInstruments.PIKA in netra_custom_instruments:
        init_pika_instrumentation()

    # Initialize psycopg instrumentation.
    if CustomInstruments.PSYCOPG in netra_custom_instruments:
        init_psycopg_instrumentation()

    # Initialize psycopg2 instrumentation.
    if CustomInstruments.PSYCOPG2 in netra_custom_instruments:
        init_psycopg2_instrumentation()

    # Initialize pymemcache instrumentation.
    if CustomInstruments.PYMEMCACHE in netra_custom_instruments:
        init_pymemcache_instrumentation()

    # Initialize pymongo instrumentation.
    if CustomInstruments.PYMONGO in netra_custom_instruments:
        init_pymongo_instrumentation()

    # Initialize pymssql instrumentation.
    if CustomInstruments.PYMSSQL in netra_custom_instruments:
        init_pymssql_instrumentation()

    # Initialize pymysql instrumentation.
    if CustomInstruments.PYMYSQL in netra_custom_instruments:
        init_pymysql_instrumentation()

    # Initialize redis instrumentation.
    if CustomInstruments.REDIS in netra_custom_instruments:
        init_redis_instrumentation()

    # Initialize remoulade instrumentation.
    if CustomInstruments.REMOULADE in netra_custom_instruments:
        init_remoulade_instrumentation()

    # Initialize requests instrumentation.
    if CustomInstruments.REQUESTS in netra_custom_instruments:
        init_requests_instrumentation()

    # Initialize sqlalchemy instrumentation.
    if CustomInstruments.SQLALCHEMY in netra_custom_instruments:
        init_sqlalchemy_instrumentation()

    # Initialize sqlite3 instrumentation.
    if CustomInstruments.SQLITE3 in netra_custom_instruments:
        init_sqlite3_instrumentation()

    # Initialize starlette instrumentation.
    if CustomInstruments.STARLETTE in netra_custom_instruments:
        init_starlette_instrumentation()

    # Initialize system_metrics instrumentation.
    if CustomInstruments.SYSTEM_METRICS in netra_custom_instruments:
        init_system_metrics_instrumentation()

    # Initialize threading instrumentation.
    if CustomInstruments.THREADING in netra_custom_instruments:
        init_threading_instrumentation()

    # Initialize tornado instrumentation.
    if CustomInstruments.TORNADO in netra_custom_instruments:
        init_tornado_instrumentation()

    # Initialize tortoiseorm instrumentation.
    if CustomInstruments.TORTOISEORM in netra_custom_instruments:
        init_tortoiseorm_instrumentation()

    # Initialize urllib instrumentation.
    if CustomInstruments.URLLIB in netra_custom_instruments:
        init_urllib_instrumentation()

    # Initialize urllib3 instrumentation.
    if CustomInstruments.URLLIB3 in netra_custom_instruments:
        init_urllib3_instrumentation()

    # Initialize cerebras instrumentation.
    if CustomInstruments.CEREBRAS in netra_custom_instruments:
        init_cerebras_instrumentation()

    # Initialize cartesia instrumentation.
    if CustomInstruments.CARTESIA in netra_custom_instruments:
        init_cartesia_instrumentation()

    # Initialize elevenlabs instrumentation.
    if CustomInstruments.ELEVENLABS in netra_custom_instruments:
        init_elevenlabs_instrumentation()

    # Initialize claude_agent_sdk instrumentation.
    if CustomInstruments.CLAUDE_AGENT_SDK in netra_custom_instruments:
        init_claude_agent_sdk_instrumentation()

    # Subprocess instrumentation always enabled for
    # context propagation in subprocess
    init_subprocess_instrumentation()


def init_groq_instrumentation() -> bool:
    """Initialize Groq instrumentation."""
    try:
        if is_package_installed("groq"):
            from netra.instrumentation.groq import NetraGroqInstrumentor

            instrumentor = NetraGroqInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Groq instrumentor")
        return False


def init_deepgram_instrumentation() -> bool:
    try:
        if is_package_installed("deepgram-sdk"):
            from netra.instrumentation.deepgram import NetraDeepgramInstrumentor

            instrumentor = NetraDeepgramInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Deepgram instrumentor")
        return False


def init_adk_instrumentation() -> bool:
    """Initialize ADK instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("google-adk"):
            from netra.instrumentation.google_adk import NetraGoogleADKInstrumentor

            instrumentor = NetraGoogleADKInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing ADK instrumentor")
        return False


def init_agno_instrumentation() -> bool:
    """Initialize Agno framework instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("agno"):
            from netra.instrumentation.agno import NetraAgnoInstrumentor

            instrumentor = NetraAgnoInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Agno instrumentor")
        return False


def init_google_genai_instrumentation() -> bool:
    """Initialize Google GenAI instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("google-genai"):
            from netra.instrumentation.google_genai import NetraGoogleGenAiInstrumentor

            instrumentor = NetraGoogleGenAiInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Google GenAI instrumentor")
        return False


def init_fastapi_instrumentation() -> bool:
    """Initialize FastAPI instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("fastapi"):
            from netra.instrumentation.fastapi import NetraFastAPIInstrumentor

            instrumentor = NetraFastAPIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing FastAPI instrumentor")
        return False


def init_qdrant_instrumentation() -> bool:
    """Initialize Qdrant instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("qdrant-client"):
            from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

            instrumentor = QdrantInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Qdrant instrumentor")
        return False


def init_weviate_instrumentation() -> bool:
    """Initialize Weaviate instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("weaviate-client"):
            from netra.instrumentation.weaviate import WeaviateInstrumentor

            instrumentor = WeaviateInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Weaviate instrumentor")
        return False


def init_httpx_instrumentation() -> bool:
    """Initialize HTTPX instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("httpx"):
            from netra.instrumentation.httpx import HTTPXInstrumentor

            instrumentor = HTTPXInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing HTTPX instrumentor")
        return False


def init_aiohttp_instrumentation() -> bool:
    """Initialize AIOHTTP instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("aiohttp"):
            from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

            instrumentor = AioHttpClientInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing AIOHTTP instrumentor")
        return False


def init_cohere_instrumentation() -> bool:
    """Initialize Cohere instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("cohere"):
            from netra.instrumentation.cohere import CohereInstrumentor

            instrumentor = CohereInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Cohere instrumentor")
        return False


def init_dspy_instrumentation() -> bool:
    """Initialize DSPy instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        # Support both "dspy-ai" (older versions) and "dspy" (v3.0+)
        if is_package_installed("dspy-ai") or is_package_installed("dspy"):
            from netra.instrumentation.dspy import NetraDSPyInstrumentor

            instrumentor = NetraDSPyInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing DSPy instrumentor")
        return False


def init_mistral_instrumentor() -> bool:
    """Initialize Mistral instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("mistralai"):
            from netra.instrumentation.mistralai import MistralAiInstrumentor

            instrumentor = MistralAiInstrumentor(
                exception_logger=lambda e: logging.exception("Error in Mistral instrumentor"),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Mistral instrumentor")
        return False


def init_litellm_instrumentation() -> bool:
    """Initialize LiteLLM instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("litellm"):
            from netra.instrumentation.litellm import LiteLLMInstrumentor

            instrumentor = LiteLLMInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing LiteLLM instrumentor")
        return False


def init_openai_instrumentation() -> bool:
    """Initialize OpenAI instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("openai"):
            from netra.instrumentation.openai import NetraOpenAIInstrumentor

            instrumentor = NetraOpenAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing OpenAI instrumentor")
        return False


def init_aio_pika_instrumentation() -> bool:
    """Initialize aio_pika instrumentation."""
    try:
        if is_package_installed("aio_pika"):
            from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor

            instrumentor = AioPikaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing aio_pika instrumentor")
        return False


def init_aiokafka_instrumentation() -> bool:
    """Initialize aiokafka instrumentation."""
    try:
        if is_package_installed("aiokafka"):
            from opentelemetry.instrumentation.aiokafka import AIOKafkaInstrumentor

            instrumentor = AIOKafkaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing aiokafka instrumentor")
        return False


def init_aiopg_instrumentation() -> bool:
    """Initialize aiopg instrumentation."""
    try:
        if is_package_installed("aiopg"):
            from opentelemetry.instrumentation.aiopg import AiopgInstrumentor

            instrumentor = AiopgInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing aiopg instrumentor")
        return False


def init_asyncclick_instrumentation() -> bool:
    """Initialize asyncclick instrumentation."""
    try:
        if is_package_installed("asyncclick"):
            from opentelemetry.instrumentation.asyncclick import AsyncClickInstrumentor

            instrumentor = AsyncClickInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing asyncclick instrumentor")
        return False


def init_asyncio_instrumentation() -> bool:
    """Initialize asyncio instrumentation."""
    try:
        if is_package_installed("asyncio"):
            from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor

            instrumentor = AsyncioInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing asyncio instrumentor")
        return False


def init_asyncpg_instrumentation() -> bool:
    """Initialize asyncpg instrumentation."""
    try:
        if is_package_installed("asyncpg"):
            from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

            instrumentor = AsyncPGInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing asyncpg instrumentor")
        return False


def init_aws_lambda_instrumentation() -> bool:
    """Initialize aws_lambda instrumentation."""
    try:
        if is_package_installed("aws_lambda"):
            from opentelemetry.instrumentation.aws_lambda import AwsLambdaInstrumentor

            instrumentor = AwsLambdaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing aws_lambda instrumentor")
        return False


def init_boto3sqs_instrumentation() -> bool:
    """Initialize boto3sqs instrumentation."""
    try:
        if is_package_installed("boto3"):
            from opentelemetry.instrumentation.boto3sqs import Boto3SQSInstrumentor

            instrumentor = Boto3SQSInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing boto3sqs instrumentor")
        return False


def init_botocore_instrumentation() -> bool:
    """Initialize botocore instrumentation."""
    try:
        if is_package_installed("botocore"):
            from opentelemetry.instrumentation.botocore import BotocoreInstrumentor

            instrumentor = BotocoreInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing botocore instrumentor")
        return False


def init_cassandra_instrumentation() -> bool:
    """Initialize cassandra instrumentation."""
    try:
        if is_package_installed("cassandra-driver") and is_package_installed("scylla-driver"):
            from opentelemetry.instrumentation.cassandra import CassandraInstrumentor

            instrumentor = CassandraInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing cassandra instrumentor")
        return False


def init_celery_instrumentation() -> bool:
    """Initialize celery instrumentation."""
    try:
        if is_package_installed("celery"):
            from opentelemetry.instrumentation.celery import CeleryInstrumentor

            instrumentor = CeleryInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing celery instrumentor")
        return False


def init_click_instrumentation() -> bool:
    """Initialize click instrumentation."""
    try:
        if is_package_installed("click"):
            from opentelemetry.instrumentation.click import ClickInstrumentor

            instrumentor = ClickInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing click instrumentor")
        return False


def init_confluent_kafka_instrumentation() -> bool:
    """Initialize confluent_kafka instrumentation."""
    try:
        if is_package_installed("confluent-kafka"):
            from opentelemetry.instrumentation.confluent_kafka import ConfluentKafkaInstrumentor

            instrumentor = ConfluentKafkaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing confluent_kafka instrumentor")
        return False


def init_django_instrumentation() -> bool:
    """Initialize django instrumentation."""
    try:
        if is_package_installed("django"):
            from opentelemetry.instrumentation.django import DjangoInstrumentor

            instrumentor = DjangoInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing django instrumentor")
        return False


def init_elasticsearch_instrumentation() -> bool:
    """Initialize elasticsearch instrumentation."""
    try:
        if is_package_installed("elasticsearch"):
            from opentelemetry.instrumentation.elasticsearch import ElasticsearchInstrumentor

            instrumentor = ElasticsearchInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing elasticsearch instrumentor")
        return False


def init_falcon_instrumentation() -> bool:
    """Initialize falcon instrumentation."""
    try:
        if is_package_installed("falcon"):
            from opentelemetry.instrumentation.falcon import FalconInstrumentor

            instrumentor = FalconInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing falcon instrumentor")
        return False


def init_flask_instrumentation() -> bool:
    """Initialize flask instrumentation."""
    try:
        if is_package_installed("flask"):
            from opentelemetry.instrumentation.flask import FlaskInstrumentor

            instrumentor = FlaskInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing flask instrumentor")
        return False


def init_grpc_instrumentation() -> bool:
    """Initialize grpc instrumentation."""
    try:
        if is_package_installed("grpcio"):
            from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient

            instrumentor = GrpcInstrumentorClient()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing grpc instrumentor")
        return False


def init_jinja2_instrumentation() -> bool:
    """Initialize jinja2 instrumentation."""
    try:
        if is_package_installed("jinja2"):
            from opentelemetry.instrumentation.jinja2 import Jinja2Instrumentor

            instrumentor = Jinja2Instrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing jinja2 instrumentor")
        return False


def init_kafka_python_instrumentation() -> bool:
    """Initialize kafka_python instrumentation."""
    try:
        if is_package_installed("kafka-python"):
            from opentelemetry.instrumentation.kafka import KafkaInstrumentor

            instrumentor = KafkaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing kafka_python instrumentor")
        return False


def init_logging_instrumentation() -> bool:
    """Initialize logging instrumentation."""
    try:
        if is_package_installed("logging"):
            from opentelemetry.instrumentation.logging import LoggingInstrumentor

            instrumentor = LoggingInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing logging instrumentor")
        return False


def init_mysql_instrumentation() -> bool:
    """Initialize mysql instrumentation."""
    try:
        if is_package_installed("mysql-connector-python"):
            from opentelemetry.instrumentation.mysql import MySQLInstrumentor

            instrumentor = MySQLInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing mysql instrumentor")
        return False


def init_mysqlclient_instrumentation() -> bool:
    """Initialize mysqlclient instrumentation."""
    try:
        if is_package_installed("mysqlclient"):
            from opentelemetry.instrumentation.mysqlclient import MySQLClientInstrumentor

            instrumentor = MySQLClientInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing mysqlclient instrumentor")
        return False


def init_pika_instrumentation() -> bool:
    """Initialize pika instrumentation."""
    try:
        if is_package_installed("pika"):
            from opentelemetry.instrumentation.pika import PikaInstrumentor

            instrumentor = PikaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing pika instrumentor")
        return False


def init_psycopg_instrumentation() -> bool:
    """Initialize psycopg instrumentation."""
    try:
        if is_package_installed("psycopg"):
            from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

            instrumentor = PsycopgInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing psycopg instrumentor")
        return False


def init_psycopg2_instrumentation() -> bool:
    """Initialize psycopg2 instrumentation."""
    try:
        if is_package_installed("psycopg2"):
            from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

            instrumentor = Psycopg2Instrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing psycopg2 instrumentor")
        return False


def init_pymemcache_instrumentation() -> bool:
    """Initialize pymemcache instrumentation."""
    try:
        if is_package_installed("pymemcache"):
            from opentelemetry.instrumentation.pymemcache import PymemcacheInstrumentor

            instrumentor = PymemcacheInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing pymemcache instrumentor")
        return False


def init_pymongo_instrumentation() -> bool:
    """Initialize pymongo instrumentation."""
    try:
        if is_package_installed("pymongo"):
            from opentelemetry.instrumentation.pymongo import PymongoInstrumentor

            instrumentor = PymongoInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing pymongo instrumentor")
        return False


def init_pymssql_instrumentation() -> bool:
    """Initialize pymssql instrumentation."""
    try:
        if is_package_installed("pymssql"):
            from opentelemetry.instrumentation.pymssql import PyMSSQLInstrumentor

            instrumentor = PyMSSQLInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing pymssql instrumentor")
        return False


def init_pymysql_instrumentation() -> bool:
    """Initialize pymysql instrumentation."""
    try:
        if is_package_installed("PyMySQL"):
            from opentelemetry.instrumentation.pymysql import PyMySQLInstrumentor

            instrumentor = PyMySQLInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing pymysql instrumentor")
        return False


def init_redis_instrumentation() -> bool:
    """Initialize redis instrumentation."""
    try:
        if is_package_installed("redis"):
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            instrumentor = RedisInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing redis instrumentor")
        return False


def init_remoulade_instrumentation() -> bool:
    """Initialize remoulade instrumentation."""
    try:
        if is_package_installed("remoulade"):
            from opentelemetry.instrumentation.remoulade import RemouladeInstrumentor

            instrumentor = RemouladeInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing remoulade instrumentor")
        return False


def init_requests_instrumentation() -> bool:
    """Initialize requests instrumentation."""
    try:
        if is_package_installed("requests"):
            from netra.instrumentation.requests import RequestsInstrumentor

            instrumentor = RequestsInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing requests instrumentor")
        return False


def init_sqlalchemy_instrumentation() -> bool:
    """Initialize sqlalchemy instrumentation."""
    try:
        if is_package_installed("sqlalchemy"):
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            instrumentor = SQLAlchemyInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing sqlalchemy instrumentor")
        return False


def init_sqlite3_instrumentation() -> bool:
    """Initialize sqlite3 instrumentation."""
    try:
        if is_package_installed("sqlite3"):
            from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor

            instrumentor = SQLite3Instrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing sqlite3 instrumentor")
        return False


def init_starlette_instrumentation() -> bool:
    """Initialize starlette instrumentation."""
    try:
        if is_package_installed("starlette"):
            from opentelemetry.instrumentation.starlette import StarletteInstrumentor

            instrumentor = StarletteInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing starlette instrumentor")
        return False


def init_system_metrics_instrumentation() -> bool:
    """Initialize system_metrics instrumentation."""
    try:
        if is_package_installed("psutil"):
            from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor

            instrumentor = SystemMetricsInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing system_metrics instrumentor")
        return False


def init_threading_instrumentation() -> bool:
    """Initialize threading instrumentation."""
    try:
        from opentelemetry.instrumentation.threading import ThreadingInstrumentor

        instrumentor = ThreadingInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing threading instrumentor")
        return False


def init_tornado_instrumentation() -> bool:
    """Initialize tornado instrumentation."""
    try:
        if is_package_installed("tornado"):
            from opentelemetry.instrumentation.tornado import TornadoInstrumentor

            instrumentor = TornadoInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing tornado instrumentor")
        return False


def init_tortoiseorm_instrumentation() -> bool:
    """Initialize tortoiseorm instrumentation."""
    try:
        if is_package_installed("tortoise-orm"):
            from opentelemetry.instrumentation.tortoiseorm import TortoiseORMInstrumentor

            instrumentor = TortoiseORMInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing tortoiseorm instrumentor")
        return False


def init_urllib_instrumentation() -> bool:
    """Initialize urllib instrumentation."""
    try:
        from opentelemetry.instrumentation.urllib import URLLibInstrumentor

        instrumentor = URLLibInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing urllib instrumentor")
        return False


def init_urllib3_instrumentation() -> bool:
    """Initialize urllib3 instrumentation."""
    try:
        if is_package_installed("urllib3"):
            from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

            instrumentor = URLLib3Instrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing urllib3 instrumentor")
        return False


def init_pydantic_ai_instrumentation() -> bool:
    """Initialize pydantic-ai instrumentation."""
    try:
        if is_package_installed("pydantic-ai"):
            from netra.instrumentation.pydantic_ai import NetraPydanticAIInstrumentor

            instrumentor = NetraPydanticAIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()

        elif is_package_installed("pydantic-ai-slim"):
            from netra.instrumentation.pydantic_ai_slim import NetraPydanticAISlimInstrumentor

            instrumentor = NetraPydanticAISlimInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        else:
            return False
        return True
    except Exception:
        logging.exception("Error initializing pydantic-ai instrumentation")
        return False


def init_cerebras_instrumentation() -> bool:
    """Initialize Cerebras instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("cerebras_cloud_sdk"):
            from netra.instrumentation.cerebras import NetraCerebrasInstrumentor

            instrumentor = NetraCerebrasInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Cerebras instrumentor")
        return False


def init_cartesia_instrumentation() -> bool:
    """Initialize Cartesia instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("cartesia"):
            from netra.instrumentation.cartesia import NetraCartesiaInstrumentor

            instrumentor = NetraCartesiaInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Cartesia instrumentor")
        return False


def init_elevenlabs_instrumentation() -> bool:
    """Initialize Elevenlabs instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("elevenlabs"):
            from netra.instrumentation.elevenlabs import NetraElevenlabsInstrumentor

            instrumentor = NetraElevenlabsInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Elevenlabs instrumentor")
        return False


def init_claude_agent_sdk_instrumentation() -> bool:
    """Initialize Claude Agent SDK instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("claude-agent-sdk"):
            from netra.instrumentation.claude_agent_sdk import NetraClaudeAgentSDKInstrumentor

            instrumentor = NetraClaudeAgentSDKInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing Claude Agent SDK instrumentor")
        return False


def init_subprocess_instrumentation() -> bool:
    """Initialize subprocess context propagation instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        from netra.instrumentation.subprocess import NetraSubprocessInstrumentor

        instrumentor = NetraSubprocessInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
        return True
    except Exception:
        logging.exception("Error initializing subprocess instrumentor")
        return False
