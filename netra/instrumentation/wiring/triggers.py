"""Maps each instrumentation to the module whose import should activate it.

This table is the correctness-critical input to lazy instrumentation: a wrong
or missing entry means an instrumentation silently never activates, and the
failure mode — missing customer telemetry — is one nobody notices for weeks.
It is therefore written and reviewed by hand rather than derived from
distribution metadata, because two cases break derivation:

* **Distribution name is not the import name, and the installation gate may
  name a different distribution than the patch target.**  LangChain is gated on
  the ``langchain`` / ``langgraph`` distributions but patches ``langchain_core``.
* **Namespace packages.**  ``livekit-agents`` lives under the ``livekit``
  namespace package; hooking ``livekit`` would fire while ``livekit.agents`` is
  still mid-import.  The trigger has to be the real module — ``livekit.agents``.
  The same applies to ``google.adk``, ``google.genai`` and ``cerebras.cloud.sdk``.

Values are the module names the instrumentor actually patches.  Multiple
triggers per instrument are expected: the first to fire activates, the rest
become no-ops.

An instrument absent from this table is applied immediately instead
(``netra.instrumentation.wiring.deferral``), so an incomplete table costs startup latency
rather than telemetry.  ``tests/test_lazy_instrumentation.py`` fails when a
member of ``DEFAULT_INSTRUMENTS`` has no entry here.
"""

from netra.instrumentation.instruments import InstrumentSet

INSTRUMENT_TRIGGERS: dict[InstrumentSet, tuple[str, ...]] = {
    # LLM / AI providers and agent frameworks
    InstrumentSet.ADK: ("google.adk",),
    InstrumentSet.AGNO: ("agno",),
    InstrumentSet.ALEPHALPHA: ("aleph_alpha_client",),
    InstrumentSet.ANTHROPIC: ("anthropic",),
    InstrumentSet.BEDROCK: ("boto3", "botocore"),
    InstrumentSet.CARTESIA: ("cartesia",),
    # Namespace package: the SDK lives at cerebras.cloud.sdk, not cerebras.
    InstrumentSet.CEREBRAS: ("cerebras.cloud.sdk",),
    InstrumentSet.CLAUDE_AGENT_SDK: ("claude_agent_sdk",),
    InstrumentSet.COHEREAI: ("cohere",),
    InstrumentSet.CREW: ("crewai",),
    InstrumentSet.DEEPGRAM: ("deepgram",),
    InstrumentSet.DSPY: ("dspy",),
    InstrumentSet.ELEVENLABS: ("elevenlabs",),
    InstrumentSet.GOOGLE_GENERATIVEAI: ("google.genai",),
    InstrumentSet.GROQ: ("groq",),
    InstrumentSet.HAYSTACK: ("haystack",),
    # hermes-agent ships flat top-level modules; the instrumentor verifies the
    # layout before patching, so a same-named module in another project is
    # rejected rather than mis-instrumented.
    InstrumentSet.HERMES_AGENT: ("agent.conversation_loop", "model_tools"),
    InstrumentSet.HONCHO: ("honcho",),
    # The gate names the langchain/langgraph distributions; the patch target is
    # langchain_core, which is installed even when `langchain` is not.
    InstrumentSet.LANGCHAIN: ("langchain_core", "langgraph", "langchain"),
    InstrumentSet.LITELLM: ("litellm",),
    InstrumentSet.LIVEKIT: ("livekit.agents",),
    InstrumentSet.LLAMA_INDEX: ("llama_index",),
    InstrumentSet.MCP: ("mcp",),
    InstrumentSet.MISTRALAI: ("mistralai",),
    InstrumentSet.OLLAMA: ("ollama",),
    InstrumentSet.OPENAI: ("openai",),
    # openai-agents imports as `agents`, not `openai_agents`.
    InstrumentSet.OPENAI_AGENTS: ("agents",),
    InstrumentSet.PYDANTIC_AI: ("pydantic_ai",),
    InstrumentSet.REPLICATE: ("replicate",),
    InstrumentSet.SAGEMAKER: ("boto3", "botocore"),
    InstrumentSet.TOGETHER: ("together",),
    InstrumentSet.TRANSFORMERS: ("transformers",),
    InstrumentSet.VERTEXAI: ("vertexai",),
    InstrumentSet.VOYAGEAI: ("voyageai",),
    InstrumentSet.WATSONX: ("ibm_watsonx_ai", "ibm_watson_machine_learning"),
    InstrumentSet.WRITER: ("writerai",),
    # Vector DBs
    InstrumentSet.CHROMA: ("chromadb",),
    InstrumentSet.LANCEDB: ("lancedb",),
    InstrumentSet.MARQO: ("marqo",),
    InstrumentSet.MILVUS: ("pymilvus",),
    InstrumentSet.PINECONE: ("pinecone",),
    InstrumentSet.QDRANTDB: ("qdrant_client",),
    InstrumentSet.WEAVIATEDB: ("weaviate",),
    # Web frameworks and servers
    InstrumentSet.DJANGO: ("django",),
    InstrumentSet.FALCON: ("falcon",),
    InstrumentSet.FASTAPI: ("fastapi",),
    InstrumentSet.FLASK: ("flask",),
    InstrumentSet.STARLETTE: ("starlette",),
    InstrumentSet.TORNADO: ("tornado",),
    # HTTP clients
    InstrumentSet.AIOHTTP: ("aiohttp",),
    InstrumentSet.HTTPX: ("httpx",),
    InstrumentSet.REQUESTS: ("requests",),
    InstrumentSet.URLLIB: ("urllib.request",),
    InstrumentSet.URLLIB3: ("urllib3",),
    # Databases and caches
    InstrumentSet.ASYNCPG: ("asyncpg",),
    InstrumentSet.CASSANDRA: ("cassandra",),
    InstrumentSet.ELASTICSEARCH: ("elasticsearch",),
    InstrumentSet.MYSQL: ("mysql.connector",),
    InstrumentSet.MYSQLCLIENT: ("MySQLdb",),
    InstrumentSet.PSYCOPG: ("psycopg",),
    InstrumentSet.PSYCOPG2: ("psycopg2",),
    InstrumentSet.PYMEMCACHE: ("pymemcache",),
    InstrumentSet.PYMONGO: ("pymongo",),
    InstrumentSet.PYMSSQL: ("pymssql",),
    InstrumentSet.PYMYSQL: ("pymysql",),
    InstrumentSet.REDIS: ("redis",),
    InstrumentSet.SQLALCHEMY: ("sqlalchemy",),
    InstrumentSet.AIOPG: ("aiopg",),
    # Queues, brokers and task runners
    InstrumentSet.AIO_PIKA: ("aio_pika",),
    InstrumentSet.AIOKAFKA: ("aiokafka",),
    InstrumentSet.BOTO3SQS: ("boto3",),
    InstrumentSet.BOTOCORE: ("botocore",),
    InstrumentSet.CELERY: ("celery",),
    InstrumentSet.CONFLUENT_KAFKA: ("confluent_kafka",),
    InstrumentSet.KAFKA_PYTHON: ("kafka",),
    InstrumentSet.PIKA: ("pika",),
    InstrumentSet.REMOULADE: ("remoulade",),
    # Misc libraries
    InstrumentSet.ASYNCCLICK: ("asyncclick",),
    InstrumentSet.CLICK: ("click",),
    InstrumentSet.GRPC: ("grpc",),
    InstrumentSet.JINJA2: ("jinja2",),
    InstrumentSet.TORTOISEORM: ("tortoise",),
    # Stdlib modules, always already imported: these activate during
    # Netra.init() itself, which is what they did before lazy activation.
    InstrumentSet.ASYNCIO: ("asyncio",),
    InstrumentSet.LOGGING: ("logging",),
    InstrumentSet.SQLITE3: ("sqlite3",),
    InstrumentSet.THREADING: ("threading",),
}


# Instrumentations that are eager *by design*, so their absence from the table
# above is a decision rather than drift.  Neither patches a library a client
# imports: SystemMetricsInstrumentor samples the process itself, and
# AwsLambdaInstrumentor patches the handler named by the Lambda runtime's own
# environment.  There is no import that could sensibly trigger either.
#
# ``tests/test_lazy_instrumentation.py`` asserts that this set plus
# ``INSTRUMENT_TRIGGERS`` covers every instrumentation with an implementation,
# so a genuinely missing trigger cannot hide behind this exemption.
INTENTIONALLY_EAGER_INSTRUMENTS: frozenset[InstrumentSet] = frozenset(
    {
        InstrumentSet.AWS_LAMBDA,
        InstrumentSet.SYSTEM_METRICS,
    }
)
