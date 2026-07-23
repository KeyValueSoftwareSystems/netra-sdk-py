"""
Tests that evaluation and simulation runs stamp the ``netra.trace.origin``
attribute on their root span so the FE/BE can distinguish these traces from
normal workflow invocations.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from netra.config import Config
from netra.evaluation.api import Evaluation
from netra.evaluation.models import ItemContext
from netra.simulation.api import Simulation
from netra.simulation.models import ConversationResponse, ConversationStatus, SimulationItem

EXPECTED_ATTRIBUTES = {Config.TRACE_ORIGIN_KEY: Config.TRACE_ORIGIN_EVALUATION}


def _span_wrapper_mock() -> MagicMock:
    """Build a SpanWrapper replacement whose context-managed span reports no
    current OTel span, so trace-id extraction in the call site is skipped."""
    mock_cls = MagicMock()
    enter_value = mock_cls.return_value.__enter__.return_value
    enter_value.get_current_span.return_value = None
    return mock_cls


def test_evaluation_root_span_tagged_with_trace_origin() -> None:
    evaluation = object.__new__(Evaluation)
    evaluation._config = MagicMock()
    evaluation._client = MagicMock()
    evaluation._client.post_run_item.return_value = "item-1"

    ctx = ItemContext(index=0, item_input="hello")

    with (
        patch("netra.evaluation.api.SpanWrapper", _span_wrapper_mock()) as mock_sw,
        patch("netra.evaluation.api.execute_task", new=AsyncMock(return_value=("output", "completed"))),
    ):
        asyncio.run(
            evaluation._execute_item_pipeline(
                run_id="run-1",
                run_name="MyRun",
                ctx=ctx,
                task=lambda _: "output",
            )
        )

    assert mock_sw.call_args.kwargs["attributes"] == EXPECTED_ATTRIBUTES


def test_simulation_root_span_tagged_with_trace_origin() -> None:
    simulation = object.__new__(Simulation)
    simulation._config = MagicMock()
    simulation._client = MagicMock()
    simulation._client.trigger_conversation.return_value = ConversationResponse(
        decision=ConversationStatus.STOP,
        reason="done",
    )

    run_item = SimulationItem(
        run_item_id="ri-1",
        dataset_item_id="di-1",
        message="hi",
        turn_id="t-1",
    )

    with (
        patch("netra.simulation.api.SpanWrapper", _span_wrapper_mock()) as mock_sw,
        patch("netra.simulation.api.execute_task", new=AsyncMock(return_value=("response", "sid-1"))),
    ):
        result = asyncio.run(
            simulation._execute_conversation(
                run_id="run-1",
                run_item=run_item,
                task=MagicMock(),
                max_turns=5,
                hooks=None,
                shared_context=None,
            )
        )

    assert result["success"] is True
    assert mock_sw.call_args.kwargs["attributes"] == EXPECTED_ATTRIBUTES
