import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

# Import the powerful Netra SDK and its decorators
from netra import Netra
from netra.decorators import agent, task, workflow

# Configure comprehensive logging for function tracing analysis
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stdout
)

logger = logging.getLogger(__name__)

load_dotenv()


# Example 1: Basic Task Functions
@task(name="process_data_task")  # type: ignore
def process_data(data: List[int]) -> Dict[str, Any]:
    """
    Process a list of numbers and return statistics.
    This function is decorated with @task to trace individual operations.
    """
    logger.info(f"Processing data: {data}")

    # Simulate some processing time
    time.sleep(0.1)

    result = {
        "count": len(data),
        "sum": sum(data),
        "average": sum(data) / len(data) if data else 0,
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }

    logger.info(f"Processing completed: {result}")
    return result


@task(name="validate_input_task")  # type: ignore
def validate_input(data: Any) -> bool:
    """
    Validate input data.
    Another task function for input validation.
    """
    logger.info("Validating input data")

    if not isinstance(data, list):
        logger.error("Input must be a list")
        return False

    if not all(isinstance(x, (int, float)) for x in data):
        logger.error("All items must be numbers")
        return False

    if len(data) == 0:
        logger.warning("Input list is empty")
        return False

    logger.info("Input validation passed")
    return True


# Example 2: Agent Functions (Decision Making)
@agent(name="decide_processing_strategy_agent")  # type: ignore
def decide_processing_strategy(data_size: int) -> str:
    """
    Decide which processing strategy to use based on data size.
    This function is decorated with @agent as it makes decisions.
    """
    logger.info(f"Deciding processing strategy for data size: {data_size}")

    if data_size < 10:
        strategy = "simple"
    elif data_size < 100:
        strategy = "optimized"
    else:
        strategy = "distributed"

    logger.info(f"Selected strategy: {strategy}")
    return strategy


@agent(name="analyze_results_agent")  # type: ignore
def analyze_results(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Analyze processing results and provide insights.
    Agent function for result analysis.
    """
    logger.info("Analyzing results")

    insights = {}

    if results.get("average", 0) > 50:
        insights["trend"] = "high_values"
    else:
        insights["trend"] = "low_values"

    if results.get("count", 0) > 10:
        insights["sample_size"] = "large"
    else:
        insights["sample_size"] = "small"

    range_val = (results.get("max", 0) or 0) - (results.get("min", 0) or 0)
    if range_val > 100:
        insights["variability"] = "high"
    else:
        insights["variability"] = "low"

    logger.info(f"Analysis insights: {insights}")
    return insights


# Example 3: Workflow Functions (Orchestration)
@workflow(name="data_processing_workflow")  # type: ignore[arg-type]
def data_processing_workflow(raw_data: List[int]) -> Dict[str, Any]:
    """
    Main workflow that orchestrates the entire data processing pipeline.
    This function is decorated with @workflow as it coordinates multiple operations.
    """
    logger.info("Starting data processing workflow")

    try:
        # Step 1: Validate input
        if not validate_input(raw_data):  # type: ignore[misc]
            raise ValueError("Input validation failed")

        # Step 2: Decide processing strategy
        strategy = decide_processing_strategy(len(raw_data))  # type: ignore[misc]

        # Step 3: Process the data
        results = process_data(raw_data)  # type: ignore[misc]

        # Step 4: Analyze results
        insights = analyze_results(results)  # type: ignore[misc]

        # Combine everything
        workflow_result = {"strategy": strategy, "results": results, "insights": insights, "status": "completed"}

        logger.info("Workflow completed successfully")
        return workflow_result

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return {"status": "failed", "error": str(e)}


# Example 4: Async Functions with Decorators
@task(name="async_data_fetch_task")  # type: ignore
async def async_data_fetch(source: str) -> List[int]:
    """
    Asynchronously fetch data from a source.
    Demonstrates async function decoration.
    """
    logger.info(f"Fetching data from: {source}")

    # Simulate async I/O operation
    await asyncio.sleep(0.2)

    # Mock data based on source
    if source == "database":
        data = [1, 2, 3, 4, 5]
    elif source == "api":
        data = [10, 20, 30, 40, 50]
    else:
        data = [100, 200, 300]

    logger.info(f"Fetched {len(data)} items from {source}")
    return data


@workflow(name="async_workflow")  # type: ignore
async def async_workflow(sources: List[str]) -> Dict[str, Any]:
    """
    Async workflow that processes data from multiple sources.
    Demonstrates async workflow decoration.
    """
    logger.info(f"Starting async workflow with sources: {sources}")

    try:
        # Fetch data from all sources concurrently
        tasks = [async_data_fetch(source) for source in sources]  # type: ignore[misc]
        all_data = await asyncio.gather(*tasks)

        # Combine all data
        combined_data = []
        for data_list in all_data:
            combined_data.extend(data_list)

        # Process combined data
        results = process_data(combined_data)  # type: ignore[misc]

        # Analyze results
        insights = analyze_results(results)  # type: ignore[misc]

        workflow_result = {
            "sources": sources,
            "total_items": len(combined_data),
            "results": results,
            "insights": insights,
            "status": "completed",
        }

        logger.info("Async workflow completed successfully")
        return workflow_result

    except Exception as e:
        logger.error(f"Async workflow failed: {e}")
        return {"status": "failed", "error": str(e)}


# Example 5: Custom Named Decorators
@task(name="advanced_metrics_calculator")  # type: ignore[arg-type]
def calculate_metrics(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate advanced metrics with custom span name.
    Demonstrates custom naming for spans.
    """
    logger.info("Calculating advanced metrics")

    if not numbers:
        return {}

    # Calculate various metrics
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std_dev = variance**0.5

    metrics = {
        "mean": mean,
        "variance": variance,
        "std_deviation": std_dev,
        "median": sorted(numbers)[len(numbers) // 2],
    }

    logger.info(f"Calculated metrics: {metrics}")
    return metrics


# Example 6: Error Handling in Decorated Functions
@task(name="risky_operation_task")  # type: ignore[arg-type]
def risky_operation(value: int) -> str:
    """
    Function that might raise an exception.
    Demonstrates error handling in decorated functions.
    """
    logger.info(f"Performing risky operation with value: {value}")

    if value < 0:
        raise ValueError("Value cannot be negative")
    elif value == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    elif value > 100:
        raise OverflowError("Value too large")

    result = f"Operation successful with value: {value}"
    logger.info(result)
    return result


@workflow(name="data_pipeline_orchestrator")  # type: ignore[arg-type]
def analyze_data(numbers: List[int]) -> Dict[str, Any]:
    """
    Analyze numerical data with comprehensive statistics

    Performs statistical analysis on numerical datasets including
    descriptive statistics, distribution analysis, and data quality metrics.

    Args:
        numbers: List of integers to analyze

    Returns:
        Dictionary containing comprehensive analysis results
    """
    logger.info(f"Analyzing data: {numbers}")

    result = {
        "count": len(numbers),
        "sum": sum(numbers),
        "average": sum(numbers) / len(numbers) if numbers else 0,
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
    }

    logger.info(f"Analysis completed: {result}")
    return result


@agent(name="data_validator")  # type: ignore[arg-type]
def validate_data(data: Any) -> bool:
    """
    Validate data integrity and quality

    Performs comprehensive data validation including type checking,
    range validation, and data quality assessment.

    Args:
        data: Data to validate (any type)

    Returns:
        Boolean indicating if data passes validation
    """
    logger.info("Validating data")

    if not isinstance(data, list):
        logger.error("Input must be a list")
        return False

    if not all(isinstance(x, (int, float)) for x in data):
        logger.error("All items must be numbers")
        return False

    if len(data) == 0:
        logger.warning("Input list is empty")
        return False

    logger.info("Validation passed")
    return True


@task(name="data_analyzer")  # type: ignore[arg-type]
def transform_data(value: int) -> str:
    """
    Transform data between different formats

    Demonstrates data transformation capabilities with type conversion,
    formatting, and data structure manipulation.

    Args:
        value: Integer value to transform

    Returns:
        Transformed string representation
    """
    logger.info(f"Transforming data: {value}")

    result = str(value)

    logger.info(f"Transformation completed: {result}")
    return result


@agent(name="data_enricher")  # type: ignore[arg-type]
def enrich_data(raw_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Enrich data with additional context and metadata

    Adds contextual information, metadata, and derived attributes
    to enhance the value and usability of raw data.

    Args:
        raw_data: Dictionary containing raw data to enrich

    Returns:
        Dictionary with enriched data and metadata
    """
    logger.info("Enriching data")

    enriched_data = {
        "context": "additional context",
        "metadata": "metadata information",
        "derived_attributes": "derived attributes",
    }

    logger.info(f"Enrichment completed: {enriched_data}")
    return enriched_data


@workflow(name="comprehensive_data_pipeline")  # type: ignore[arg-type]
def process_data_pipeline(data: List[int]) -> Dict[str, Any]:
    """
    Execute comprehensive data processing pipeline

    Orchestrates multiple data processing steps including analysis,
    validation, transformation, and enrichment in a cohesive workflow.

    Args:
        data: List of integers to process through the pipeline

    Returns:
        Dictionary containing complete pipeline results
    """
    logger.info("Starting data processing pipeline")

    try:
        # Step 1: Analyze data
        analysis_result = analyze_data(data)  # type: ignore[misc]

        # Step 2: Validate data
        if not validate_data(data):  # type: ignore[misc]
            raise ValueError("Data validation failed")

        # Step 3: Transform data
        transformed_data = transform_data(data[0])  # type: ignore[misc]

        # Step 4: Enrich data
        enriched_data = enrich_data({"raw_data": data})  # type: ignore[misc]

        # Combine everything
        pipeline_result = {
            "analysis": analysis_result,
            "validation": "passed",
            "transformation": transformed_data,
            "enrichment": enriched_data,
            "status": "completed",
        }

        logger.info("Pipeline completed successfully")
        return pipeline_result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return {"status": "failed", "error": str(e)}


@workflow(name="async_data_processor")  # type: ignore[arg-type]
async def process_async_data(query: str) -> List[int]:
    """
    Process data asynchronously with high performance

    Demonstrates asynchronous data processing capabilities for
    high-throughput scenarios and concurrent operations.

    Args:
        query: Query string to process

    Returns:
        List of processed integer results
    """
    logger.info(f"Processing async data: {query}")

    try:
        # Simulate async I/O operation
        await asyncio.sleep(0.2)

        # Mock data based on query
        data = [1, 2, 3, 4, 5]

        logger.info(f"Async processing completed: {data}")
        return data

    except Exception as e:
        logger.error(f"Async processing failed: {e}")
        return []


@task(name="async_data_aggregator")  # type: ignore[arg-type]
async def aggregate_async_data(datasets: List[str]) -> Dict[str, Any]:
    """
    Aggregate data from multiple sources asynchronously

    Combines data from multiple sources with parallel processing
    for efficient aggregation and analysis.

    Args:
        datasets: List of dataset identifiers to aggregate

    Returns:
        Dictionary containing aggregated results and metadata
    """
    logger.info(f"Aggregating async data: {datasets}")

    try:
        # Simulate async I/O operation
        await asyncio.sleep(0.2)

        # Mock data based on datasets
        aggregated_data = {"aggregated": "data"}

        logger.info(f"Async aggregation completed: {aggregated_data}")
        return aggregated_data

    except Exception as e:
        logger.error(f"Async aggregation failed: {e}")
        return {}


@workflow(name="statistical_analysis_workflow")  # type: ignore[arg-type]
def perform_statistical_analysis(values: List[float]) -> Dict[str, float]:
    """
    Perform advanced statistical analysis

    Conducts comprehensive statistical analysis including descriptive
    statistics, correlation analysis, and distribution fitting.

    Args:
        values: List of float values for statistical analysis

    Returns:
        Dictionary containing statistical analysis results
    """
    logger.info(f"Performing statistical analysis: {values}")

    result = {
        "mean": sum(values) / len(values) if values else 0,
        "variance": sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values) if values else 0,
        "std_deviation": (
            (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5 if values else 0
        ),
    }

    logger.info(f"Statistical analysis completed: {result}")
    return result


@task(name="data_quality_assessment")  # type: ignore[arg-type]
def assess_data_quality(sample_size: int) -> str:
    """
    Assess data quality and completeness

    Evaluates data quality metrics including completeness, accuracy,
    consistency, and reliability for data governance purposes.

    Args:
        sample_size: Size of the data sample to assess

    Returns:
        String containing data quality assessment report
    """
    logger.info(f"Assessing data quality: {sample_size}")

    result = "Data quality assessment report"

    logger.info(f"Data quality assessment completed: {result}")
    return result


def demonstrate_sync_functions() -> None:
    """Demonstrate synchronous decorated functions."""
    print("\n🔄 Demonstrating Synchronous Functions")
    print("-" * 50)

    # Test data
    test_data = [1, 5, 10, 15, 20, 25, 30]

    # Run the workflow
    result = data_processing_workflow(test_data)  # type: ignore[misc]
    print(f"Workflow result: {result}")

    # Test custom named function
    metrics = calculate_metrics([1.5, 2.3, 3.7, 4.1, 5.9])  # type: ignore[misc]
    print(f"Custom metrics: {metrics}")

    # Test error handling
    for test_value in [5, -1, 0, 150]:
        try:
            operation_result: str = risky_operation(test_value)  # type: ignore[misc]
            print(f"Risky operation result: {operation_result}")
        except Exception as e:
            print(f"Risky operation failed with value {test_value}: {e}")


async def demonstrate_async_functions() -> None:
    """Demonstrate asynchronous decorated functions."""
    print("\n⚡ Demonstrating Asynchronous Functions")
    print("-" * 50)

    # Test async workflow
    sources = ["database", "api", "cache"]
    result = await async_workflow(sources)  # type: ignore[misc]
    print(f"Async workflow result: {result}")


def main() -> None:
    """
    Main function demonstrating Netra SDK function decorators.
    """
    # Initialize Netra SDK
    try:
        Netra.init(
            app_name="function-decorators-example",
            environment="development",
            trace_content=True,
            headers=f"x-api-key={os.getenv('NETRA_API_KEY', 'demo-key')}",
        )
        logger.info("✅ Netra SDK initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Netra SDK: {e}")
        return

    # Set user context
    Netra.set_user_id("decorator_demo_user")
    Netra.set_session_id("decorator_demo_session")

    print("=" * 70)
    print("🎯 Netra SDK Function Decorators Example")
    print("=" * 70)

    # Demonstrate synchronous functions
    demonstrate_sync_functions()

    # Demonstrate asynchronous functions
    asyncio.run(demonstrate_async_functions())

    print("\n" + "=" * 70)
    print("🎉 Function decorators example completed!")


if __name__ == "__main__":
    main()
