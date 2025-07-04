import logging
import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

# Import the Netra SDK and decorators
from netra import Netra
from netra.decorators import agent, task, workflow

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stdout
)

logger = logging.getLogger(__name__)

load_dotenv()


# Example 1: Task Class (All methods traced)
@workflow(name="data_processing_workflow")  # type: ignore[arg-type]
class DataProcessor:
    """
    Data processing class with all methods automatically traced.
    The @workflow decorator instruments all public methods.
    """

    def __init__(self, name: str) -> None:
        """Initialize the data processor."""
        self.name = name
        self.processed_count = 0
        logger.info(f"DataProcessor '{name}' initialized")

    def clean_data(self, data: List[Any]) -> List[Any]:
        """Clean and filter the input data."""
        logger.info(f"Cleaning data with {len(data)} items")

        # Remove None values and convert to numbers
        cleaned = []
        for item in data:
            if item is not None:
                try:
                    cleaned.append(float(item))
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid item: {item}")

        logger.info(f"Cleaned data: {len(cleaned)} valid items")
        return cleaned

    def transform_data(self, data: List[float]) -> List[float]:
        """Transform the data by applying mathematical operations."""
        logger.info(f"Transforming {len(data)} items")

        # Apply transformation (square root of absolute value)
        transformed = [abs(x) ** 0.5 for x in data]

        self.processed_count += len(transformed)
        logger.info(f"Transformation completed. Total processed: {self.processed_count}")
        return transformed

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {"processor_name": self.name, "total_processed": self.processed_count, "status": "active"}


# Example 2: Agent Class (Decision Making)
@agent(name="data_analysis_agent")  # type: ignore[arg-type]
class DataAnalyzer:
    """
    Data analysis class that makes decisions about data processing.
    The @agent decorator instruments all public methods.
    """

    def __init__(self) -> None:
        """Initialize the analyzer with default settings."""
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("DataAnalyzer initialized")

    def analyze_distribution(self, data: List[float]) -> Dict[str, Any]:
        """Analyze the distribution of data."""
        logger.info(f"Analyzing distribution of {len(data)} items")

        if not data:
            return {"error": "No data to analyze"}

        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)

        analysis = {
            "mean": mean,
            "variance": variance,
            "std_dev": variance**0.5,
            "min": min(data),
            "max": max(data),
            "count": len(data),
        }

        self.analysis_history.append(analysis)
        logger.info(f"Distribution analysis completed: mean={mean:.2f}")
        return analysis

    def recommend_action(self, analysis: Dict[str, Any]) -> str:
        """Recommend an action based on analysis results."""
        logger.info("Determining recommended action")

        mean = analysis.get("mean", 0)
        std_dev = analysis.get("std_dev", 0)

        if mean > 10.0:
            if std_dev > mean * 0.5:
                recommendation = "high_variance_processing"
            else:
                recommendation = "standard_high_processing"
        else:
            if std_dev > mean * 0.8:
                recommendation = "stabilization_needed"
            else:
                recommendation = "standard_low_processing"

        logger.info(f"Recommended action: {recommendation}")
        return recommendation

    def get_history_summary(self) -> Dict[str, Any]:
        """Get a summary of analysis history."""
        return {"total_analyses": len(self.analysis_history)}


# Example 3: Workflow Class (Orchestration)
@task(name="data_pipeline_orchestrator")  # type: ignore[arg-type]
class DataPipeline:
    """
    Data pipeline class that orchestrates the entire processing workflow.
    The @task decorator instruments all public methods.
    """

    def __init__(self, pipeline_name: str) -> None:
        """Initialize the data pipeline."""
        self.pipeline_name = pipeline_name
        self.processor = DataProcessor(f"{pipeline_name}_processor")
        self.analyzer = DataAnalyzer()
        self.execution_count = 0
        logger.info(f"DataPipeline '{pipeline_name}' initialized")

    def execute_pipeline(self, raw_data: List[Any]) -> Dict[str, Any]:
        """Execute the complete data processing pipeline."""
        logger.info(f"Executing pipeline '{self.pipeline_name}' with {len(raw_data)} items")

        try:
            # Step 1: Clean the data
            cleaned_data = self.processor.clean_data(raw_data)

            if not cleaned_data:
                return {"status": "failed", "error": "No valid data after cleaning"}

            # Step 2: Transform the data
            transformed_data = self.processor.transform_data(cleaned_data)

            # Step 3: Analyze the data
            analysis = self.analyzer.analyze_distribution(transformed_data)

            # Step 4: Get recommendation
            recommendation = self.analyzer.recommend_action(analysis)

            # Step 5: Compile results
            self.execution_count += 1

            result = {
                "pipeline_name": self.pipeline_name,
                "execution_count": self.execution_count,
                "input_size": len(raw_data),
                "processed_size": len(transformed_data),
                "analysis": analysis,
                "recommendation": recommendation,
                "processor_stats": self.processor.get_statistics(),
                "analyzer_stats": self.analyzer.get_history_summary(),
                "status": "completed",
            }

            logger.info(f"Pipeline execution completed successfully")
            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {"status": "failed", "error": str(e), "pipeline_name": self.pipeline_name}

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline."""
        return {
            "pipeline_name": self.pipeline_name,
            "executions": self.execution_count,
            "processor_status": self.processor.get_statistics(),
            "analyzer_status": self.analyzer.get_history_summary(),
        }


# Example 4: Mixed Decoration Strategy
class MixedDecorationExample:
    """
    Class demonstrating mixed decoration strategies.
    Some methods are decorated individually, others are not.
    """

    def __init__(self) -> None:
        """Initialize mixed decoration example."""
        logger.info("MixedDecorationExample initialized")

    @workflow(name="text_processing_workflow")  # type: ignore[arg-type]
    def process_text(self, text: str) -> str:
        """
        Process text with workflow tracking

        Applies comprehensive text processing including normalization,
        validation, and transformation with full workflow visibility.

        Args:
            text: Input text to process

        Returns:
            Processed and transformed text
        """

        logger.info(f"Processing text: {text}")
        time.sleep(0.1)  # Simulate work
        result = f"Processed: {text.upper()}"
        logger.info(f"Text processing completed: {result}")
        return result

    def utility_function(self, value: int) -> int:
        """Utility function that doesn't need tracing."""
        return value * 2

    @agent(name="text_analysis_agent")  # type: ignore[arg-type]
    def analyze_text(self, words: List[str]) -> str:
        """
        Analyze text with intelligent agent capabilities

        Performs advanced text analysis including sentiment analysis,
        entity recognition, and linguistic pattern detection.

        Args:
            words: List of words to analyze

        Returns:
            Analysis results and insights
        """

        logger.info(f"Analyzing text with {len(words)} words")

        # Simple analysis logic
        if "positive" in words:
            result = "Positive sentiment detected"
        elif "negative" in words:
            result = "Negative sentiment detected"
        else:
            result = "Neutral sentiment detected"

        logger.info(f"Text analysis completed: {result}")
        return result

    @task(name="text_summarization_task")  # type: ignore[arg-type]
    def summarize_text(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Summarize text content with task tracking

        Creates intelligent summaries with key point extraction,
        relevance scoring, and content optimization.

        Args:
            sentences: List of sentences to summarize

        Returns:
            Dictionary containing summary and metadata
        """

        logger.info(f"Summarizing text with {len(sentences)} sentences")

        # Simple summarization logic
        summary = " ".join([sentence.split(".")[0] for sentence in sentences])

        result = {
            "summary": summary,
            "metadata": {"sentence_count": len(sentences), "word_count": len(summary.split())},
        }

        logger.info(f"Text summarization completed: {result}")
        return result

    def orchestrate_operations(self, input_data: List[str]) -> Dict[str, Any]:
        """Orchestrate multiple operations."""
        logger.info("Orchestrating operations")

        results = []
        for data in input_data:
            # Use important operation
            processed = self.process_text(data)  # type: ignore[misc]
            results.append(processed)

        # Make a decision
        options = ["good", "better", "optimal"]
        decision = self.analyze_text(options)  # type: ignore[misc]

        return {"processed_items": results, "decision": decision, "total_processed": len(results)}


def demonstrate_class_decorators() -> None:
    """Demonstrate various class decoration patterns."""
    print("\n🏗️  Demonstrating Class Decorators")
    print("-" * 50)

    # Test data
    test_data = [1, 2, "3", 4.5, None, "invalid", 6, 7.8, 9]

    # Create and test data pipeline
    pipeline = DataPipeline("demo_pipeline")
    result = pipeline.execute_pipeline(test_data)
    print(f"Pipeline result: {result['status']}")
    print(f"Processed {result.get('processed_size', 0)} items")

    # Test mixed decoration
    mixed_example = MixedDecorationExample()
    mixed_result = mixed_example.orchestrate_operations(["hello", "world", "test"])
    print(f"Mixed decoration result: {mixed_result}")


def main() -> None:
    """
    Main function demonstrating Netra SDK class decorators.
    """
    # Initialize Netra SDK
    try:
        Netra.init(
            app_name="class-decorators-example",
            environment="development",
            trace_content=True,
            headers=f"x-api-key={os.getenv('NETRA_API_KEY', 'demo-key')}",
        )
        logger.info("✅ Netra SDK initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Netra SDK: {e}")
        return

    # Set user context
    Netra.set_user_id("class_decorator_demo_user")
    Netra.set_session_id("class_decorator_demo_session")

    print("=" * 70)
    print("🎯 Netra SDK Class Decorators Example")
    print("=" * 70)

    # Demonstrate class decorators
    demonstrate_class_decorators()

    print("\n" + "=" * 70)
    print("🎉 Class decorators example completed!")


if __name__ == "__main__":
    main()
