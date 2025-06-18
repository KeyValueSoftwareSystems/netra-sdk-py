"""
Netra SDK Integrated Example

This example demonstrates a realistic usage scenario that integrates multiple
Netra SDK features in a simulated chatbot application. It shows:
- Session management
- User context tracking
- Prompt injection detection
- PII detection and handling
- Error handling and logging
- Realistic response generation
"""

import logging
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# Import the Netra SDK
from netra import Netra
from netra.exceptions.injection import InjectionException
from netra.input_scanner import InputScanner, ScannerType
from netra.pii import get_default_detector


class ChatResponse(TypedDict):
    status: str
    message: str
    metadata: Dict[str, Any]


class PIIViolation(TypedDict):
    entity_type: str
    start: int
    end: int
    score: float


# Type for PII detection result
class PIIDetectionResult:
    def __init__(
        self,
        has_pii: bool = False,
        pii_entities: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        masked_text: str = "",
        action_type: str = "",
    ):
        self.has_pii = has_pii
        self.pii_entities = pii_entities or {}
        self.masked_text = masked_text
        self.action_type = action_type


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Sample responses for the chatbot
RESPONSES = [
    "I understand you're asking about {topic}. Here's what I can share: {info}",
    "Regarding {topic}, I can tell you that {info}",
    "That's an interesting question about {topic}. {info}",
]

# Sample information for different topics
TOPIC_INFO = {
    "machine learning": "Machine learning is a branch of AI that enables systems to learn from data. "
    "It's widely used in various applications like recommendation systems, "
    "image recognition, and natural language processing.",
    "artificial intelligence": "AI is the simulation of human intelligence in machines. "
    "It's transforming industries from healthcare to finance.",
    "data science": "Data science combines statistics, programming, and domain expertise "
    "to extract insights from data.",
    "cybersecurity": "Cybersecurity focuses on protecting systems and data from digital attacks. "
    "It's crucial in our increasingly connected world.",
    "cloud computing": "Cloud computing delivers computing services over the internet, "
    "enabling flexible resources and economies of scale.",
}


def get_response(topic: str, user_message: str) -> str:
    """Generate a contextual response based on the topic and user message."""
    info = TOPIC_INFO.get(
        topic.lower(), "I'm still learning about that topic. " "Could you tell me more about what interests you?"
    )
    response_template = random.choice(RESPONSES)
    return response_template.format(topic=topic, info=info)


class ExampleChatbot:
    """Example chatbot application using Netra SDK for monitoring and security."""

    def __init__(self) -> None:
        """Initialize the chatbot with Netra SDK integration."""
        try:
            # Initialize the Netra SDK with detailed configuration
            Netra.init(app_name="example-chatbot", trace_content=True, environment="demo")  # Enable content tracing
            logger.info("Netra SDK initialized successfully")

            # Initialize security scanners with specific configurations
            self.scanner = InputScanner(scanner_types=[ScannerType.PROMPT_INJECTION])

            # Get the default PII detector with custom configuration
            self.pii_detector = get_default_detector()
            logger.info("Security scanners initialized")

            # Track conversation history and metadata
            self.conversation_history: List[Dict[str, str]] = []
            self.conversation_metadata: Dict[str, Any] = {}

        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {str(e)}")
            raise

    def start_session(self, user_id: Optional[str] = None, session_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new chat session with enhanced context.

        Args:
            user_id: Optional user ID. If not provided, one will be generated.
            session_metadata: Optional dictionary of metadata to attach to the session.

        Returns:
            str: The session ID for this chat session.
        """
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())

            # Set basic context using available Netra SDK methods
            Netra.set_session_id(session_id)
            user_id = user_id or f"anon-{uuid.uuid4()}"
            Netra.set_user_id(user_id)
            Netra.set_tenant_id("demo-tenant")

            # Store metadata as conversation context
            self.conversation_metadata = session_metadata or {}
            logger.info(f"Started session {session_id} for user {user_id}")
            logger.debug(f"Session metadata: {self.conversation_metadata}")

            return session_id

        except Exception as e:
            logger.error(f"Failed to start session: {str(e)}")
            raise

    def _detect_topic(self, message: str) -> Tuple[str, float]:
        """Detect the main topic from the user's message.

        In a real application, this would use NLP to detect topics.
        For this example, we'll use simple keyword matching.
        """
        message_lower = message.lower()
        for topic in TOPIC_INFO:
            if topic in message_lower:
                return topic, 0.9  # Arbitrary confidence score

        # If no specific topic found, return a default
        return "general", 0.5

    def _handle_pii(self, message: str) -> Tuple[str, Dict[str, int]]:
        """Handle PII detection and redaction.

        Args:
            message: The input message to scan for PII.

        Returns:
            Tuple of (processed_message, pii_detections)
        """
        try:
            pii_result = self.pii_detector.detect(message)

            if not pii_result.has_pii:
                return message, {}

            # Log PII detections
            pii_entities = pii_result.pii_entities
            pii_types = list(pii_entities.keys()) if pii_entities else []
            logger.warning(f"PII detected: {', '.join(pii_types)}")
            return message, pii_entities

        except Exception as e:
            logger.warning(f"Error during PII scanning: {str(e)}")
            return message, {}

    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message through the security pipeline.

        Args:
            message: The user's message to process.

        Returns:
            Dict: The response from the chatbot or an error message.
        """
        try:
            # Record message received
            logger.info(f"Processing message: {message}")

            # Store message in conversation history
            self.conversation_history.append({"role": "user", "content": message})

            # Step 1: Scan for prompt injections
            logger.info("Scanning for prompt injections")
            try:
                scan_result = self.scanner.scan(message)
                if (
                    hasattr(scan_result, "has_violation")
                    and scan_result.has_violation
                    and hasattr(scan_result, "violations")
                ):
                    violations = getattr(scan_result, "violations", [])
                    print(f"✗ Injection detected: {', '.join(violations)}")
                    return {
                        "status": "error",
                        "error_type": "injection",
                        "message": "Your message was flagged as potentially unsafe. " "Please rephrase and try again.",
                        "violations": scan_result.violations,
                    }
            except InjectionException as e:
                logger.warning(f"Injection blocked: {str(e)}")
                return {
                    "status": "error",
                    "error_type": "injection",
                    "message": "Your message contained potentially unsafe content and was blocked.",
                    "violations": getattr(e, "violations", ["unknown"]),
                }

            # Step 2: Scan for PII
            logger.info("Scanning for PII")
            processed_message, pii_detections = self._handle_pii(message)

            # Check for PII
            pii_result = self.pii_detector.detect(message)
            if pii_result.has_pii:
                pii_types = list(pii_result.pii_entities.keys())
                logger.warning(f"PII detected: {', '.join(pii_types)}")
                return {
                    "status": "error",
                    "error_type": "pii_detected",
                    "message": "Your message contains sensitive information that cannot be processed.",
                    "metadata": {
                        "pii_detected": True,
                        "pii_types": pii_types,
                    },
                }

            # Step 3: Detect topic for response generation
            topic, confidence = self._detect_topic(processed_message)
            logger.info(f"Detected topic: {topic} (confidence: {confidence:.2f})")

            # Step 4: Generate response
            logger.info("Generating response")
            time.sleep(0.5)  # Simulate processing time

            # Generate a contextual response
            bot_response = get_response(topic, processed_message)

            # Store bot response in conversation history
            self.conversation_history.append({"role": "assistant", "content": bot_response})

            # Log successful response
            logger.info(f"Generated response (length: {len(bot_response)})")

            # Return the response with metadata
            return {
                "status": "success",
                "message": bot_response,
                "metadata": {
                    "topic": topic,
                    "confidence": confidence,
                    "pii_detected": bool(pii_detections),
                    "pii_types": list(pii_detections.keys()) if pii_detections else [],
                },
            }

        except Exception as e:
            # Log the full error with stack trace
            logger.error(f"Unexpected error processing message: {str(e)}", exc_info=True)

            return {
                "status": "error",
                "error_type": "processing",
                "message": "Sorry, I encountered an error while processing your message. "
                "Our team has been notified. Please try again in a moment.",
                "error": str(e),
            }


def run_chatbot() -> None:
    """Run an interactive chat session with the example chatbot."""
    try:
        print("\n" + "=" * 60)
        print("Netra SDK Example Chatbot".center(60))
        print("Type 'exit' or 'quit' to end the session".center(60))
        print("=" * 60 + "\n")

        # Initialize the chatbot
        print("Initializing chatbot...")
        chatbot: ExampleChatbot = ExampleChatbot()

        # Start a new session with metadata
        session_metadata = {"client_type": "cli", "client_version": "1.0", "interaction_type": "chat"}
        user_id = f"user-{uuid.uuid4().hex[:8]}"
        session_id = chatbot.start_session(user_id=user_id, session_metadata=session_metadata)

        print(f"\nSession started (User: {user_id}, Session: {session_id})")
        print("I'm your AI assistant. Ask me anything or type 'help' for options.\n")

        # Interactive chat loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Check for exit commands
                if user_input.lower() in ("exit", "quit"):
                    print("\nEnding chat session. Goodbye!")
                    break

                if user_input.lower() == "help":
                    print("\nAvailable commands:")
                    print("- Type any question to chat")
                    print("- Try asking about: machine learning, AI, data science")
                    print("- Type 'exit' or 'quit' to end the session\n")
                    continue

                if not user_input:
                    continue

                # Process the message
                response = chatbot.process_message(user_input)

                # Display the response or error
                if response["status"] == "success":
                    print(f"\nAssistant: {response['message']}")
                    if "metadata" in response and response["metadata"].get("pii_detected"):
                        pii_types = ", ".join(response["metadata"]["pii_types"])
                        print(f"[Note: Detected and handled PII: {pii_types}]")
                else:
                    print(f"\n[Error: {response['message']}]")

                print()  # Add spacing between exchanges

            except KeyboardInterrupt:
                print("\n\nEnding chat session...")
                break
            except Exception as e:
                print(f"\n[System Error: {str(e)}]")
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)

        print("\nChat session ended. Thank you for using the Netra SDK example!")

    except Exception as e:
        logger.critical(f"Fatal error in chatbot: {str(e)}", exc_info=True)
        print(f"\nA critical error occurred: {str(e)}")
        print("Please check the logs for more details.")


def main() -> None:
    """Main entry point for the example."""
    try:
        run_chatbot()
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
