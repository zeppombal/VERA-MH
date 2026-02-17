import copy
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Role(Enum):
    """Role of the LLM in a conversation."""

    PERSONA = "persona"  # User roleplaying as a human seeking help
    PROVIDER = "provider"  # Chatbot providing support
    JUDGE = "judge"  # Judge role, used for judge operations


# Default prompt sent to the LLM when starting a conversation (trigger mode).
DEFAULT_TRIGGER_MESSAGE = "Start the conversation based on the system prompt"


class LLMInterface(ABC):
    """Abstract base class for LLM implementations.

    Provides basic text generation capabilities. All LLM implementations
    must support basic text generation and system prompt management.

    When conversation history is empty:
    - initial_message: If set, return this string as the first turn (no LLM call).
    - trigger_message: If initial_message is not set, use this as the prompt to
      the LLM to generate the first turn. Defaults to DEFAULT_TRIGGER_MESSAGE.
    """

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        initial_message: Optional[str] = None,
        trigger_message: Optional[str] = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt or ""
        self.initial_message = initial_message  # static first message (no LLM call)
        self.trigger_message = trigger_message  # prompt to LLM when history empty
        self._last_response_metadata: Dict[str, Any] = {}
        self.conversation_id = self.create_conversation_id()

    @property
    def last_response_metadata(self) -> Dict[str, Any]:
        """Metadata from the last generate_response call. Returns a deep copy so
        callers cannot mutate internal state (including nested dicts like usage).
        """
        return copy.deepcopy(self._last_response_metadata)

    @last_response_metadata.setter
    def last_response_metadata(self, value: Optional[Dict[str, Any]]) -> None:
        """Set metadata; use _last_response_metadata for in-place updates."""
        self._last_response_metadata = value or {}

    def create_conversation_id(self) -> str:
        """Create a new unique conversation id.

        Used at init and when the API does not return one in response metadata.
        Subclasses may override to use a different id format.
        """
        return str(uuid.uuid4())

    def _update_conversation_id_from_metadata(self) -> None:
        """If the API returned a conversation_id in response metadata, use it.

        Call after generate_response once _last_response_metadata is set.
        APIs that ignore our request conversation_id but return their own
        will overwrite self.conversation_id here.
        """
        cid = (self._last_response_metadata or {}).get("conversation_id")
        if cid is not None:
            self.conversation_id = cid

    def _set_response_metadata(self, provider: str, **extra: Any) -> None:
        """Set last_response_metadata with common fields; pass extra keys as kwargs.

        Always sets: response_id, model, provider, role, timestamp, usage.
        Override or add keys via kwargs (e.g. error=..., response_time_seconds=...).
        """
        self.last_response_metadata = {
            "response_id": extra.pop("response_id", None),
            "model": extra.pop("model", getattr(self, "model_name", None)),
            "provider": provider,
            "role": self.role.value,
            "timestamp": datetime.now().isoformat(),
            "usage": extra.pop("usage", {}),
            **extra,
        }

    def get_initial_trigger_turns(self) -> List[Dict[str, Any]]:
        """Build the initial turn(s) used to trigger the LLM when history is empty.

        Returns a list of dicts (e.g. [{"turn": 0, "response": "<trigger or
        DEFAULT_TRIGGER_MESSAGE>"}]) that can be passed to the message builder
        and then to the LLM. Used by raw LLM implementations that delegate from
        start_conversation() to generate_response(...). Subclasses may override.

        Returns:
            List of dicts representing the initial conversation turn(s)
            (e.g. [{"turn": 0, "response": "<trigger text>"}]).
        """
        trigger = (
            self.trigger_message
            if self.trigger_message is not None
            else DEFAULT_TRIGGER_MESSAGE
        )
        return [{"turn": 0, "response": trigger}]

    @abstractmethod
    async def start_conversation(self) -> str:
        """Produce the first response of the conversation.

        Called by the simulator on turn 0. When initial_message is set, return
        it (and set metadata) without calling the API. Otherwise, raw LLM
        implementations may call generate_response(self.get_initial_trigger_turns());
        service-based clients may call their own start endpoint (e.g. POST
        /start_conversation) and return the returned message.

        Returns:
            str: The first response text. Metadata in self.last_response_metadata
                (getter returns a copy so callers need not copy).
        """
        pass

    @abstractmethod
    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response based on conversation history.

        When used by the simulator, conversation_history is only passed for
        turns 1+ (turn 0 uses start_conversation()). Callers may still pass
        empty history for backward compatibility; implementations can delegate
        to await self.start_conversation() in that case if desired.

        Args:
            conversation_history: List of previous conversation turns.
                Each turn is a dict with 'turn' and 'response'. Format depends
                on whether the first entry is a trigger or a prior response:

                - **Turn 0 as trigger**: When history is built from
                  get_initial_trigger_turns(), the first entry has turn=0 and
                  'response' (trigger text). Speaker is not required: that
                  entry is input used to elicit the LLM's first response for
                  turn 1, not a prior utterance.
                - **Later turns**: Each turn must include 'turn', 'speaker',
                  and 'response' fields. The 'speaker' field is required for
                  correct LangChain message construction from conversation
                  history.
                  See llm_clients/claude_llm.py for an example.

        Returns:
            str: The response text. Metadata in self.last_response_metadata
                (getter returns a copy so callers need not copy).

        Note:
            For API thread/session identification, use self.conversation_id
            (set at init; send as request metadata). If your API returns a
            conversation_id in response metadata, call
            self._update_conversation_id_from_metadata() after setting
            _last_response_metadata to overwrite.
        """
        pass

    @abstractmethod
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        pass

    async def cleanup(self) -> None:
        """Clean up any resources used by this LLM instance.

        Subclasses that need cleanup (like AzureLLM) should override this method.
        Default implementation does nothing.
        """
        pass

    def __getattr__(self, name):
        """Delegate attribute access to the underlying llm object.

        This allows accessing attributes like temperature, max_tokens, etc.
        directly on the LLM instance, which will be forwarded to the
        underlying LangChain model (self.llm).
        """
        # Check if self.llm exists by looking in __dict__ to avoid recursion
        # Only delegate if self.llm exists and has the attribute
        if "llm" in self.__dict__ and hasattr(self.llm, name):
            return getattr(self.llm, name)
        # If the attribute doesn't exist on self.llm, raise AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class JudgeLLM(LLMInterface):
    """Extended LLM interface that supports structured output generation.

    This interface is required for LLM implementations that can be used
    as judges, where structured output (using Pydantic models) is necessary
    for reliable evaluation results.

    Implementations: Claude, OpenAI, Gemini, Azure
    Not supported by: Ollama
    """

    @abstractmethod
    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        """Generate a structured response using Pydantic model.

        Args:
            message: The prompt message
            response_model: Pydantic model class to structure the response

        Returns:
            Instance of the response_model with structured data

        Raises:
            RuntimeError: If structured output generation fails
        """
        pass
