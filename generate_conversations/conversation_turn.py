"""Represents a single turn in a conversation with LangChain message and metadata."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llm_clients.llm_interface import Role


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation with LangChain message and metadata.

    This class wraps a LangChain BaseMessage (HumanMessage or AIMessage) with
    additional metadata needed for conversation tracking, such as turn number,
    speaker identification, input prompts, early termination flags, and logging
    information.

    Attributes:
        turn: Sequential turn number (1-indexed)
        speaker: Identifier for the speaker (e.g., "persona", "chatbot", "agent")
        input_message: The message that prompted this response
        message: The LangChain message object (HumanMessage or AIMessage)
        role: Role of the speaker (Role.PERSONA or Role.PROVIDER)
        early_termination: Whether this turn marked the end of conversation
        logging_metadata: Metadata from LLM provider (tokens, timing, etc.)
    """

    turn: int
    speaker: str
    input_message: str
    message: BaseMessage
    role: Optional[Role] = None
    early_termination: bool = False
    logging_metadata: Optional[Dict[str, Any]] = None

    @property
    def response(self) -> str:
        """Get the response text from the message.

        Returns:
            The content of the LangChain message
        """
        content = self.message.content
        # LangChain messages can have string or list content; we expect strings
        if isinstance(content, str):
            return content
        # Fallback for unexpected types (shouldn't happen in normal usage)
        return str(content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dict format for file export and backward compatibility.

        Returns:
            Dictionary with keys: turn, speaker, input, response,
            role, early_termination, logging
        """
        result = {
            "turn": self.turn,
            "speaker": self.speaker,
            "input": self.input_message,
            "response": self.response,
            "early_termination": self.early_termination,
            "logging": self.logging_metadata or {},
        }
        # Include role if present
        if self.role is not None:
            result["role"] = self.role.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from legacy dict format.

        Args:
            data: Dictionary with keys: turn, speaker, input, response, etc.

        Returns:
            ConversationTurn instance
        """
        # Determine message type based on speaker (for backward compatibility)
        speaker = data["speaker"]
        if speaker == "persona":
            message = HumanMessage(content=data["response"])
        else:  # chatbot/agent
            message = AIMessage(content=data["response"])

        # Extract role if present, otherwise infer from speaker (backward compatibility)
        role = None
        if "role" in data:
            try:
                role = Role(data["role"])
            except ValueError:
                # Fall back to inferring role from speaker if role string is invalid
                if speaker == "persona":
                    role = Role.PERSONA
                elif speaker in ("chatbot", "agent"):
                    role = Role.PROVIDER
       elif speaker == "persona":
           role = Role.PERSONA
       elif speaker in ("chatbot", "agent"):
           role = Role.PROVIDER

       return cls(
           turn=data["turn"],
           speaker=speaker,
           input_message=data["input"],
           message=message,
           role=role,
           early_termination=data.get("early_termination", False),
           logging_metadata=data.get("logging"),
       )
