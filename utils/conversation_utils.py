"""Utilities for conversation management and file operations."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llm_clients.llm_interface import Role

from .debug import debug_print


def generate_conversation_filename(prefix: str = "conversation") -> str:
    """
    Generate a timestamped filename for conversation logs.

    Args:
        prefix: Prefix for the filename

    Returns:
        Formatted filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.txt"


def save_conversation_to_file(
    conversation_history: List[Dict[str, Any]],
    filename: str,
    folder: str,
    llm1_name: Optional[str] = None,
) -> None:
    """
    Save conversation history to a text file.

    Args:
        conversation_history: List of conversation turns
        filename: Output filename
        folder: Output folder
        llm1_name: Name of LLM1 to identify it as 'user'
    """
    # Ensure folder exists
    Path(folder).mkdir(parents=True, exist_ok=True)

    summary = format_conversation_summary(conversation_history, llm1_name)
    with open(f"{folder}/{filename}", "w", encoding="utf-8") as f:
        f.write(summary)


def format_conversation_summary(
    conversation_history: List[Dict[str, Any]], llm1_name: Optional[str] = None
) -> str:
    """
    Format conversation history into a readable summary.

    Args:
        conversation_history: List of conversation turns
        llm1_name: Name of LLM1 to identify it as 'user'

    Returns:
        Formatted conversation summary
    """
    if not conversation_history:
        return "No conversation recorded."

    summary = ""

    for i, turn in enumerate(conversation_history):
        speaker_name = turn["speaker"]

        # Determine if this is LLM1 (user) or LLM2 (agent)
        if llm1_name and llm1_name in speaker_name:
            prefix = "user:"
        else:
            prefix = "chatbot:"

        # Add the response with proper formatting
        summary += f"{prefix} {turn['response']}\n"

        # Check for early termination flag
        if turn.get("early_termination", False):
            summary += f"[CONVERSATION ENDED - {speaker_name} signaled termination]\n"

        # Add empty line between speakers, but not after the last turn
        if i < len(conversation_history) - 1:
            summary += "\n"

    return summary


def build_persona_role_reminder() -> str:
    """
    Build a role reminder message for personas to prevent role confusion.
    This reminder is injected before conversation history to reinforce that
    the persona should maintain their character as a user in distress, not
    adopt the chatbot's supportive counselor role.
    Returns:
        The role reminder text to inject as a HumanMessage
    """
    return (
        "REMINDER: You are the USER seeking help from the chatbot. "
        "The conversation below shows YOUR messages (what you said) and "
        "the CHATBOT's responses (what it said). Continue responding AS "
        "YOURSELF - the user in distress. Do NOT provide advice, support, "
        "or counseling. Do NOT copy the chatbot's supportive language."
    )


def build_langchain_messages(
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    role: Optional[Role] = None,
) -> List[BaseMessage]:
    """
    Build a list of LangChain messages from conversation history for the given role.

    Args:
        conversation_history: Optional list of previous conversation turns.
            Each turn is a dict with keys: 'turn', 'speaker', 'response', 'role'.
            Turn 0 contains the initial message with speaker="system".
        role: Optional role of the LLM (Role.PERSONA, Role.PROVIDER, or None).
            Messages from the same role are converted to AIMessage (what "I"
            said), messages from the other role are converted to HumanMessage
            (what "they" said).

    Returns:
        List of LangChain message objects (HumanMessage, AIMessage)
    """
    messages = []

    # Add conversation history if provided
    if conversation_history:
        hist_len = len(conversation_history)
        debug_print(f"[DEBUG] Processing {hist_len} turns from history:")
        for turn in conversation_history:
            turn_number = turn.get("turn")
            text = turn.get("response")
            message_role = turn.get("role")

            # Skip turns without turn number or response
            if turn_number is None or text is None:
                continue

            # Handle turn 0 (initial message) specially
            if turn_number == 0:
                debug_print(f"  Turn 0 -> HumanMessage (initial): {text[:50]}...")
                messages.append(HumanMessage(content=text))
                continue

            # Determine message type based on role comparison
            # If we have role information, use it directly
            if message_role is not None and role is not None:
                try:
                    turn_role = (
                        Role(message_role)
                        if isinstance(message_role, str)
                        else message_role
                    )
                    # If the message's role matches the requesting role, it's an
                    # AIMessage (what "I" said). Otherwise, it's a HumanMessage
                    # (what "they" said).
                    if turn_role == role:
                        message = AIMessage(content=text)
                    else:
                        message = HumanMessage(content=text)
                except (ValueError, TypeError):
                    # Invalid role value, fall back to backward compatibility
                    debug_print(
                        f"  Warning: Invalid role value '{message_role}', "
                        "using backward compatibility"
                    )
                    message = _determine_message_type_fallback(turn_number, text, role)
            else:
                # Backward compatibility: use turn number logic if role is not available
                message = _determine_message_type_fallback(turn_number, text, role)

            msg_type = type(message).__name__
            preview = text[:50] + "..." if len(text) > 50 else text
            debug_print(f"  Turn {turn_number} -> {msg_type}: {preview}")
            messages.append(message)

    return messages


def _determine_message_type_fallback(
    turn_number: int, text: str, role: Optional[Role]
) -> BaseMessage:
    """
    Fallback method to determine message type using turn number logic.

    This is used for backward compatibility when role information is not
    available.

    Args:
        turn_number: The turn number (1-indexed)
        text: The message content
        role: The role of the requesting LLM

    Returns:
        AIMessage or HumanMessage
    """
    # Persona speaks on odd turns (1, 3, 5...), provider on even (2, 4, 6...)
    is_persona_turn = turn_number % 2 == 1

    # Flip message types when LLM is playing persona role
    if role == Role.PERSONA:
        # Persona responses are AIMessage, provider inputs are HumanMessage
        return (
            AIMessage(content=text) if is_persona_turn else HumanMessage(content=text)
        )
    else:
        # Persona inputs are HumanMessage, provider responses are AIMessage
        return (
            HumanMessage(content=text) if is_persona_turn else AIMessage(content=text)
        )


def format_conversation_as_string(
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    role: Optional[Role] = None,
) -> str:
    """
    Format conversation history as a string for string-based LLMs (e.g., Ollama).

    This function reuses build_langchain_messages() and converts the result to
    a string format with Human/Assistant labels.

    Args:
        conversation_history: Optional list of previous conversation turns
        system_prompt: Optional system prompt to prepend
        role: Optional role of the LLM (Role.PERSONA, Role.PROVIDER, or None)

    Returns:
        Formatted string with System, Human, and Assistant labels
    """
    full_message = ""

    # Add system prompt if provided
    if system_prompt:
        full_message = f"System: {system_prompt}\n\n"

    # Build LangChain messages using existing utility
    messages = build_langchain_messages(conversation_history, role)

    # Convert messages to string format
    for message in messages:
        if isinstance(message, HumanMessage):
            full_message += f"Human: {message.content}\n\n"
        elif isinstance(message, AIMessage):
            full_message += f"Assistant: {message.content}\n\n"

    # Add "Assistant:" prompt at the end to signal the LLM should respond
    if messages and full_message.endswith("\n\n"):
        full_message += "Assistant:"

    return full_message
