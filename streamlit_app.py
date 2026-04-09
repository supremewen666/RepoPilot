"""Streamlit entry point for RepoPilot."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Iterator

from repopilot.agent.runner import invoke_agent
from repopilot.memory.store import get_relevant_memories, save_memory_if_needed
from repopilot.schemas import FinalResponse

try:
    import streamlit as st
except ImportError:  # pragma: no cover - exercised only without optional deps.
    st = None


def init_session_state() -> None:
    """
    Initialize Streamlit session state for a new chat thread.

    Why:
        Keep UI state management separate from agent logic so the app can
        rerender safely without duplicating conversation setup.

    Stores:
        - thread_id: unique conversation id used by LangChain short-term memory
        - user_id: stable user identity used by mem0
        - messages: UI-only chat history for rendering
    """

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "user_id" not in st.session_state:
        st.session_state.user_id = "local-demo-user"
    if "messages" not in st.session_state:
        st.session_state.messages = []


def handle_user_query(user_query: str, *, user_id: str, thread_id: str) -> FinalResponse:
    """
    Orchestrate one end-to-end assistant turn from the UI layer.

    Flow:
        1. Load relevant long-term memory for this user
        2. Invoke the main LangChain agent
        3. Persist any memory worth saving
        4. Return a structured response for rendering

    Notes:
        This function does not implement retrieval or GitHub access itself.
        It is the bridge between Streamlit events and backend orchestration.
    """

    memory_context = get_relevant_memories(user_id=user_id, query=user_query)
    response = invoke_agent(
        user_query=user_query,
        user_id=user_id,
        thread_id=thread_id,
        memory_context=memory_context,
    )
    save_memory_if_needed(
        user_id=user_id,
        user_query=user_query,
        assistant_answer=response.answer,
    )
    return response


def _answer_chunks(text: str, *, delay: float = 0.01) -> Iterator[str]:
    """Yield small text chunks so Streamlit can render the answer progressively."""

    words = text.split()
    if not words:
        return
    for index, word in enumerate(words):
        suffix = " " if index < len(words) - 1 else ""
        yield f"{word}{suffix}"
        time.sleep(delay)


def stream_user_query(user_query: str, *, user_id: str, thread_id: str, status_placeholder: object) -> FinalResponse:
    """Run the backend in a thread and keep the UI responsive with live status updates."""

    result_queue: queue.Queue[FinalResponse | Exception] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result_queue.put(handle_user_query(user_query, user_id=user_id, thread_id=thread_id))
        except Exception as exc:  # pragma: no cover - defensive UI fallback.
            result_queue.put(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    steps = [
        "Inspecting local documentation index",
        "Preparing retrieval queries",
        "Gathering grounded evidence",
        "Composing the final response",
    ]
    tick = 0
    while thread.is_alive() and result_queue.empty():
        step = steps[min(tick // 8, len(steps) - 1)]
        dots = "." * (tick % 4)
        status_placeholder.markdown(f"_{step}{dots}_")
        time.sleep(0.15)
        tick += 1

    outcome = result_queue.get()
    status_placeholder.empty()
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


def render_response(response: FinalResponse) -> None:
    """
    Render answer text, citations, and memory usage in a predictable UI layout.

    Why:
        The assistant must always expose sources, so citations are rendered as a
        first-class section rather than hidden inside free-form text.
    """

    st.markdown(response.answer)
    st.caption(f"Confidence: {response.confidence}")

    if response.citations:
        st.markdown("### Sources")
        for citation in response.citations:
            st.markdown(f"- **{citation.label or citation.source_type}**")
            if citation.url_or_path:
                st.code(citation.url_or_path)
            if citation.snippet:
                st.write(citation.snippet)

    if response.used_memory:
        st.markdown("### Memory Used")
        for item in response.used_memory:
            st.write(f"- {item}")


def main() -> None:
    """Run the Streamlit application if the dependency is installed."""

    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements.txt to run the UI.")

    st.set_page_config(page_title="RepoPilot", page_icon=":bookmark_tabs:", layout="wide")
    st.title("RepoPilot")
    st.caption("Single-repository engineering assistant with RAG, GitHub MCP, and memory.")
    init_session_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_response(FinalResponse.model_validate(message["content"]))
            else:
                st.markdown(message["content"])

    user_query = st.chat_input("Ask about docs, issues, PRs, or repository context")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response = stream_user_query(
            user_query,
            user_id=st.session_state.user_id,
            thread_id=st.session_state.thread_id,
            status_placeholder=status_placeholder,
        )
        streamed_answer = st.write_stream(_answer_chunks(response.answer))
        response.answer = str(streamed_answer or response.answer)
        st.caption(f"Confidence: {response.confidence}")

        if response.citations:
            st.markdown("### Sources")
            for citation in response.citations:
                st.markdown(f"- **{citation.label or citation.source_type}**")
                if citation.url_or_path:
                    st.code(citation.url_or_path)
                if citation.snippet:
                    st.write(citation.snippet)

        if response.used_memory:
            st.markdown("### Memory Used")
            for item in response.used_memory:
                st.write(f"- {item}")

    st.session_state.messages.append({"role": "assistant", "content": response.model_dump()})


if __name__ == "__main__":
    main()
