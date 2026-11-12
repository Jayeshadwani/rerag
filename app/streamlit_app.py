"""The chat page of NoteRecall (proposal 009). No business logic: it shows what the API returns.

The page keeps the last five turns in Streamlit's session state (one browser session; nothing is
stored on a server), sends them with each question, and shows each answer claim by claim with its
`[C1]` tags and the cited sources as expandable cards. It reads the visitor's IP from the first
`X-Forwarded-For` entry of the request headers (`st.context.headers`) and sends it to the API in
`X-UI-Client-IP` together with the shared secret `UI_API_SECRET`, so the API's daily limit counts
the visitor and not this service. Settings come from the environment: `API_URL`, `UI_API_SECRET`.
"""

import os
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

HISTORY_TURNS = 5
TITLE = "ReRAG"
DESCRIPTION = (
    "Ask questions about my ML course notes: Jupyter notebooks covering data analysis, classical "
    "ML, deep learning and NLP. Every answer cites the notebook cells it comes from; if the notes "
    "don't cover something, it says so."
)
LIMITS_LINE = "Answers come only from these notes, not general knowledge. 15 questions per day."
TOPICS_LABEL = "What's in the notes"
# Each was checked against the real index: the chunks that answer it rank within the 6 chunks the
# model is given (see the PR description for the ranks).
EXAMPLES = (
    "What does -1 mean in reshape?",
    "Why does bagging reduce variance?",
    "How do I choose k in KMeans with the elbow method?",
)
TOPICS_TIMEOUT_S = (3, 5)  # the page must not hang while a sleeping service wakes up
TOPICS_UNAVAILABLE = (
    "The list could not be loaded (the service may be waking up). Reload the page in a minute."
)
CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 120  # a sleeping free instance needs about a minute to wake up
WAKING = (
    "The service is waking up (free instances sleep when idle). "
    "This takes about a minute; ask again shortly."
)
LIMIT = "Daily limit reached. Questions are limited per day; try again after the reset."
NOT_IN_NOTES = "This is not in your notes."
UNAVAILABLE = "The service is unavailable right now. Try again in a moment."


@dataclass(frozen=True)
class Reply:
    """What the API call came to: `kind` is ok, limit, waking or error."""

    kind: str
    body: dict[str, Any]
    message: str = ""


def client_ip() -> str | None:
    """The visitor's IP: first `X-Forwarded-For` entry, else the socket address Streamlit saw."""
    forwarded = st.context.headers.get("X-Forwarded-For")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return st.context.ip_address


def history_payload(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The last turns in the shape of the API: question and the answer text that was shown."""
    return [{"question": t["question"], "answer": t["text"]} for t in turns[-HISTORY_TURNS:]]


def ask_api(question: str, turns: list[dict[str, Any]], ip: str | None) -> Reply:
    """One call to `POST /ask`."""
    headers = {"X-UI-Secret": os.environ["UI_API_SECRET"]}
    if ip:
        headers["X-UI-Client-IP"] = ip
    payload: dict[str, Any] = {"question": question, "history": history_payload(turns)}
    try:
        response = requests.post(
            os.environ["API_URL"].rstrip("/") + "/ask",
            json=payload,
            headers=headers,
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        )
    except (requests.ConnectionError, requests.Timeout):
        return Reply("waking", {}, WAKING)
    try:
        body = response.json()
    except ValueError:
        # An HTML page from the platform's proxy: the service behind it is not up yet.
        return Reply("waking", {}, WAKING)
    if response.status_code == 429:
        return Reply("limit", body, LIMIT)
    if response.status_code != 200:
        return Reply("error", body, f"{UNAVAILABLE} ({body.get('detail', response.status_code)})")
    return Reply("ok", body)


def fetch_topics() -> dict[str, Any] | None:
    """`GET /topics`: the notebook titles and their count, or None if the API did not answer."""
    try:
        response = requests.get(
            os.environ["API_URL"].rstrip("/") + "/topics", timeout=TOPICS_TIMEOUT_S
        )
        body = response.json()
    except (requests.ConnectionError, requests.Timeout, ValueError):
        return None
    if response.status_code != 200 or not isinstance(body.get("notebooks"), list):
        return None
    topics: dict[str, Any] = body
    return topics


def show_topics() -> None:
    """The "What's in the notes" section: notebook titles from the API, loaded once per session."""
    if "topics" not in st.session_state:
        topics = fetch_topics()
        if topics is not None:
            st.session_state["topics"] = topics
    topics = st.session_state.get("topics")
    label = TOPICS_LABEL if topics is None else f"{TOPICS_LABEL} ({topics['count']} notebooks)"
    with st.expander(label):
        if topics is None:
            st.info(TOPICS_UNAVAILABLE)
        else:
            st.text("\n".join(topics["notebooks"]))


def pick_example() -> None:
    """Callback of the example chips: the chosen example becomes the next question."""
    st.session_state["pending_question"] = st.session_state["example"]
    st.session_state["example"] = None  # so the same chip can be chosen again


def show_examples() -> None:
    st.pills("Try asking", EXAMPLES, key="example", on_change=pick_example)


def answer_text(body: dict[str, Any]) -> str:
    """The answer as plain text, kept in the history that goes back to the API."""
    if body["status"] == "refused":
        return NOT_IN_NOTES
    return " ".join(
        f"{c['text']} {''.join(f'[{i}]' for i in c['citations'])}" for c in body["claims"]
    )


def show_answer(body: dict[str, Any]) -> None:
    """Claims with tags, then source cards, or the refusal message."""
    if body["status"] == "refused":
        st.info(NOT_IN_NOTES)
        return
    for claim in body["claims"]:
        tags = " ".join(f"**[{i}]**" for i in claim["citations"])
        st.markdown(f"{claim['text']} {tags}")
    for source in body["sources"]:
        where = " > ".join([source["notebook"], *source["heading_path"]])
        with st.expander(f"[{source['id']}] {where} (cell {source['cell_index']})"):
            st.text(source["text"])


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="📓")
    st.title(TITLE)
    st.markdown(DESCRIPTION)
    st.caption(LIMITS_LINE)
    missing = [n for n in ("API_URL", "UI_API_SECRET") if not os.environ.get(n)]
    if missing:
        st.error(f"Not configured: set {', '.join(missing)}.")
        st.stop()
    show_topics()
    turns: list[dict[str, Any]] = st.session_state.setdefault("turns", [])
    if not turns:
        show_examples()
    for turn in turns[-HISTORY_TURNS:]:
        st.chat_message("user").write(turn["question"])
        with st.chat_message("assistant"):
            show_answer(turn["body"])
    typed = st.chat_input("Ask about your notes")
    question = st.session_state.pop("pending_question", None) or typed
    if not question:
        return
    st.chat_message("user").write(question)
    with st.chat_message("assistant"), st.spinner("Searching the notes..."):
        reply = ask_api(question, turns, client_ip())
    if reply.kind != "ok":
        (st.warning if reply.kind in ("limit", "waking") else st.error)(reply.message)
        return
    turns.append({"question": question, "body": reply.body, "text": answer_text(reply.body)})
    del turns[:-HISTORY_TURNS]
    st.rerun()


if __name__ == "__main__":
    main()
