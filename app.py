from __future__ import annotations

import os
import re

import streamlit as st

if not os.getenv("GOOGLE_API_KEY"):
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

from src import config, guardrails, rag  
from src.vectorstore import get_collection  

st.set_page_config(
    page_title="GitLab Handbook Assistant", page_icon="🦊", layout="centered"
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def linkify_citations(text: str, hits: list[dict]) -> str:
    """Turn inline [n] markers into clickable links to the cited source URL."""
    def repl(m: re.Match) -> str:
        idx = int(m.group(1))
        if 1 <= idx <= len(hits):
            return f"[\\[{idx}\\]]({hits[idx - 1]['url']})"
        return m.group(0)

    return _CITATION_RE.sub(repl, text)


def render_sources(hits: list[dict]) -> None:
    """Transparency panel: exactly what the model was allowed to read."""
    with st.expander(f"📚 Sources ({len(hits)} retrieved)"):
        for i, h in enumerate(hits, start=1):
            header = h["title"] + (f" — {h['heading']}" if h["heading"] else "")
            st.markdown(
                f"**[{i}] [{header}]({h['url']})**  \n"
                f"<span style='color:#6E49CB'>similarity {h['similarity']:.2f}</span>",
                unsafe_allow_html=True,
            )
            preview = h["text"].replace("\n", " ").strip()
            st.caption(preview[:280] + ("…" if len(preview) > 280 else ""))


@st.cache_resource(show_spinner=False)
def index_stats() -> dict:
    try:
        return {"ok": True, "count": get_collection().count()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "count": 0, "error": str(exc)}


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🦊 GitLab Handbook Assistant")
    st.caption(
        "Answers grounded in GitLab's public "
        "[Handbook](https://handbook.gitlab.com/) and "
        "[Direction](https://about.gitlab.com/direction/) pages."
    )
    stats = index_stats()
    st.divider()
    st.markdown("**Index**")
    if stats["ok"] and stats["count"] > 0:
        st.metric("Chunks indexed", f"{stats['count']:,}")
    else:
        st.warning("Index is empty. Run `python ingest.py` first.")
    st.markdown(f"**Chat model**  \n`{config.CHAT_MODEL}`")
    st.markdown(f"**Embeddings**  \n`{config.EMBED_MODEL}`")
    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Session state -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {role, content, hits?, note?}

st.title("GitLab Handbook Assistant 🦊")

typed = st.chat_input("Ask about GitLab's handbook, values, or direction…")

pending_question: str | None = None
if not st.session_state.messages and not typed:
    st.markdown("Ask me anything about how GitLab works. For example:")
    cols = st.columns(2)
    for i, q in enumerate(config.STARTER_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True):
            pending_question = q

# --- Replay history ----------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🦊" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"], unsafe_allow_html=False)
        if msg.get("note"):
            st.caption(msg["note"])
        if msg.get("hits"):
            render_sources(msg["hits"])

# --- Handle new input --------------------------------------------------------
question = typed or pending_question

if question:
    verdict = guardrails.check_input(question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🦊"):
        if not verdict.ok:
            st.markdown(verdict.message)
            st.session_state.messages.append(
                {"role": "assistant", "content": verdict.message}
            )
        else:
            history = st.session_state.messages[:-1]
            with st.spinner("Searching the handbook…"):
                result = rag.retrieve(question, history)

            note_bits = []
            if result.rewritten:
                note_bits.append(f"🔁 Interpreted as: *{result.question}*")
            if verdict.injection_flagged:
                note_bits.append("🛡️ Possible prompt-injection ignored; sticking to the handbook.")
            note = "  \n".join(note_bits)

            if not result.grounded:
                st.markdown(result.refusal)
                if note:
                    st.caption(note)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.refusal, "note": note}
                )
            else:
                placeholder = st.empty()
                acc = ""
                for piece in rag.stream_answer(result.question, result.hits, history):
                    acc += piece
                    placeholder.markdown(acc + "▌")
                final = linkify_citations(acc, result.hits)
                placeholder.markdown(final, unsafe_allow_html=False)
                if note:
                    st.caption(note)
                render_sources(result.hits)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final,
                        "hits": result.hits,
                        "note": note,
                    }
                )
    if pending_question and not typed:
        st.rerun()
