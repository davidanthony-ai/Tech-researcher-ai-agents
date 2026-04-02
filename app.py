"""
Streamlit UI for the CrewAI researcher + writer workflow.
Deploy on Streamlit Community Cloud: connect the GitHub repo and set secrets (see .env_example keys).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Secrets: Streamlit Cloud exposes st.secrets; local dev can use .streamlit/secrets.toml (gitignored)
# ---------------------------------------------------------------------------
_SECRET_KEYS = (
    "OPENAI_API_KEY",
    "SERPER_API_KEY",
    "OPENAI_MODEL",
    "LLM_TEMPERATURE",
    "USE_OLLAMA",
    "OLLAMA_MODEL",
    "OLLAMA_HOST",
)


def apply_streamlit_secrets_to_environ() -> None:
    """Copy known keys from st.secrets into os.environ so CrewAI and tools read them."""
    try:
        secrets = st.secrets
    except (FileNotFoundError, RuntimeError):
        return
    for key in _SECRET_KEYS:
        if key in secrets and str(secrets[key]).strip():
            os.environ[key] = str(secrets[key]).strip()


apply_streamlit_secrets_to_environ()

from setup_crew_ai import CrewConfigurationError, run_crew  # noqa: E402


def main() -> None:
    st.set_page_config(
        page_title="CrewAI Research & Writer",
        page_icon="🧭",
        layout="wide",
    )

    st.title("CrewAI — Research analyst + Tech writer")
    st.caption(
        "Sequential crew: web research (Serper) with OpenAI or Ollama, then blog-style writing."
    )

    with st.sidebar:
        st.subheader("Configuration")
        st.markdown(
            "Set **OPENAI_API_KEY** and **SERPER_API_KEY** in "
            "[Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) "
            "or in `.env` for local runs."
        )
        use_ollama = st.checkbox(
            "Use Ollama (local)",
            value=os.getenv("USE_OLLAMA", "").lower() in ("1", "true", "yes"),
            help="USE_OLLAMA=1. Only works where Ollama is reachable (not Streamlit Community Cloud).",
        )
        if use_ollama:
            os.environ["USE_OLLAMA"] = "1"
        else:
            os.environ.pop("USE_OLLAMA", None)

    col1, col2 = st.columns(2)
    with col1:
        research_focus = st.text_input(
            "Research topic",
            value="artificial intelligence",
            help="Domain or theme for the analyst to investigate.",
        )
    with col2:
        year_label = st.text_input(
            "Time frame label",
            value="2026",
            help='e.g. "2026", "Q1 2026", or "the current year".',
        )

    run = st.button("Run crew", type="primary", use_container_width=True)

    if not run:
        return

    apply_streamlit_secrets_to_environ()
    if use_ollama:
        os.environ["USE_OLLAMA"] = "1"

    # Isolated output directory per run (Streamlit Cloud filesystem is ephemeral)
    run_dir = Path(tempfile.gettempdir()) / "crewai_streamlit" / str(uuid.uuid4())
    run_dir.mkdir(parents=True, exist_ok=True)

    with st.status("Running crew (research → write)…", expanded=True) as status:
        try:
            result = run_crew(
                research_focus=research_focus.strip() or "artificial intelligence",
                year_label=year_label.strip() or "the current year",
                verbose=False,
                output_dir=run_dir,
            )
        except CrewConfigurationError as err:
            status.update(label="Configuration error", state="error")
            st.error(str(err))
            return
        except Exception as err:  # noqa: BLE001 — show any runtime failure in the UI
            status.update(label="Run failed", state="error")
            st.exception(err)
            return
        status.update(label="Done", state="complete")

    st.subheader("Final output")
    st.markdown(str(result))

    st.subheader("Saved task files (this run)")
    for name in ("researcher.txt", "writer.txt"):
        path = run_dir / name
        if path.is_file():
            with st.expander(name, expanded=(name == "writer.txt")):
                st.code(path.read_text(encoding="utf-8", errors="replace"), language="markdown")


if __name__ == "__main__":
    main()
