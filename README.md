# Tech Researcher — AI agents

A small [CrewAI](https://www.crewai.com/) workflow: a **research analyst** (web search via [Serper](https://serper.dev/)) followed by a **tech writer** that turns findings into a blog-style draft. Run it from the **command line** or through a **Streamlit** web UI.

## What it does

1. **Research task** — Searches the web and produces a bullet-point analysis (saved as `researcher.txt` when using the CLI default output directory, or under a temp folder from Streamlit).
2. **Writing task** — Uses that analysis to generate an accessible article (writer output file).

The LLM is [CrewAI’s native `LLM` wrapper](https://docs.crewai.com/) (OpenAI by default, or Ollama when enabled).

## Requirements

- Python **3.10+** (3.11 recommended)
- **OpenAI** API key and **Serper** API key for the default cloud setup
- Optional: [Ollama](https://ollama.com/) for local models (works on your machine or self-hosted; **not** on Streamlit Community Cloud)

## Local setup

```bash
git clone https://github.com/davidanthony-ai/Tech-researcher-ai-agents.git
cd Tech-researcher-ai-agents

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env_example .env        # then edit .env with real keys
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal. For local Streamlit you can also copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) to `.streamlit/secrets.toml` (gitignored) instead of relying on `.env`.

### Run the CLI crew

```bash
python setup_crew_ai.py
```

Outputs are written next to the script by default (`researcher.txt`, `writer.txt` in the project root).

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (unless Ollama) | [OpenAI API key](https://platform.openai.com/api-keys) |
| `SERPER_API_KEY` | Yes | [Serper.dev](https://serper.dev/) key for Google search |
| `OPENAI_MODEL` | No | Default: `gpt-4o-mini` |
| `LLM_TEMPERATURE` | No | Default: `0.7` |
| `USE_OLLAMA` | No | Set to `1`, `true`, or `yes` to use Ollama |
| `OLLAMA_MODEL` | No | Default: `llama3.1` |
| `OLLAMA_HOST` | No | Ollama OpenAI-compatible base URL (see [CrewAI docs](https://docs.crewai.com/)) |

Never commit `.env` or `.streamlit/secrets.toml`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (without secrets).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app from the repo.
3. Set **Main file** to `app.py`.
4. Under **App settings → Secrets**, add at minimum:

   ```toml
   OPENAI_API_KEY = "sk-..."
   SERPER_API_KEY = "..."
   ```

5. Redeploy or reboot the app after changing secrets.

**Note:** Ollama is only useful where the app can reach your Ollama server; use OpenAI + Serper for the hosted Streamlit tier.

## Project layout

```
.
├── app.py              # Streamlit UI
├── setup_crew_ai.py    # Crew definition + CLI entrypoint
├── requirements.txt
├── .env_example
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```
