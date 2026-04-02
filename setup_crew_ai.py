"""
CrewAI crew: researcher (Serper web search) + writer, sequential process.
Env: OPENAI_API_KEY, SERPER_API_KEY (.env — see .env_example).
Optional: USE_OLLAMA=1, OLLAMA_MODEL, OLLAMA_HOST, LLM_TEMPERATURE, OPENAI_MODEL,
CREW_RESEARCHER_MAX_ITER, CREW_WRITER_MAX_ITER (cap LLM/tool loops — default was 25 in CrewAI).
Uses crewai.llm.LLM (not LangChain chat models — required by CrewAI 1.x Agent validation).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


class CrewConfigurationError(RuntimeError):
    """Raised when required environment variables or configuration are missing."""


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise CrewConfigurationError(
            f"{name} is missing or empty. Copy .env_example to .env (local) "
            "or configure Streamlit / hosting secrets."
        )
    return value


def get_llm():
    """OpenAI by default; Ollama when USE_OLLAMA=1 (or true/yes)."""
    from crewai.llm import LLM

    use_ollama = os.getenv("USE_OLLAMA", "").lower() in ("1", "true", "yes")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    if use_ollama:
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        return LLM(model=model, provider="ollama", temperature=temperature)

    _require_env("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return LLM(model=model, temperature=temperature)


def build_agents(llm, search_tool, *, verbose: bool = True):
    from crewai import Agent

    # CrewAI default max_iter is 25 — each step can be LLM + Serper call, so runs easily exceed 5+ minutes.
    researcher_max_iter = _int_env("CREW_RESEARCHER_MAX_ITER", 7)
    writer_max_iter = _int_env("CREW_WRITER_MAX_ITER", 5)

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and data science",
        backstory="""You work at a leading tech think tank.
        Your expertise lies in identifying emerging trends.
        You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
        tools=[search_tool],
        max_iter=researcher_max_iter,
    )
    writer = Agent(
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
        You transform complex concepts into compelling narratives.""",
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
        max_iter=writer_max_iter,
    )
    return researcher, writer


def build_tasks(
    researcher,
    writer,
    *,
    research_focus: str,
    year_label: str,
    output_dir: Path,
):
    from crewai import Task

    output_dir.mkdir(parents=True, exist_ok=True)
    task_research = Task(
        description=f"""Summarize the latest advancements in {research_focus} ({year_label}).
        Be efficient: use the web search tool at most 2 times (broad queries, then refine only if needed).
        Focus on the highest-signal trends and impacts — do not aim for exhaustive coverage.""",
        expected_output="5–8 bullet points, one sentence each where possible",
        agent=researcher,
        output_file=str(output_dir / "researcher.txt"),
    )
    task_write = Task(
        description=f"""Using only the researcher's bullet list on {research_focus}, write a short blog post.
        No new web search. Accessible for a tech-savvy audience, natural tone, avoid hollow AI-speak.""",
        expected_output="3–4 paragraphs, roughly 80–150 words each",
        agent=writer,
        output_file=str(output_dir / "writer.txt"),
    )
    return task_research, task_write


def run_crew(
    *,
    research_focus: str = "artificial intelligence",
    year_label: str = "the current year",
    verbose: bool = True,
    output_dir: Path | None = None,
):
    """
    Build and run the researcher → writer crew.

    Returns:
        Final crew output (typically a string from the last task).

    Raises:
        CrewConfigurationError: If required API keys are not set.
    """
    from crewai import Crew, Process
    from crewai_tools import SerperDevTool

    _require_env("SERPER_API_KEY")

    out = output_dir if output_dir is not None else ROOT

    llm = get_llm()
    search_tool = SerperDevTool()
    researcher, writer = build_agents(llm, search_tool, verbose=verbose)
    task_research, task_write = build_tasks(
        researcher,
        writer,
        research_focus=research_focus,
        year_label=year_label,
        output_dir=out,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task_research, task_write],
        verbose=verbose,
        process=Process.sequential,
        share_crew=False,
    )
    return crew.kickoff()


def main() -> None:
    try:
        result = run_crew()
    except CrewConfigurationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("Final crew output")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
