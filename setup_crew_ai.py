"""
CrewAI crew: researcher (Serper web search) + writer, sequential process.
Env: OPENAI_API_KEY, SERPER_API_KEY (.env — see .env_example).
Optional: USE_OLLAMA=1, OLLAMA_MODEL, OLLAMA_HOST, LLM_TEMPERATURE, OPENAI_MODEL.
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
    )
    writer = Agent(
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
        You transform complex concepts into compelling narratives.""",
        verbose=verbose,
        allow_delegation=True,
        llm=llm,
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
        description=f"""Conduct a comprehensive analysis of the latest advancements in {research_focus}
        ({year_label}). Identify key trends, breakthrough technologies, and potential industry impacts.""",
        expected_output="Full analysis report in bullet points",
        agent=researcher,
        output_file=str(output_dir / "researcher.txt"),
    )
    task_write = Task(
        description=f"""Using the researcher's report on {research_focus}, write an engaging blog post highlighting
        the most significant developments. Keep it accessible for a tech-savvy audience, natural in tone,
        and avoid hollow jargon or an obviously AI-generated style.""",
        expected_output="Full blog post of at least 4 paragraphs",
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
