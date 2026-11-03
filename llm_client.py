"""
llm_client.py

Single traced entry point for every Groq call in the project.
Both the audit graph (agent/graph.py) and the extractor
(ingestion/llm_extractor.py) go through groq_chat(), so each LLM call shows up
in LangSmith as an "llm" run with its prompt, answer, token usage and
finish_reason.
"""
import os
from langsmith import traceable

_client = None


def _get_client():
    """Groq client, created once per process."""
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


@traceable(run_type="llm", name="groq_chat")
def groq_chat(**kwargs) -> dict:
    """
    Calls chat.completions.create(**kwargs) and returns the response as a plain
    OpenAI-shaped dict (so LangSmith can display it and read the token usage).
    """
    r = _get_client().chat.completions.create(**kwargs)
    choice = r.choices[0]
    u = getattr(r, "usage", None)
    return {
        "choices": [{
            "message": {"role": "assistant", "content": choice.message.content or ""},
            "finish_reason": choice.finish_reason,
        }],
        "model": getattr(r, "model", kwargs.get("model")),
        "usage": {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        },
    }
