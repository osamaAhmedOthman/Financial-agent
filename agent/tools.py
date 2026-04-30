"""
agent/tools.py

The two tools the LangGraph agent can call:
  1. search_legal_knowledge  — queries Pinecone hybrid search (Phase 2)
  2. search_web              — queries Tavily for live data (exchange rates, etc.)

These are plain Python functions. LangGraph calls them as node steps,
not as LangChain tool wrappers — keeping dependencies minimal.
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


def search_legal_knowledge(
    query: str,
    namespace: str = "egyptian-tax-law",
    top_k: int = 4,
) -> List[dict]:
    """
    Search the Pinecone knowledge base using hybrid vector + BM25 search.
    Returns the most relevant legal articles as plain dicts.

    Called by: legal_retriever_node in graph.py

    Args:
        query:     The legal question to answer (Arabic or English)
        namespace: 'egyptian-tax-law' or 'company-policies'
        top_k:     Number of results to return

    Returns:
        List of dicts with keys: article_number, law_name, law_code, text, score
    """
    try:
        from knowledge_base.retriever import retrieve
        results = retrieve(query=query, namespace=namespace, top_k=top_k)

        return [
            {
                "article_number": r.chunk.article_number or "N/A",
                "law_name":       r.chunk.law_name,
                "law_code":       r.chunk.law_code,
                "text":           r.chunk.text,
                "score":          round(r.score, 4),
                "source":         r.source,
            }
            for r in results
        ]
    except Exception as e:
        print(f"Legal search failed for query '{query[:40]}': {e}")
        return []


def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web using Tavily API for live data.
    Used ONLY when the agent needs something not in the knowledge base:
      - Current USD/EGP exchange rate
      - Recent tax law amendments
      - Current commodity prices

    Called by: web_search_node in graph.py (conditional — only when needed)

    Args:
        query:       Search query in English or Arabic
        max_results: Number of web results to include

    Returns:
        Concatenated search result snippets as a single string
    """
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not tavily_key:
        return "Web search unavailable: TAVILY_API_KEY not set."

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        snippets = [
            f"[{r.get('title', 'Result')}]: {r.get('content', '')}"
            for r in response.get("results", [])
        ]
        return "\n\n".join(snippets) if snippets else "No results found."

    except ImportError:
        return "Web search unavailable: install tavily-python (pip install tavily-python)."
    except Exception as e:
        return f"Web search failed: {e}"


def needs_web_search(document_type: str, currency: str, raw_text: str) -> bool:
    """
    Decides whether the agent needs to call the web search tool.

    Rules:
      - Always search if document has non-EGP currency (need exchange rate)
      - Search if document references recent dates (law may have changed)
      - Never search for standard EGP invoice auditing (laws are in Pinecone)

    Called by: the conditional edge in graph.py
    """
    # Non-EGP currency = definitely need live exchange rate
    if currency and currency.upper() != "EGP":
        return True

    # Keywords that suggest live data is needed
    live_data_keywords = [
        "دولار", "يورو", "dollar", "euro", "usd", "eur",
        "سعر الصرف", "exchange rate", "استيراد", "import",
    ]
    text_lower = raw_text.lower()
    return any(kw in text_lower for kw in live_data_keywords)
