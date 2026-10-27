"""Explicit opt-in cloud ranking. Never runs for exact search or by default."""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request


class RerankError(RuntimeError):
    pass


class CohereReranker:
    endpoint = "https://api.cohere.com/v2/rerank"

    def __init__(self, api_key: str, model="rerank-v5.0-pro", timeout=8):
        if not api_key or not model:
            raise ValueError("A Cohere API key and model are required")
        self._api_key = api_key
        self.model_id = model
        self.timeout = timeout

    def _request(self, payload):
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                raise RerankError("Cloud endpoint redirected; local results retained")

        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                        headers={"Content-Type": "application/json",
                                                 "Authorization": "Bearer " + self._api_key}, method="POST")
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise RerankError("Cloud response exceeded the limit")
                return json.loads(raw)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise RerankError("Cloud ranking unavailable; local results retained") from exc

    def rerank(self, query, documents):
        if not documents:
            return []
        if len(documents) > 30 or len(query) > 12000 or any(len(d) > 9000 for d in documents):
            raise ValueError("Cloud ranking accepts at most 30 bounded source windows")
        result = self._request({"model": self.model_id, "query": query, "documents": documents,
                                "top_n": len(documents), "max_tokens_per_doc": 3000})
        rows = result.get("results") if isinstance(result, dict) else None
        if not isinstance(rows, list) or len(rows) != len(documents):
            raise RerankError("Cloud ranking returned an incomplete candidate list")
        seen, ranked = set(), []
        for row in rows:
            index = row.get("index") if isinstance(row, dict) else None
            score = row.get("relevance_score") if isinstance(row, dict) else None
            if type(index) is not int or not 0 <= index < len(documents) or index in seen:
                raise RerankError("Cloud ranking returned invalid candidate identifiers")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise RerankError("Cloud ranking returned an invalid score")
            seen.add(index)
            ranked.append((index, float(score)))
        return ranked


def rerank_results(index, response, reranker):
    """Reorder source candidates while keeping their verified local evidence."""
    documents = []
    for hit in response["results"]:
        context = index.context(hit["conversation_id"], hit["message_id"], hit["source_id"], hit["account_id"])
        messages = context["messages"]
        position = next(i for i, message in enumerate(messages) if message["id"] == hit["message_id"])
        prefix = "\n".join(m["role"] + ": " + m["text"] for m in messages[max(0, position - 2):position])[-1800:]
        # Search supplies an original Unicode code-point offset for its best
        # semantic window. Literal evidence takes precedence when present.
        # Older results may omit the additive hint; retain their safe fallback.
        text = hit["text"]
        hint = hit.get("focus_offset", 0)
        focus = hint if type(hint) is int else 0
        matches = hit.get("matches")
        if isinstance(matches, list) and matches:
            first = matches[0]
            if (isinstance(first, (list, tuple)) and len(first) == 2
                    and type(first[0]) is int and type(first[1]) is int
                    and 0 <= first[0] < first[1] <= len(text)):
                focus = first[0]
        focus = max(0, min(len(text), focus))
        original = hit["text"][max(0, focus - 1000):focus + 5000]
        documents.append(f"Topic title: {hit['title'][:500]}\n{prefix}\n{hit['role']}: {original}")
    order = reranker.rerank(response["query"], documents)
    result = dict(response)
    result["results"] = [{**response["results"][i], "rerank_score": score} for i, score in order]
    return result
