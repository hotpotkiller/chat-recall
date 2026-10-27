import io
import json
import math
import unittest
from unittest.mock import patch

from chat_recall.providers import OllamaProvider, ProviderError, normalized_vector


class ProvidersTests(unittest.TestCase):
    def test_finite_nonzero_vector_contract(self):
        self.assertEqual(normalized_vector([3, 4]), [0.6, 0.8])
        for bad in ([], [0, 0], [True, 2], [float("nan"), 1], [math.inf], ["1"], None):
            with self.assertRaises(ProviderError):
                normalized_vector(bad)

    def test_provider_is_local_only_and_constructor_is_offline(self):
        with patch("urllib.request.build_opener") as opener:
            model = OllamaProvider()
            self.assertEqual(model.model_id, "ollama:qwen3-embedding:0.6b")
            opener.assert_not_called()
        for bad in ("https://example.org", "http://example.org", "http://127.0.0.1/path",
                    "http://user:password@127.0.0.1", "http://localhost?x=1"):
            with self.assertRaises(ValueError):
                OllamaProvider(base_url=bad)

    def test_embedding_request_is_bounded_and_never_truncates_silently(self):
        model = OllamaProvider()
        with patch.object(model, "_request", return_value={"embeddings": [[3, 4], [0, 5]]}) as request:
            self.assertEqual(model.embed(["hello", "中文"]), [[0.6, 0.8], [0.0, 1.0]])
            path, body = request.call_args.args
            self.assertEqual(path, "/api/embed")
            self.assertFalse(body["truncate"])
            self.assertEqual(body["input"], ["hello", "中文"])
        with self.assertRaises(ValueError):
            model.embed(["x" * 12001])

    def test_rejects_partial_or_inconsistent_embeddings(self):
        model = OllamaProvider()
        for response in ({"embeddings": []}, {"embeddings": [[1, 2], [3, 4, 5]]},
                         {"embeddings": [[0, 0], [1, 2]]}):
            with patch.object(model, "_request", return_value=response):
                with self.assertRaises(ProviderError):
                    model.embed(["one", "two"])

    def test_topic_json_and_memory_release(self):
        model = OllamaProvider()
        with patch.object(model, "_request", return_value={"message": {"content": '{"topic":"Reducing soup acidity"}'}}) as request:
            self.assertEqual(model.summarize("A synthetic dialogue"), "Reducing soup acidity")
            self.assertEqual(request.call_args.args[1]["keep_alive"], 0)
            self.assertFalse(request.call_args.args[1]["think"])
        with patch.object(model, "_request", return_value={"message": {"content": "not json"}}):
            with self.assertRaises(ProviderError):
                model.summarize("A synthetic dialogue")

    def test_http_response_limit_and_no_network_in_test(self):
        model = OllamaProvider()
        fake_body = io.BytesIO(json.dumps({"embeddings": [[1, 0]]}).encode())
        with patch("urllib.request.build_opener") as factory:
            factory.return_value.open.return_value = fake_body
            result = model.embed(["hello"])
            self.assertEqual(result, [[1.0, 0.0]])
            request = factory.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/embed")


if __name__ == "__main__":
    unittest.main()
