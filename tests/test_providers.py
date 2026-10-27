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
            self.assertEqual(body["keep_alive"], 0)
            self.assertEqual(body["options"], {"num_ctx": 1024, "num_batch": 64})
            self.assertEqual(body["input"], ["hello", "中文"])
        with self.assertRaises(ValueError):
            model.embed(["x" * 12001])

    def test_embedding_960_byte_guard_zero_http(self):
        model = OllamaProvider()
        self.assertEqual(model.max_input_bytes, 960)
        with patch.object(model, "_request") as mock_request:
            # Under limit: 960 ASCII bytes and Chinese / emoji bytes
            under_ascii = "a" * 960
            mock_request.return_value = {"embeddings": [[1.0, 0.0]]}
            model.embed([under_ascii])
            self.assertEqual(mock_request.call_count, 1)

            # Chinese / emoji under 960 bytes: 320 Chinese chars * 3 bytes = 960 bytes
            under_chinese = "中" * 320
            self.assertEqual(len(under_chinese.encode("utf-8")), 960)
            model.embed([under_chinese])
            self.assertEqual(mock_request.call_count, 2)

            # Over limit: 961 ASCII bytes
            mock_request.reset_mock()
            over_ascii = "a" * 961
            with self.assertRaises(ProviderError) as ctx:
                model.embed([over_ascii])
            self.assertIn("shorter Smart description", str(ctx.exception))
            mock_request.assert_not_called()

            # Over limit: Chinese / emoji exceeding 960 bytes
            over_chinese_emoji = "中" * 320 + "😀"
            with self.assertRaises(ProviderError) as ctx:
                model.embed([over_chinese_emoji])
            self.assertIn("shorter Smart description", str(ctx.exception))
            mock_request.assert_not_called()

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


    def test_provider_minimum_similarity_configuration(self):
        # Default qwen3-embedding:0.6b is calibrated to 0.5
        qwen_model = OllamaProvider()
        self.assertEqual(qwen_model.minimum_similarity, 0.5)

        # Other models default to 0.3
        other_model = OllamaProvider(embedding_model="nomic-embed-text")
        self.assertEqual(other_model.minimum_similarity, 0.3)


if __name__ == "__main__":
    unittest.main()
