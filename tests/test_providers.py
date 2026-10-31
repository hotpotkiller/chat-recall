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

    def test_query_formatting_and_byte_cap(self):
        model = OllamaProvider()
        prefix = model.QWEN_QUERY_PREFIX
        self.assertIn("Instruct: Given a description of a past conversation", prefix)
        self.assertIn("\nQuery: ", prefix)

        # embed_query formats query with prefix
        with patch.object(model, "_request") as mock_req:
            mock_req.return_value = {"embeddings": [[0.6, 0.8]]}
            model.embed_query("test query")
            mock_req.assert_called_once()
            path, body = mock_req.call_args.args[:2]
            self.assertEqual(path, "/api/embed")
            self.assertEqual(body["input"], [f"{prefix}test query"])

        # embed_documents and raw embed do NOT prepend prefix
        with patch.object(model, "_request") as mock_req:
            mock_req.return_value = {"embeddings": [[0.6, 0.8]]}
            model.embed_documents(["doc text"])
            self.assertEqual(mock_req.call_args.args[1]["input"], ["doc text"])

        with patch.object(model, "_request") as mock_req:
            mock_req.return_value = {"embeddings": [[0.6, 0.8]]}
            model.embed(["raw doc text"])
            self.assertEqual(mock_req.call_args.args[1]["input"], ["raw doc text"])

        # Final byte cap applied AFTER prefix in embed_query
        prefix_bytes_len = len(prefix.encode("utf-8"))
        allowed_query_len = 960 - prefix_bytes_len
        valid_query = "a" * allowed_query_len
        with patch.object(model, "_request") as mock_req:
            mock_req.return_value = {"embeddings": [[0.6, 0.8]]}
            model.embed_query(valid_query)
            mock_req.assert_called_once()

        overbudget_query = "a" * (allowed_query_len + 1)
        with patch.object(model, "_request") as mock_req:
            with self.assertRaises(ProviderError) as ctx:
                model.embed_query(overbudget_query)
            self.assertIn("shorter Smart description", str(ctx.exception))
            mock_req.assert_not_called()

    def test_resolve_identity_metadata_only_and_fingerprint(self):
        model = OllamaProvider()
        self.assertIsNone(model.cached_identity)
        tags_response = {
            "models": [
                {"name": "qwen3-embedding:0.6b", "digest": "sha256:abc1234567890"}
            ]
        }
        with patch.object(model, "_request", return_value=tags_response) as mock_req:
            index_id = model.resolve_identity({"version": "v1", "max_chunk_bytes": 512})
            self.assertTrue(index_id.startswith("ollama:qwen3-embedding:0.6b:"))
            self.assertEqual(model.cached_identity, index_id)
            self.assertEqual(model.index_id, index_id)
            self.assertEqual(mock_req.call_args.args[0], "/api/tags")
            self.assertEqual(mock_req.call_args.kwargs.get("timeout"), 5.0)

    def test_resolve_identity_model_matching(self):
        # Configured with tag (:0.6b) must not match :latest or other tags
        model = OllamaProvider(embedding_model="qwen3-embedding:0.6b")
        mismatched_tags = {
            "models": [
                {"name": "qwen3-embedding:latest", "digest": "sha256:latestdigest"},
                {"name": "qwen3-embedding:1.5b", "digest": "sha256:15bdigest"}
            ]
        }
        with patch.object(model, "_request", return_value=mismatched_tags):
            with self.assertRaises(ProviderError):
                model.resolve_identity({"version": "v1"})

        # Configured without tag matches :latest when listed
        bare_model = OllamaProvider(embedding_model="nomic-embed-text")
        latest_tags = {
            "models": [
                {"name": "nomic-embed-text:latest", "digest": "sha256:nomicdigest"}
            ]
        }
        with patch.object(bare_model, "_request", return_value=latest_tags):
            id_res = bare_model.resolve_identity({"version": "v1"})
            self.assertTrue(id_res.startswith("ollama:nomic-embed-text:"))

    def test_revision_and_profile_mismatch_exclusion(self):
        model = OllamaProvider()
        tags1 = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:digest111"}]}
        tags2 = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:digest222"}]}

        profile1 = {"version": "v1", "max_chunk_bytes": 512}
        profile2 = {"version": "v1", "max_chunk_bytes": 256}

        with patch.object(model, "_request", return_value=tags1):
            id_p1 = model.resolve_identity(profile1)

        with patch.object(model, "_request", return_value=tags1):
            id_p2 = model.resolve_identity(profile2)

        with patch.object(model, "_request", return_value=tags2):
            id_digest2 = model.resolve_identity(profile1)

        # Profile change changes identity
        self.assertNotEqual(id_p1, id_p2)
        # Digest change changes identity
        self.assertNotEqual(id_p1, id_digest2)

    def test_query_only_template_reuse_of_document_identity(self):
        model = OllamaProvider()
        tags = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:stable_digest"}]}
        profile = {"version": "v1", "max_chunk_bytes": 512}

        with patch.object(model, "_request", return_value=tags):
            id1 = model.resolve_identity(profile)

        # Mutate query_template_version and calibration_version
        model.query_template_version = "qwen-instruct-v2"
        model.calibration_version = "calibrated-v2"

        with patch.object(model, "_request", return_value=tags):
            id2 = model.resolve_identity(profile)

        # Changing only query template / calibration must not change document index identity
        self.assertEqual(id1, id2)

    def test_tag_change_during_inference_rejects_vectors(self):
        model = OllamaProvider()
        initial_tags = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:initial"}]}
        mutated_tags = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:mutated"}]}

        # Resolve identity with initial digest
        with patch.object(model, "_request", return_value=initial_tags):
            model.resolve_identity({"version": "v1"})

        # Case 1: Tag changes before embedding request
        with patch.object(model, "_request", side_effect=[mutated_tags]):
            with self.assertRaises(ProviderError) as ctx:
                model.embed_query("test query")
            self.assertIn("Model digest changed", str(ctx.exception))

        # Case 2: Tag changes after embedding request
        embed_resp = {"embeddings": [[0.6, 0.8] * 512]}
        with patch.object(model, "_request", side_effect=[initial_tags, embed_resp, mutated_tags]):
            with self.assertRaises(ProviderError) as ctx:
                model.embed_documents(["document chunk"])
            self.assertIn("Model digest changed", str(ctx.exception))

    def test_explicit_unknown_index_id_rejected_zero_http(self):
        model = OllamaProvider()
        # Resolve one known identity
        initial_tags = {"models": [{"name": "qwen3-embedding:0.6b", "digest": "sha256:initial"}]}
        with patch.object(model, "_request", return_value=initial_tags):
            known_id = model.resolve_identity({"version": "v1"})

        with patch.object(model, "_request") as mock_req:
            # Explicit unknown index_id on embed_query must be rejected with ProviderError before ANY HTTP call
            with self.assertRaises(ProviderError) as ctx_q:
                model.embed_query("test query", index_id="ollama:qwen3-embedding:0.6b:unknown123")
            self.assertIn("Unknown or nonmatching index_id", str(ctx_q.exception))
            mock_req.assert_not_called()

            # Explicit unknown index_id on embed_documents must be rejected with ProviderError before ANY HTTP call
            with self.assertRaises(ProviderError) as ctx_d:
                model.embed_documents(["doc text"], index_id="ollama:qwen3-embedding:0.6b:unknown123")
            self.assertIn("Unknown or nonmatching index_id", str(ctx_d.exception))
            mock_req.assert_not_called()

            # Omitted index_id preserves backward compatibility (embed_documents / raw embed)
            mock_req.return_value = {"embeddings": [[0.6, 0.8] * 512]}
            with patch.object(model, "_fetch_model_digest", return_value="sha256:initial"):
                res = model.embed(["raw doc"])
                self.assertEqual(len(res), 1)
                self.assertEqual(mock_req.call_count, 1)


if __name__ == "__main__":
    unittest.main()
