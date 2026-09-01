import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile

from orderflow_agent.runtime.providers import (
    KernelLoomProvider,
    HuggingFaceEndpointProvider,
    OpenAIProvider,
    OpenAgentProvider,
    ProviderStatus,
    ProviderUnavailable,
    RuntimeSettings,
)
from orderflow_agent.runtime.customer_service import check_runtime

TEST_CREDENTIAL = "unit-test-placeholder"


class ProviderAdapterTest(unittest.TestCase):
    @patch("orderflow_agent.runtime.customer_service.streaming_provider")
    def test_runtime_check_uses_provider_status_contract(self, provider):
        provider.return_value.check.return_value = ProviderStatus(
            True,
            "Local model",
            "Model files are available.",
        )

        self.assertEqual(check_runtime(), (True, "Model files are available."))

    def test_safe_summary_never_contains_secret(self):
        settings = RuntimeSettings(
            provider_id="openai",
            api_key=TEST_CREDENTIAL,
            response_model="model",
        )
        summary = settings.safe_summary()

        self.assertNotIn(TEST_CREDENTIAL, str(summary))
        self.assertEqual(summary["credential_source"], "session or environment")

    @patch("openai.OpenAI")
    def test_openai_uses_responses_without_storage_and_configured_embeddings(self, client_type):
        client = client_type.return_value
        client.responses.create.return_value = SimpleNamespace(output_text="Approved wording")
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.2, 0.4])]
        )
        provider = OpenAIProvider(
            RuntimeSettings(
                provider_id="openai",
                api_key=TEST_CREDENTIAL,
                response_model="response-model",
                embedding_model="embedding-model",
            )
        )

        self.assertEqual(provider.generate("policy", [("customer", "hello")]), "Approved wording")
        self.assertEqual(provider.embed_texts(["pizza"]), [[0.2, 0.4]])
        self.assertFalse(client.responses.create.call_args.kwargs["store"])
        self.assertEqual(client.embeddings.create.call_args.kwargs["model"], "embedding-model")
        provider.close()
        client.close.assert_called_once_with()

    @patch("openai.OpenAI")
    def test_openai_yields_native_response_events(self, client_type):
        class EventStream:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                return iter(
                    [
                        SimpleNamespace(type="response.output_text.delta", delta="Fresh "),
                        SimpleNamespace(type="response.output_text.delta", delta="reply"),
                        SimpleNamespace(type="response.completed"),
                    ]
                )

            def close(self):
                self.closed = True

        stream = EventStream()
        client_type.return_value.responses.create.return_value = stream
        provider = OpenAIProvider(
            RuntimeSettings(
                provider_id="openai",
                api_key=TEST_CREDENTIAL,
                response_model="response-model",
            )
        )

        self.assertEqual(list(provider.stream_generate("policy", [("user", "hello")])), ["Fresh ", "reply"])
        self.assertTrue(stream.closed)
        self.assertTrue(client_type.return_value.responses.create.call_args.kwargs["stream"])
        self.assertFalse(client_type.return_value.responses.create.call_args.kwargs["store"])

    def test_openai_requires_an_explicit_credential(self):
        with self.assertRaises(ProviderUnavailable):
            OpenAIProvider(RuntimeSettings(provider_id="openai", response_model="response-model"))

    @patch("openai.OpenAI")
    def test_openai_never_guesses_an_unconfigured_model_id(self, client_type):
        provider = OpenAIProvider(
            RuntimeSettings(provider_id="openai", api_key=TEST_CREDENTIAL)
        )

        self.assertFalse(provider.capabilities.text)
        self.assertFalse(provider.capabilities.embeddings)
        with self.assertRaises(ProviderUnavailable):
            provider.generate("policy", [("customer", "hello")])
        client_type.return_value.responses.create.assert_not_called()

    @patch("orderflow_agent.runtime.providers._json_request")
    def test_kernelloom_uses_verified_chat_and_embedding_contracts(self, request):
        request.side_effect = [
            {"choices": [{"message": {"content": "Local answer"}}]},
            {"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        ]
        provider = KernelLoomProvider(
            RuntimeSettings(
                provider_id="kernelloom",
                base_url="http://127.0.0.1:11435",
                response_model="chat-local",
                embedding_model="embed-local",
            )
        )

        self.assertEqual(provider.generate("policy", [("customer", "hello")]), "Local answer")
        self.assertEqual(provider.embed_texts(["hello"]), [[0.1, 0.2]])
        self.assertEqual(request.call_args_list[0].args[1], "/v1/chat/completions")
        self.assertEqual(request.call_args_list[1].args[1], "/v1/embeddings")

    @patch("orderflow_agent.runtime.providers._load_kernelloom_package")
    def test_kernelloom_python_package_loads_models_lazily_and_closes_them(self, package):
        class FakeConfig:
            def __init__(self, **values):
                self.values = values

        class FakeModel:
            instances = []

            def __init__(self, config):
                self.config = config
                self.closed = False
                self.__class__.instances.append(self)

            def generate(self, prompt):
                self.prompt = prompt
                return SimpleNamespace(text="Package answer")

            def stream(self, prompt, **generation):
                self.prompt = prompt
                self.generation = generation
                yield "Package "
                yield "stream"

            def embed(self, text):
                return [float(len(text)), 1.0]

            def close(self):
                self.closed = True

        package.return_value = (FakeModel, FakeConfig, "0.4.1")
        provider = KernelLoomProvider(
            RuntimeSettings(
                provider_id="kernelloom",
                kernelloom_transport="python",
                kernelloom_chat_model_path="chat.gguf",
                kernelloom_embedding_model_path="embed.gguf",
                response_model="chat-local",
                embedding_model="embed-local",
                kernelloom_cpu_profile="latency",
                kernelloom_reserve_cores=2,
            )
        )

        self.assertEqual(FakeModel.instances, [])
        self.assertEqual(provider.generate("policy", [("customer", "hello")]), "Package answer")
        self.assertEqual(
            list(provider.stream_generate("policy", [("customer", "hello")])),
            ["Package ", "stream"],
        )
        self.assertEqual(provider.embed_texts(["pizza"]), [[5.0, 1.0]])
        self.assertEqual(len(FakeModel.instances), 2)
        self.assertFalse(FakeModel.instances[0].config.values["embedding"])
        self.assertTrue(FakeModel.instances[1].config.values["embedding"])
        self.assertEqual(FakeModel.instances[0].config.values["reserve_cores"], 2)

        provider.close()
        self.assertTrue(all(model.closed for model in FakeModel.instances))

    @patch("orderflow_agent.runtime.providers._load_kernelloom_package")
    def test_kernelloom_python_check_reports_missing_model_path(self, package):
        package.return_value = (object, object, "0.4.1")
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.gguf"
            provider = KernelLoomProvider(
                RuntimeSettings(
                    provider_id="kernelloom",
                    kernelloom_transport="python",
                    kernelloom_chat_model_path=str(missing),
                    response_model="chat-local",
                )
            )
            status = provider.check()

        self.assertFalse(status.ok)
        self.assertIn("does not exist", status.detail)

    @patch("orderflow_agent.runtime.providers._load_kernelloom_package")
    def test_kernelloom_inner_stream_closes_when_customer_reply_stops_early(self, package):
        class FakeConfig:
            def __init__(self, **values):
                self.values = values

        class ClosingStream:
            def __init__(self):
                self.closed = False
                self.fragments = iter(("First sentence. ", "Second sentence. ", "unused"))

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.fragments)

            def close(self):
                self.closed = True

        source = ClosingStream()

        class FakeModel:
            def __init__(self, config):
                self.config = config

            def stream(self, prompt, **generation):
                return source

            def close(self):
                return None

        package.return_value = (FakeModel, FakeConfig, "0.4.1")
        provider = KernelLoomProvider(
            RuntimeSettings(
                provider_id="kernelloom",
                kernelloom_transport="python",
                kernelloom_chat_model_path="chat.gguf",
                response_model="chat-local",
            )
        )
        stream = provider.stream_generate("policy", [("customer", "hello")])

        self.assertEqual(next(stream), "First sentence. ")
        stream.close()

        self.assertTrue(source.closed)

    @patch("orderflow_agent.runtime.providers._json_request")
    def test_openagent_reads_assistant_message(self, request):
        request.return_value = {
            "messages": [
                {"role": "user", "content": "request"},
                {"role": "assistant", "content": "OpenAgent answer"},
            ]
        }
        provider = OpenAgentProvider(
            RuntimeSettings(provider_id="openagent", base_url="http://127.0.0.1:8765")
        )

        self.assertEqual(provider.generate("policy", [("customer", "hello")]), "OpenAgent answer")
        payload = request.call_args.kwargs["payload"]
        self.assertFalse(payload["allow_external"])

    def test_rejects_credentials_inside_url(self):
        with self.assertRaises(ProviderUnavailable):
            OpenAgentProvider(
                RuntimeSettings(provider_id="openagent", base_url="http://user:password@localhost:8765")
            )

    @patch("orderflow_agent.runtime.providers._json_request")
    def test_hugging_face_uses_chat_completion_contract(self, request):
        request.return_value = {"choices": [{"message": {"content": "Hosted answer"}}]}
        provider = HuggingFaceEndpointProvider(
            RuntimeSettings(
                provider_id="huggingface",
                base_url="https://example.endpoints.huggingface.cloud/v1",
                api_key="hf_test",
                response_model="endpoint-name",
            )
        )

        self.assertEqual(provider.generate("policy", [("customer", "hello")]), "Hosted answer")
        self.assertEqual(request.call_args.args[1], "/chat/completions")

    def test_hugging_face_requires_versioned_base_url(self):
        with self.assertRaises(ProviderUnavailable):
            HuggingFaceEndpointProvider(
                RuntimeSettings(
                    provider_id="huggingface",
                    base_url="https://example.endpoints.huggingface.cloud",
                    api_key="hf_test",
                    response_model="endpoint-name",
                )
            )


if __name__ == "__main__":
    unittest.main()
