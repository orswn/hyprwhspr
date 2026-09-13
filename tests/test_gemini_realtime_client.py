import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from gemini_realtime_client import GeminiRealtimeClient
from provider_registry import get_models_for_backend, get_realtime_mode


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(json.loads(payload))


class GeminiRealtimeClientTests(unittest.TestCase):
    def _client(self, model="gemini-3.5-transcribe-live", mode="transcribe"):
        client = GeminiRealtimeClient(mode=mode)
        client.connected = True
        client.ws = FakeWebSocket()
        client.model = model
        return client

    def test_transcribe_model_detection(self):
        client = self._client(model="gemini-3.5-transcribe-live")
        self.assertTrue(client._is_transcribe_model())

        client_legacy = self._client(model="gemini-3.1-flash-live-preview")
        self.assertFalse(client_legacy._is_transcribe_model())

    def test_gemini_35_transcribe_setup_payload(self):
        client = self._client(model="gemini-3.5-transcribe-live")
        client.language = "en"
        client._after_open(client.ws)

        self.assertEqual(len(client.ws.sent), 1)
        setup = client.ws.sent[0].get("setup", {})
        self.assertEqual(setup.get("model"), "models/gemini-3.5-transcribe-live")
        self.assertEqual(setup.get("generationConfig"), {"responseModalities": ["TEXT"]})
        self.assertEqual(setup.get("inputAudioTranscription"), {"languageCodes": ["en"]})
        self.assertNotIn("speechConfig", setup.get("generationConfig", {}))
        self.assertNotIn("systemInstruction", setup)
        self.assertNotIn("outputAudioTranscription", setup)

    def test_gemini_35_transcribe_setup_payload_without_language(self):
        client = self._client(model="gemini-3.5-transcribe-live")
        client.language = None
        client._after_open(client.ws)

        setup = client.ws.sent[0].get("setup", {})
        self.assertEqual(setup.get("inputAudioTranscription"), {})

    def test_legacy_gemini_setup_payload(self):
        client = self._client(model="gemini-3.1-flash-live-preview")
        client.language = "it"
        client._after_open(client.ws)

        self.assertEqual(len(client.ws.sent), 1)
        setup = client.ws.sent[0].get("setup", {})
        self.assertEqual(setup.get("model"), "models/gemini-3.1-flash-live-preview")
        self.assertEqual(setup.get("generationConfig", {}).get("responseModalities"), ["AUDIO"])
        self.assertIn("speechConfig", setup.get("generationConfig", {}))
        self.assertIn("systemInstruction", setup)
        self.assertIn("outputAudioTranscription", setup)

    def test_interim_and_final_transcription_events(self):
        client = self._client(model="gemini-3.5-transcribe-live")
        previews = []
        client.set_partial_transcript_callback(previews.append)

        client._handle_event({
            "serverContent": {
                "interimInputTranscription": {"text": "hello world"}
            }
        })
        self.assertEqual(previews[-1], "hello world")
        self.assertFalse(client.response_complete)

        client._handle_event({
            "serverContent": {
                "inputTranscription": {"text": "Hello, world!"}
            }
        })
        self.assertEqual(previews[-1], "Hello, world!")
        self.assertEqual(client.current_response_text, "Hello, world!")
        self.assertTrue(client.response_complete)
        self.assertTrue(client.response_event.is_set())

    def test_provider_registry_contains_gemini_35_transcribe(self):
        models = get_models_for_backend("google", "realtime-ws")
        self.assertIn("gemini-3.5-transcribe-live", models)
        self.assertEqual(get_realtime_mode("google", "gemini-3.5-transcribe-live"), "transcribe")


    def test_gemini_transcribe_rejects_converse_mode(self):
        from unittest import mock
        from backends.realtime_ws_backend import RealtimeWsBackend

        class FakeConfig:
            def __init__(self, values):
                self.values = values
            def get_setting(self, key, default=None):
                return self.values.get(key, default)
            def migrate_api_key_to_credential_manager(self):
                pass

        class FakeManager:
            def __init__(self, config):
                self.config = config
                self.ready = True

        config = FakeConfig({
            "transcription_backend": "realtime-ws",
            "websocket_provider": "google",
            "websocket_model": "gemini-3.5-transcribe-live",
            "realtime_mode": "converse",
        })
        backend = RealtimeWsBackend(FakeManager(config))
        with mock.patch("backends.realtime_ws_backend.get_credential", return_value="fake-key"):
            self.assertFalse(backend.initialize())
        self.assertIsNone(backend._realtime_client)


if __name__ == "__main__":
    unittest.main()
