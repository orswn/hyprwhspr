import sys
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends.rest_api_backend import RestApiBackend


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


class RestApiBackendGoogleTests(unittest.TestCase):
    def test_google_transcribe_payload_and_response(self):
        config = FakeConfig({
            "transcription_backend": "rest-api",
            "rest_api_provider": "google",
            "rest_endpoint_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-transcribe:generateContent",
            "rest_body": {"model": "gemini-3.5-transcribe"},
            "language": "en",
        })
        backend = RestApiBackend(FakeManager(config))

        dummy_audio = np.zeros(16000, dtype=np.float32)

        fake_resp = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "audioTranscription": {
                                    "text": "Hello world from Gemini 3.5 Transcribe."
                                }
                            }
                        ]
                    }
                }
            ]
        }

        with mock.patch("backends.rest_api_backend.get_credential", return_value="fake-google-key"):
            with mock.patch("requests.post", return_value=fake_resp) as mock_post:
                result = backend.transcribe(dummy_audio, 16000)

        self.assertEqual(result, "Hello world from Gemini 3.5 Transcribe.")
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "fake-google-key")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        payload = kwargs["json"]
        self.assertIn("contents", payload)
        self.assertIn("inlineData", payload["contents"][0]["parts"][0])
        self.assertEqual(payload["generationConfig"]["audioTranscriptionConfig"]["mode"], "SMART")
        self.assertEqual(payload["generationConfig"]["audioTranscriptionConfig"]["languageCodes"], ["en"])


if __name__ == "__main__":
    unittest.main()
