import unittest

from app import app, meta, weights
from inference import generate_name, tokenize_starter_text


class StarterTextTests(unittest.TestCase):
    def test_tokenize_valid_lowercases_and_strips(self):
        normalized, token_ids = tokenize_starter_text("  Al ", meta)
        self.assertEqual(normalized, "al")
        self.assertEqual(len(token_ids), 2)

    def test_tokenize_rejects_invalid_chars(self):
        with self.assertRaises(ValueError):
            tokenize_starter_text("a-", meta)

    def test_generate_name_respects_forced_prefix(self):
        _, starter_tokens = tokenize_starter_text("al", meta)
        name, steps = generate_name(
            weights,
            meta,
            temperature=0.5,
            seed=123,
            starter_tokens=starter_tokens,
        )
        self.assertGreaterEqual(len(steps), 2)
        self.assertEqual(steps[0]["output_char"], "a")
        self.assertEqual(steps[1]["output_char"], "l")
        self.assertTrue(name.startswith("al"))


class ApiStarterTextTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_generate_rejects_invalid_starter_text(self):
        resp = self.client.post(
            "/api/generate",
            json={"temperature": 0.5, "starter_text": "a!"},
        )
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json()
        self.assertIn("error", payload)

    def test_step_init_prefills_token_ids_and_steps(self):
        resp = self.client.post(
            "/api/step/init",
            json={"temperature": 0.5, "starter_text": "al"},
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload["token_ids"][0], meta["BOS"])
        self.assertEqual(len(payload["steps"]), 2)
        self.assertEqual(payload["steps"][0]["output_char"], "a")
        self.assertEqual(payload["steps"][1]["output_char"], "l")


if __name__ == "__main__":
    unittest.main()
