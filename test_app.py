import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

os.environ.setdefault("GROQ_API_KEY", "test-only-not-a-real-key")

import app


class AppSmokeTest(unittest.TestCase):
    def test_empty_model_responses_are_rejected(self):
        for content in (None, "", " \n "):
            with self.subTest(content=content):
                response = SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )])
                with patch.object(app.client.chat.completions, "create", return_value=response):
                    with self.assertRaisesRegex(ValueError, "no valid categories"):
                        app.predict_categories("Savings", "Open an account")
                    with self.assertRaisesRegex(ValueError, "no answer"):
                        app.chat_with_llama("Banking", "Savings", "Open an account")

    def test_lambda_prefix_matches_complete_path_segments(self):
        prefix = "/dev/genai/v0.0.1"
        for path, expected in (
            (prefix, "/"),
            (prefix + "/", "/"),
            (prefix + "/predict_categories", "/predict_categories"),
            (prefix + "0/predict_categories", prefix + "0/predict_categories"),
            ("/other" + prefix, "/other" + prefix),
            ("/", "/"),
        ):
            with self.subTest(path=path):
                event = {"path": path}
                with patch.object(app.serverless_wsgi, "handle_request") as handle:
                    app.lambda_handler(event, None)
                self.assertEqual(handle.call_args.args[1]["path"], expected)
                self.assertEqual(event["path"], path)

    def test_category_parsing_deduplicates_before_limiting(self):
        self.assertEqual(
            app._parse_categories("Banking, banking\r\nFinance\r\nUnknown, Investing, Books"),
            ["Banking", "Finance", "Investing"],
        )

    def test_routes_and_validation(self):
        client = app.app.test_client()

        self.assertEqual(client.get("/").get_json(), {"message": "API is running"})
        self.assertEqual(
            app._parse_categories('1. banking, "Finance"\n- Investing'),
            ["Banking", "Finance", "Investing"],
        )
        self.assertEqual(client.post("/predict_categories", json={}).status_code, 400)
        self.assertEqual(
            client.post("/predict_categories", data="{", content_type="application/json").status_code,
            400,
        )
        self.assertEqual(client.post("/predict_categories", json=[]).status_code, 400)
        self.assertEqual(
            client.post(
                "/predict_categories",
                json={"subject": 123, "description": "Open an account"},
            ).status_code,
            400,
        )
        self.assertEqual(
            client.post(
                "/generate_answer",
                json={"category": "Unknown", "subject": "Test", "description": "Test"},
            ).status_code,
            400,
        )

        with patch.object(app, "predict_categories", return_value=["Banking"]):
            response = client.post(
                "/predict_categories",
                json={"subject": "Savings", "description": "Open an account"},
            )
        self.assertEqual(response.get_json(), ["Banking"])

    def test_categories_match_prompts(self):
        self.assertEqual(len(app.categories), 38)
        self.assertEqual(set(app.categories), set(app.category_prompts))

    def test_generate_answer_accepts_any_category_case(self):
        client = app.app.test_client()
        with patch.object(app, "chat_with_llama", return_value="Use a trellis.") as chat:
            response = client.post(
                "/generate_answer",
                json={"category": " gardening ", "subject": "Beans", "description": "Support?"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), "Use a trellis.")
        chat.assert_called_once_with("Gardening", "Beans", "Support?")

    def test_oversized_fields_are_rejected(self):
        client = app.app.test_client()
        too_long = "x" * (app.MAX_FIELD_CHARS + 1)
        with patch.object(app, "predict_categories") as predict:
            response = client.post(
                "/predict_categories",
                json={"subject": "Savings", "description": too_long},
            )
        self.assertEqual(response.status_code, 400)
        predict.assert_not_called()

    def test_upstream_failures_return_502(self):
        client = app.app.test_client()
        with patch.object(app, "predict_categories", side_effect=app.UpstreamError("empty")):
            response = client.post(
                "/predict_categories",
                json={"subject": "Savings", "description": "Open an account"},
            )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json(), {"error": "upstream_error"})

    def test_unknown_routes_still_return_404(self):
        self.assertEqual(app.app.test_client().get("/missing").status_code, 404)

    def test_role_prompt_is_sent_as_system_message_with_guardrails(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="See a doctor.")
        )])
        with patch.object(app.client.chat.completions, "create", return_value=response) as create:
            app.chat_with_llama("Eye Medical", "Blurry", "Why is my vision blurry?")
        system, user = create.call_args.kwargs["messages"]
        self.assertEqual(system["role"], "system")
        self.assertIn("not a diagnosis", system["content"])
        self.assertEqual(user, {"role": "user", "content": "Subject: Blurry\nQuestion: Why is my vision blurry?"})


if __name__ == "__main__":
    unittest.main()
