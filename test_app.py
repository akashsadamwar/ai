import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-only-not-a-real-key")

import app


class AppSmokeTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
