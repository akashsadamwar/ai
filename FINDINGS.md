# New Joiner Task Findings

## Local setup

The `NewJoineeTask` branch was cloned and its dependencies were installed from
`requirements.txt` with Python 3.13.

1. Create a personal key in the [Groq Console](https://console.groq.com/keys).
2. Copy `.env.example` to `.env` and replace the sample value with that key.
3. Run `py -m unittest -v`.
4. Run `py app.py`, then call the API at `http://localhost:8000`.

Never commit `.env`; it is excluded by `.gitignore`.

## Code flow

1. `app.py` loads `GROQ_API_KEY`, creates the global Groq client, and initializes Flask.
2. `POST /predict_categories` validates `subject` and `description`, asks Groq to choose
   three entries from `categories`, and filters the response through `_parse_categories`.
3. `POST /generate_answer` validates the category and question fields, selects a
   category-specific role prompt, sends it to Groq, and returns the generated text.
4. `lambda_handler` removes the configured API Gateway prefix and passes the event to
   Flask through `serverless_wsgi`.
5. Running `app.py` directly starts Flask on port 8000 for local development.

## Verification

- Dependency installation: passed.
- Python syntax compilation: passed.
- Offline Flask smoke tests for health, validation, and category response: passed.
- Live Groq requests: passed for category prediction and answer generation.

## Findings

Fixed locally:

- `.env` was loaded after the Groq client was created, so startup failed even with a
  correctly configured `.env` file.
- `llama-3.1-8b-instant` was shut down for free and developer tiers on August 16, 2026;
  Groq recommends `openai/gpt-oss-20b` as its replacement.
- Malformed JSON became a 500 response, and unknown categories triggered unnecessary
  Groq calls. Both now return 400 responses.
- Classification was stochastic and could silently return unrelated fallback categories.
  It is now deterministic, accepts minor formatting variations, and fails explicitly when
  Groq returns no valid category.
- The repository did not protect `.env` or include an executable test.

Still recommended:

- Restrict wildcard CORS and disable Flask debug mode outside local development.
- Add input-length limits, request authentication, rate limiting, and Groq timeouts.
- Add medical and financial safety language; the current prompts present the model as a
  clinician or advisor without guardrails.
- Pin or lock dependency versions.
