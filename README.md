# CSV Comment Parser (Streamlit)

A small Python project with a Streamlit UI to parse CSV comment files containing:

- `display_name`
- `message`

## Project Structure

```text
.
├── app.py
├── parser.py
├── requirements.txt
└── tests
    └── test_parser.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Run Tests

```bash
pytest -q
```
