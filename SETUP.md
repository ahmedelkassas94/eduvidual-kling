# Setup

## Python version (fixes the warnings)

Use **Python 3.10 or newer**. This removes:

- **Google auth / oauth2**: “Python version 3.9 past its end of life” warnings.
- **urllib3 / OpenSSL**: On macOS, system Python is often built with LibreSSL; Python 3.10+ and Homebrew Python use OpenSSL, so the “urllib3 v2 only supports OpenSSL” warning goes away.

### On macOS (recommended)

**If you don’t have Homebrew**, install it first (one command, then follow the on-screen instructions):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After install, the installer may tell you to add Homebrew to your PATH (e.g. for Apple Silicon: `echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile` then open a new terminal).

Then install Python (built with OpenSSL):

```bash
brew install python@3.12
```

**Alternative without Homebrew:** install Python 3.12 from [python.org/downloads](https://www.python.org/downloads/) (macOS installer). Then use that interpreter to create the venv, e.g. `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv` (path may vary).

Create and use a venv with that interpreter:

```bash
# From the project root (Eduvidual/)
python3.12 -m venv .venv
source .venv/bin/activate   # or: .venv/bin/activate on some shells
pip install -r requirements.txt
```

If you use **pyenv**, the repo includes `.python-version` set to `3.12`. Run `pyenv install` if needed, then create the venv:

```bash
pyenv install -s
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### On Linux / Windows

Use Python 3.10+ from your package manager or [python.org](https://www.python.org/downloads/), then:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Env and API keys

Copy `.env.example` to `.env` and set your API keys (see `.env.example` for required variables). The planner needs `GEMINI_API_KEY`; use a key with **no application restrictions** so it works from all environments.

## Verify

- **Gemini key**: `python verify_gemini_key.py`
- **Planner**: `python planner.py "Your topic" 15`
- **Orchestrator**: `python orchestrator.py projects/<project_id>`
