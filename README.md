# First Draft Studio

Streamlit application for journalists who want a stronger first draft from source material and a reusable Voice Print.

## What it does

- User login, signup, password changes, and an admin panel.
- Default admin account is `admin` / `admin`; change it in the UI or set different values in `.env` before first launch.
- Journalists can upload up to 50 writing samples to generate a Voice Print.
- Article drafting accepts documents, images, and zip archives with nested directories; supported files inside zips are unpacked and used individually.
- Long drafts are designed for up to 10,000 words with a configurable +/- 10% target range.
- Drafts over 2,500 words use a parallel section workflow: source brief, section plan, concurrent section drafting, then assembly.
- Users bring their own model settings in Account > Model API: OpenAI, local Ollama, or any OpenAI-compatible Chat Completions endpoint.
- Ollama can use either the native local chat endpoint, `http://localhost:11434/api/chat`, or the OpenAI-compatible endpoint, `http://localhost:11434/v1/chat/completions`.
- Drafts and Voice Prints are stored locally in SQLite.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Edit `.env` for app-level settings such as `APP_SECRET_KEY`, admin bootstrap credentials, storage paths, and default model names. Do not put shared provider keys in `.env`; users add their own keys in Account > Model API after signing in.

Then run:

```powershell
streamlit run app.py
```

For local Ollama, start Ollama and pull a model such as:

```powershell
ollama pull qwen3:8b
ollama pull qwen3:14b
ollama pull qwen3:30b
ollama pull qwen3-coder:30b
```

In Account > Model API, choose Local Ollama, then either Native Ollama Chat API with `http://localhost:11434/api/chat`, or OpenAI-compatible Chat Completions with `http://localhost:11434/v1`. Model names can be local tags such as `qwen3:8b`.

## Supported uploads

Documents: `txt`, `md`, `csv`, `json`, `xml`, `html`, `rtf`, `pdf`, `docx`, `pptx`, `xlsx`, `odt`.

Images: `png`, `jpg`, `jpeg`, `webp`, `gif`, `bmp`, `tif`, `tiff`.

Archives: `zip`. Zip files can include nested folders and nested zip files.

Legacy binary Office formats such as `.doc` and `.xls` are intentionally not parsed because they require platform-specific converters. Save them as `.docx` or `.xlsx` first.

## Production notes

- Replace default admin credentials before real use.
- Use a strong `APP_SECRET_KEY`.
- Keep `APP_SECRET_KEY` stable after users save provider keys; it is used to decrypt those keys.
- Keep `.env` and `data/` out of source control.
- Back up `data/app.db` and `data/uploads/`.
- For shared deployments, put Streamlit behind HTTPS and an authentication-aware reverse proxy.
