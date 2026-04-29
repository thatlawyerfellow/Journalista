from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

from src.config import AppConfig, load_config
from src.db import (
    authenticate,
    create_user,
    get_user,
    get_user_model_settings,
    get_user_model_settings_for_provider,
    get_voice_print,
    init_db,
    list_drafts,
    list_users,
    list_voice_prints,
    save_draft,
    save_user_model_settings,
    save_voice_print,
    set_active_voice_print,
    update_password,
    update_user,
)
from src.ingest import (
    SUPPORTED_EXTENSIONS,
    build_manifest,
    collect_images,
    combine_text_context,
    ingest_uploads,
)
from src.openai_service import ArticleGenerator, OpenAIConfigurationError
from src.security import password_is_reasonable


st.set_page_config(
    page_title="First Draft Studio",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)


def boot() -> AppConfig:
    config = load_config()
    init_db(config)
    config.upload_dir.mkdir(parents=True, exist_ok=True)
    return config


CONFIG = boot()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


class GenerationProgress:
    def __init__(self, steps: list[str]):
        self.steps = steps
        self.current_index = 0
        self.percent = 0
        self.messages: list[str] = []
        self.progress_slot = st.progress(0, text="Starting...")
        self.steps_slot = st.empty()
        self.detail_slot = st.empty()
        self.log_slot = st.empty()
        self.update(0, 0, "Starting.")

    def update(self, step_index: int, percent: int | None = None, detail: str | None = None) -> None:
        self.current_index = max(0, min(step_index, len(self.steps) - 1))
        if percent is not None:
            self.percent = max(self.percent, max(0, min(100, int(percent))))
        label = self.steps[self.current_index]
        self.progress_slot.progress(self.percent, text=f"{self.percent}% - {label}")
        self.steps_slot.markdown(self._steps_markdown())
        if detail:
            self.detail_slot.caption(detail)

    def note(self, message: str) -> None:
        if not self.messages or self.messages[-1] != message:
            self.messages.append(message)
        recent = self.messages[-6:]
        self.log_slot.markdown("Recent activity:\n" + "\n".join(f"- {item}" for item in recent))

    def complete(self, detail: str = "Complete.") -> None:
        self.current_index = len(self.steps) - 1
        self.percent = 100
        self.update(self.current_index, 100, detail)

    def _steps_markdown(self) -> str:
        lines = []
        for index, label in enumerate(self.steps):
            if index < self.current_index:
                marker = "[x]"
            elif index == self.current_index:
                marker = "[>]"
            else:
                marker = "[ ]"
            lines.append(f"{marker} {label}")
        return "\n".join(lines)


def summarize_ingested_items(items: list) -> dict[str, int]:
    readable = [item for item in items if not item.error and (item.text.strip() or item.image_data_url)]
    return {
        "total": len(items),
        "readable": len(readable),
        "documents": sum(1 for item in readable if item.text.strip()),
        "images": sum(1 for item in readable if item.image_data_url),
        "unsupported": sum(1 for item in items if item.kind == "unsupported"),
        "errors": sum(1 for item in items if item.error and item.kind != "unsupported"),
    }


def make_ingest_progress_callback(
    tracker: GenerationProgress,
    *,
    step_index: int,
    start_percent: int,
    end_percent: int,
):
    def progress(message: str, completed: int, total: int) -> None:
        total = max(1, total)
        completed = max(0, min(completed, total))
        percent = start_percent + int((end_percent - start_percent) * completed / total)
        tracker.note(message)
        tracker.update(step_index, percent, message)

    return progress


def progress_from_fraction(start: int, end: int, completed: int, total: int) -> int:
    total = max(1, total)
    completed = max(0, min(completed, total))
    return start + int((end - start) * completed / total)


def parse_fraction(message: str) -> tuple[int, int] | None:
    match = re.search(r"\b(\d+)\s+of\s+(\d+)\b", message)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"\b(\d+)/(\d+)\b", message)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def current_user() -> dict | None:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return None
    user = get_user(CONFIG, int(user_id))
    if user and user["active"]:
        user.pop("password_hash", None)
        return user
    st.session_state.clear()
    return None


def login_view() -> None:
    st.title("First Draft Studio")
    st.caption("Voice-aware article drafting for journalists.")

    login_tab, signup_tab = st.tabs(["Login", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")
        if submitted:
            ok, message, user = authenticate(CONFIG, username, password)
            if ok and user:
                st.session_state["user_id"] = user["id"]
                st.rerun()
            st.error(message)

    with signup_tab:
        with st.form("signup_form"):
            username = st.text_input("Choose a username")
            email = st.text_input("Email", placeholder="name@example.com")
            password = st.text_input("Choose a password", type="password")
            submitted = st.form_submit_button("Create account")
        if submitted:
            ok_password, password_message = password_is_reasonable(password)
            if not username.strip():
                st.error("Username is required.")
            elif not ok_password:
                st.error(password_message)
            else:
                ok, message = create_user(CONFIG, username, email or None, password)
                if ok:
                    st.success("Account created. You can log in now.")
                else:
                    st.error(message)


def sidebar(user: dict) -> str:
    with st.sidebar:
        st.title("First Draft")
        st.write(f"Signed in as **{user['username']}**")
        options = ["Draft Article", "Voice Print", "Draft Archive", "Account"]
        if user["role"] == "admin":
            options.insert(3, "Admin")
        page = st.radio("Workspace", options, label_visibility="collapsed")
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        if user["role"] == "admin" and CONFIG.admin_username == "admin" and CONFIG.admin_password == "admin":
            st.warning("Default admin credentials are enabled in .env.")
    return page


def voice_print_page(user: dict) -> None:
    st.header("Voice Print")
    existing = list_voice_prints(CONFIG, user["id"])
    active = next((vp for vp in existing if vp["active"]), None)

    if active:
        st.subheader("Active Voice Print")
        st.write(f"**{active['name']}** · {active['sample_count']} samples · {active['created_at']}")
        st.text_area("Instructions", active["instructions"], height=360, disabled=True)

    with st.expander("Create or replace Voice Print", expanded=not bool(active)):
        name = st.text_input("Voice Print name", value="Primary Voice Print")
        author_notes = st.text_area(
            "Author notes",
            placeholder="Beat, publication context, preferred boundaries, or known quirks.",
            height=120,
        )
        samples = st.file_uploader(
            "Upload writing samples",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            key="voice_samples",
        )
        st.caption(
            f"Recommended {CONFIG.recommended_min_sample_files}-{CONFIG.max_sample_files} samples. "
            "Documents, images, and zip archives are accepted."
        )
        if samples and len(samples) > CONFIG.max_sample_files:
            st.error(f"Upload no more than {CONFIG.max_sample_files} sample files at once.")

        if st.button("Generate Voice Print", type="primary", disabled=not samples):
            if len(samples) > CONFIG.max_sample_files:
                st.stop()
            steps = [
                "Validate uploads",
                "Read and unzip samples",
                "Prepare Voice Print context",
                "Connect to model API",
                "Generate Voice Print",
                "Save Voice Print",
                "Complete",
            ]
            with st.status("Generating Voice Print...", expanded=True) as status:
                tracker = GenerationProgress(steps)
                tracker.update(0, 5, f"{len(samples)} top-level upload(s) selected.")
                tracker.update(1, 15, "Reading files and expanding zip archives.")
                items = ingest_uploads(
                    samples,
                    CONFIG,
                    user["id"],
                    "voice-samples",
                    progress_callback=tracker.note,
                    progress_event_callback=make_ingest_progress_callback(
                        tracker,
                        step_index=1,
                        start_percent=15,
                        end_percent=30,
                    ),
                )
                manifest = build_manifest(items)
                summary = summarize_ingested_items(items)
                readable = [item for item in items if not item.error and (item.text.strip() or item.image_data_url)]
                tracker.update(
                    1,
                    30,
                    (
                        f"Read {summary['readable']} usable item(s): "
                        f"{summary['documents']} document(s), {summary['images']} image(s)."
                    ),
                )
                if summary["unsupported"] or summary["errors"]:
                    tracker.note(
                        f"Skipped {summary['unsupported']} unsupported item(s) and {summary['errors']} errored item(s)."
                    )
                if not readable:
                    status.update(label="No readable samples found.", state="error")
                    st.error("None of the uploaded samples could be read.")
                    return
                if len(readable) < CONFIG.recommended_min_sample_files:
                    tracker.note(
                        "Using fewer than the recommended sample count. You can regenerate later with more work."
                    )
                tracker.update(2, 38, "Building model context from readable sample files.")
                context = combine_text_context(readable, CONFIG.max_total_context_chars)
                images = collect_images(readable, CONFIG.max_images_per_request)
                tracker.update(
                    2,
                    48,
                    f"Prepared {len(context):,} text characters and {len(images)} image(s).",
                )
                live_voice_print: list[str] = []
                voice_preview = st.empty()
                last_render = {"chars": 0}

                def openai_progress(message: str) -> None:
                    step_index = 3 if "Connecting" in message or "accepted" in message else 4
                    percent = 54 if step_index == 3 else max(tracker.percent, 58)
                    tracker.update(step_index, percent, message)

                def voice_delta(delta: str) -> None:
                    live_voice_print.append(delta)
                    text = "".join(live_voice_print)
                    chars = len(text)
                    if chars - last_render["chars"] >= 500:
                        percent = min(88, 58 + chars // 150)
                        tracker.update(4, percent, f"Generated {chars:,} characters.")
                        voice_preview.markdown(text)
                        last_render["chars"] = chars

                try:
                    model_settings = get_user_model_settings(CONFIG, user["id"], include_secret=True)
                    generator = ArticleGenerator(CONFIG, model_settings)
                    tracker.update(3, 52, f"Using model {model_settings['voice_model']}.")
                    instructions = generator.generate_voice_print(
                        sample_context=context,
                        image_inputs=images,
                        author_notes=author_notes,
                        sample_count=len(readable),
                        on_progress=openai_progress,
                        on_text_delta=voice_delta,
                    )
                except OpenAIConfigurationError as exc:
                    status.update(label="Model API is not configured.", state="error")
                    st.error(str(exc))
                    return
                except Exception as exc:
                    status.update(label="Voice Print generation failed.", state="error")
                    st.error(f"Model request failed: {exc}")
                    return
                voice_preview.markdown(instructions)
                tracker.update(5, 94, "Saving Voice Print and making it active.")
                save_voice_print(
                    CONFIG,
                    user["id"],
                    name,
                    instructions,
                    len(readable),
                    json.dumps(manifest, ensure_ascii=False),
                )
                tracker.complete("Voice Print saved and set as active.")
                status.update(label="Voice Print created.", state="complete")
            st.success("Voice Print saved and set as active.")
            st.text_area("Generated Voice Print", instructions, height=360, disabled=True)

    if len(existing) > 1:
        st.subheader("Saved Voice Prints")
        for vp in existing:
            cols = st.columns([3, 1, 1])
            cols[0].write(f"**{vp['name']}** · {vp['created_at']}")
            cols[1].write("Active" if vp["active"] else "")
            if not vp["active"] and cols[2].button("Make active", key=f"activate-vp-{vp['id']}"):
                set_active_voice_print(CONFIG, user["id"], vp["id"])
                st.rerun()


def draft_article_page(user: dict) -> None:
    st.header("Draft Article")
    voice_prints = list_voice_prints(CONFIG, user["id"])
    voice_options = {"No Voice Print": None}
    for vp in voice_prints:
        label = f"{vp['name']} ({'active' if vp['active'] else vp['created_at']})"
        voice_options[label] = vp["id"]

    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        title = st.text_input("Title or topic")
        selected_voice_label = st.selectbox("Voice Print", list(voice_options.keys()))
        word_count = st.number_input(
            "Target word count",
            min_value=250,
            max_value=CONFIG.article_max_words,
            value=900,
            step=50,
        )
        st.caption(
            f"Designed for up to {CONFIG.article_max_words:,} words with "
            f"+/- {CONFIG.article_word_tolerance_pct}% target range."
        )
        tone_strength = st.select_slider(
            "Voice Print strength",
            options=[0, 20, 40, 60, 80, 100],
            value=60,
            format_func=lambda value: {
                0: "None",
                20: "Light",
                40: "Moderate",
                60: "Balanced",
                80: "Close",
                100: "Very close",
            }[value],
        )
        include_editor_notes = st.checkbox("Include editor notes", value=True)
        instructions = st.text_area(
            "Draft instructions",
            placeholder="Angle, audience, must-include points, exclusions, structure, deadline context.",
            height=160,
        )
        uploads = st.file_uploader(
            "Source files",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            key="source_uploads",
        )
        generate = st.button("Generate Draft", type="primary", disabled=not uploads)

    with right:
        st.subheader("Latest Output")
        output_placeholder = st.empty()

    if generate:
        selected_voice_id = voice_options[selected_voice_label]
        selected_voice = get_voice_print(CONFIG, selected_voice_id, user["id"]) if selected_voice_id else None
        steps = [
            "Validate sources",
            "Read and unzip sources",
            "Prepare article context",
            "Connect to model API",
            "Generate draft",
            "Save draft",
            "Complete",
        ]
        with st.status("Generating draft...", expanded=True) as status:
            tracker = GenerationProgress(steps)
            tracker.update(0, 5, f"{len(uploads)} top-level upload(s) selected.")
            tracker.update(1, 15, "Reading files and expanding zip archives.")
            items = ingest_uploads(
                uploads,
                CONFIG,
                user["id"],
                "article-sources",
                progress_callback=tracker.note,
                progress_event_callback=make_ingest_progress_callback(
                    tracker,
                    step_index=1,
                    start_percent=15,
                    end_percent=30,
                ),
            )
            manifest = build_manifest(items)
            readable = [item for item in items if not item.error and (item.text.strip() or item.image_data_url)]
            summary = summarize_ingested_items(items)
            if not readable:
                status.update(label="No readable sources found.", state="error")
                st.error("None of the uploaded files could be read.")
                return
            tracker.update(
                1,
                30,
                (
                    f"Read {summary['readable']} usable item(s): "
                    f"{summary['documents']} document(s), {summary['images']} image(s)."
                ),
            )
            if summary["unsupported"] or summary["errors"]:
                tracker.note(
                    f"Skipped {summary['unsupported']} unsupported item(s) and {summary['errors']} errored item(s)."
                )
            tracker.update(2, 38, "Building source context for the model.")
            context = combine_text_context(readable, CONFIG.max_total_context_chars)
            images = collect_images(readable, CONFIG.max_images_per_request)
            tracker.update(
                2,
                48,
                f"Prepared {len(context):,} text characters and {len(images)} image(s).",
            )
            live_article: list[str] = []
            last_render = {"chars": 0, "words": 0}

            def openai_progress(message: str) -> None:
                fraction = parse_fraction(message)
                if "source material" in message or "source brief chunk" in message:
                    step_index = 2
                    percent = progress_from_fraction(49, 56, *fraction) if fraction else 50
                elif "Condensing article source brief" in message:
                    step_index = 2
                    percent = max(tracker.percent, 56)
                elif "Connecting" in message or "accepted" in message:
                    step_index = 3
                    percent = 57
                elif "Building source brief and" in message:
                    step_index = 4
                    percent = 60
                elif "Parallel drafting" in message:
                    step_index = 4
                    percent = 62
                elif "Section" in message and "drafted" in message and fraction:
                    step_index = 4
                    percent = progress_from_fraction(64, 86, *fraction)
                elif "Draft is short" in message or "finished writing text" in message:
                    step_index = 4
                    percent = max(tracker.percent, 86)
                else:
                    step_index = 4
                    percent = max(tracker.percent, 58)
                tracker.update(step_index, percent, message)

            def article_delta(delta: str) -> None:
                live_article.append(delta)
                text = "".join(live_article)
                chars = len(text)
                words = count_words(text)
                should_render = chars - last_render["chars"] >= 600 or words - last_render["words"] >= 60
                if should_render:
                    target = max(1, int(word_count))
                    percent = min(88, 58 + int(30 * min(words, target) / target))
                    tracker.update(4, percent, f"Generated about {words:,} word(s).")
                    output_placeholder.markdown(text)
                    last_render["chars"] = chars
                    last_render["words"] = words

            try:
                model_settings = get_user_model_settings(CONFIG, user["id"], include_secret=True)
                generator = ArticleGenerator(CONFIG, model_settings)
                tracker.update(3, 52, f"Using model {model_settings['model']}.")
                article = generator.generate_article(
                    title=title,
                    source_context=context,
                    image_inputs=images,
                    user_instructions=instructions,
                    word_count=int(word_count),
                    tone_strength=int(tone_strength),
                    voice_print=selected_voice["instructions"] if selected_voice else None,
                    include_editor_notes=include_editor_notes,
                    on_progress=openai_progress,
                    on_text_delta=article_delta,
                )
            except OpenAIConfigurationError as exc:
                status.update(label="Model API is not configured.", state="error")
                st.error(str(exc))
                return
            except Exception as exc:
                status.update(label="Draft generation failed.", state="error")
                st.error(f"Model request failed: {exc}")
                return
            output_placeholder.markdown(article)
            tracker.update(5, 94, f"Saving draft with {count_words(article):,} generated word(s).")
            draft_id = save_draft(
                CONFIG,
                user["id"],
                selected_voice_id,
                title,
                instructions,
                int(word_count),
                int(tone_strength),
                manifest,
                article,
            )
            tracker.complete(f"Draft #{draft_id} saved.")
            status.update(label=f"Draft #{draft_id} created.", state="complete")
        st.download_button(
            "Download Markdown",
            article,
            file_name=f"draft-{draft_id}.md",
            mime="text/markdown",
        )


def archive_page(user: dict) -> None:
    st.header("Draft Archive")
    drafts = list_drafts(CONFIG, user["id"])
    if not drafts:
        st.info("No drafts yet.")
        return
    for draft in drafts:
        title = draft["title"] or f"Draft #{draft['id']}"
        with st.expander(f"{title} · {draft['created_at']}"):
            st.caption(
                f"{draft['word_count']} words target · voice strength {draft['tone_strength']}/100 · "
                f"{draft['voice_print_name'] or 'no voice print'}"
            )
            st.markdown(draft["article"])
            st.download_button(
                "Download Markdown",
                draft["article"],
                file_name=f"draft-{draft['id']}.md",
                mime="text/markdown",
                key=f"download-{draft['id']}",
            )


def model_settings_page_section(user: dict) -> None:
    st.subheader("Model API")
    active_settings = get_user_model_settings(CONFIG, user["id"])
    provider_labels = {
        "openai": "OpenAI",
        "ollama": "Local Ollama",
        "custom": "OpenAI-compatible API",
    }
    active_label = provider_labels.get(active_settings["provider"], "OpenAI")
    st.caption(f"Active connection: {active_label} · {active_settings['model']}")

    def normalize_provider_values(provider: str, api_mode: str, base_url: str) -> tuple[str, str]:
        effective_base_url = base_url.strip()
        effective_api_mode = api_mode
        if provider == "ollama" and effective_api_mode == "responses":
            effective_api_mode = "ollama_native"
        if provider == "ollama" and not effective_base_url:
            if effective_api_mode == "chat":
                effective_base_url = "http://localhost:11434/v1"
            else:
                effective_base_url = "http://localhost:11434/api/chat"
                effective_api_mode = "ollama_native"
        if provider == "openai" and not effective_base_url:
            effective_base_url = ""
        return effective_api_mode, effective_base_url

    def connection_settings(
        *,
        provider: str,
        api_mode: str,
        base_url: str,
        model: str,
        voice_model: str,
        api_key: str,
        keep_existing_key: bool,
        reasoning_effort: str,
        max_output_tokens: int,
        include_secret: bool,
    ) -> dict | None:
        effective_api_mode, effective_base_url = normalize_provider_values(provider, api_mode, base_url)
        if provider == "custom" and not effective_base_url:
            st.error("API endpoint is required for a custom OpenAI-compatible API.")
            return None
        if not model.strip():
            st.error("Article model is required.")
            return None
        saved = get_user_model_settings_for_provider(CONFIG, user["id"], provider, include_secret=include_secret)
        effective_key = api_key
        if not effective_key and keep_existing_key:
            effective_key = saved.get("api_key", "") if include_secret else ("saved" if saved["has_api_key"] else "")
        if not effective_key and provider != "ollama":
            st.error("API key is required unless you keep an existing saved key.")
            return None
        return {
            "provider": provider,
            "api_key": effective_key,
            "base_url": effective_base_url or None,
            "model": model.strip(),
            "voice_model": (voice_model or model).strip(),
            "api_mode": effective_api_mode,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": int(max_output_tokens),
        }

    def render_provider_tab(provider: str) -> None:
        settings = get_user_model_settings_for_provider(CONFIG, user["id"], provider)
        is_active = active_settings["provider"] == provider
        if is_active:
            st.success("This connection is active.")
        elif settings["updated_at"]:
            st.caption("Saved, but not active.")
        else:
            st.caption("Not configured yet.")
        if settings["has_api_key"]:
            st.caption("An API key is saved for this connection. It is encrypted at rest and is never displayed.")

        if provider == "openai":
            api_modes = ["responses", "chat"]
            endpoint_placeholder = "https://api.openai.com/v1"
        elif provider == "ollama":
            api_modes = ["ollama_native", "chat"]
            endpoint_placeholder = "http://localhost:11434/api/chat or http://localhost:11434/v1"
        else:
            api_modes = ["chat"]
            endpoint_placeholder = "https://your-provider.example/v1"

        api_mode_index = api_modes.index(settings["api_mode"]) if settings["api_mode"] in api_modes else 0
        reasoning_options = ["none", "low", "medium", "high"]
        reasoning_index = (
            reasoning_options.index(settings["reasoning_effort"])
            if settings["reasoning_effort"] in reasoning_options
            else 2
        )

        with st.form(f"model_settings_{provider}"):
            api_mode = st.selectbox(
                "API mode",
                api_modes,
                index=api_mode_index,
                format_func=lambda value: {
                    "responses": "OpenAI Responses API",
                    "chat": "OpenAI-compatible Chat Completions",
                    "ollama_native": "Native Ollama Chat API",
                }[value],
                key=f"api-mode-{provider}",
            )
            base_url = st.text_input(
                "API endpoint",
                value=settings["base_url"] or "",
                placeholder=endpoint_placeholder,
                key=f"base-url-{provider}",
            )
            model = st.text_input("Article model", value=settings["model"], key=f"model-{provider}")
            voice_model = st.text_input("Voice Print model", value=settings["voice_model"], key=f"voice-model-{provider}")
            api_key = st.text_input(
                "API key",
                type="password",
                placeholder="Leave blank to keep saved key" if settings["has_api_key"] else "Paste your API key",
                disabled=provider == "ollama",
                key=f"api-key-{provider}",
            )
            keep_existing_key = st.checkbox(
                "Keep saved key",
                value=settings["has_api_key"],
                disabled=not settings["has_api_key"] or provider == "ollama",
                key=f"keep-key-{provider}",
            )
            reasoning_effort = st.selectbox(
                "Reasoning effort",
                reasoning_options,
                index=reasoning_index,
                key=f"reasoning-{provider}",
            )
            max_output_tokens = st.number_input(
                "Max output tokens",
                min_value=1000,
                max_value=100000,
                value=int(settings["max_output_tokens"] or CONFIG.max_output_tokens),
                step=1000,
                key=f"max-tokens-{provider}",
            )
            test_submitted = st.form_submit_button("Test connection")
            save_submitted = st.form_submit_button("Save and make active", type="primary")

        if test_submitted:
            test_settings = connection_settings(
                provider=provider,
                api_mode=api_mode,
                base_url=base_url,
                model=model,
                voice_model=voice_model,
                api_key=api_key,
                keep_existing_key=keep_existing_key,
                reasoning_effort=reasoning_effort,
                max_output_tokens=int(max_output_tokens),
                include_secret=True,
            )
            if test_settings:
                try:
                    reply = ArticleGenerator(CONFIG, test_settings).test_connection()
                    st.success(f"Connection works. Model replied: {reply[:240]}")
                except Exception as exc:
                    st.error(f"Connection test failed: {exc}")

        if save_submitted:
            save_settings = connection_settings(
                provider=provider,
                api_mode=api_mode,
                base_url=base_url,
                model=model,
                voice_model=voice_model,
                api_key=api_key,
                keep_existing_key=keep_existing_key,
                reasoning_effort=reasoning_effort,
                max_output_tokens=int(max_output_tokens),
                include_secret=False,
            )
            if save_settings:
                save_user_model_settings(
                    CONFIG,
                    user["id"],
                    provider=provider,
                    api_key=api_key or None,
                    keep_existing_key=keep_existing_key and not api_key,
                    base_url=save_settings["base_url"],
                    model=save_settings["model"],
                    voice_model=save_settings["voice_model"],
                    api_mode=save_settings["api_mode"],
                    reasoning_effort=save_settings["reasoning_effort"],
                    max_output_tokens=save_settings["max_output_tokens"],
                    active=True,
                )
                st.success(f"{provider_labels[provider]} saved and made active.")
                st.rerun()

    openai_tab, ollama_tab, custom_tab = st.tabs(["OpenAI", "Local Ollama", "OpenAI-compatible"])
    with openai_tab:
        render_provider_tab("openai")
    with ollama_tab:
        render_provider_tab("ollama")
    with custom_tab:
        render_provider_tab("custom")


def account_page(user: dict) -> None:
    st.header("Account")
    st.write(f"Username: **{user['username']}**")
    st.write(f"Role: **{user['role']}**")
    model_settings_page_section(user)
    st.divider()
    st.subheader("Password")
    with st.form("change_password"):
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Update password", type="primary")
    if submitted:
        ok_password, password_message = password_is_reasonable(new_password)
        if not ok_password:
            st.error(password_message)
        elif new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            update_password(CONFIG, user["id"], new_password)
            st.success("Password updated.")


def admin_page(user: dict) -> None:
    st.header("Admin")
    if user["role"] != "admin":
        st.error("Admin access required.")
        return

    create_tab, users_tab = st.tabs(["Create user", "Manage users"])

    with create_tab:
        with st.form("admin_create_user"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Temporary password", type="password")
            role = st.selectbox("Role", ["user", "admin"])
            active = st.checkbox("Active", value=True)
            submitted = st.form_submit_button("Create user", type="primary")
        if submitted:
            if not username.strip():
                st.error("Username is required.")
            elif not password:
                st.error("Password is required.")
            else:
                ok, message = create_user(CONFIG, username, email or None, password, role=role, active=active)
                st.success(message) if ok else st.error(message)

    with users_tab:
        users = list_users(CONFIG)
        for account in users:
            with st.expander(f"{account['username']} · {account['role']} · {'active' if account['active'] else 'disabled'}"):
                with st.form(f"user-form-{account['id']}"):
                    email = st.text_input("Email", value=account["email"] or "", key=f"email-{account['id']}")
                    role = st.selectbox(
                        "Role",
                        ["user", "admin"],
                        index=0 if account["role"] == "user" else 1,
                        key=f"role-{account['id']}",
                    )
                    active = st.checkbox("Active", value=bool(account["active"]), key=f"active-{account['id']}")
                    new_password = st.text_input("Reset password", type="password", key=f"reset-{account['id']}")
                    submitted = st.form_submit_button("Save changes")
                if submitted:
                    update_user(CONFIG, account["id"], email or None, role, active)
                    if new_password:
                        update_password(CONFIG, account["id"], new_password)
                    st.success("User updated.")
                    st.rerun()


def main() -> None:
    user = current_user()
    if not user:
        login_view()
        return
    page = sidebar(user)
    if page == "Draft Article":
        draft_article_page(user)
    elif page == "Voice Print":
        voice_print_page(user)
    elif page == "Draft Archive":
        archive_page(user)
    elif page == "Admin":
        admin_page(user)
    elif page == "Account":
        account_page(user)


if __name__ == "__main__":
    main()
