from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests
from openai import OpenAI

from .config import AppConfig


ProgressCallback = Callable[[str], None]
TextDeltaCallback = Callable[[str], None]


class OpenAIConfigurationError(RuntimeError):
    pass


class OpenAIResponseIncompleteError(RuntimeError):
    def __init__(self, reason: str, partial_text: str):
        super().__init__(f"OpenAI response incomplete: {reason}")
        self.reason = reason
        self.partial_text = partial_text


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    api_key: str
    base_url: str | None
    model: str
    voice_model: str
    api_mode: str
    reasoning_effort: str
    max_output_tokens: int

    @classmethod
    def from_user_settings(cls, settings: dict[str, Any]) -> "ModelSettings":
        model = str(settings.get("model") or "").strip()
        return cls(
            provider=str(settings.get("provider") or "openai"),
            api_key=str(settings.get("api_key") or ""),
            base_url=str(settings.get("base_url") or "").strip() or None,
            model=model,
            voice_model=str(settings.get("voice_model") or model).strip(),
            api_mode=str(settings.get("api_mode") or "chat"),
            reasoning_effort=str(settings.get("reasoning_effort") or "medium").strip().lower(),
            max_output_tokens=int(settings.get("max_output_tokens") or 30000),
        )

    @property
    def label(self) -> str:
        if self.provider == "ollama":
            return "Ollama"
        if self.provider == "custom":
            return "OpenAI-compatible API"
        return "OpenAI"


class ArticleGenerator:
    def __init__(self, config: AppConfig, model_settings: dict[str, Any]):
        self.config = config
        self.model_settings = ModelSettings.from_user_settings(model_settings)
        if not self.model_settings.model:
            raise OpenAIConfigurationError("Choose a model in Account settings before generating.")
        if not self.model_settings.api_key and self.model_settings.provider != "ollama":
            raise OpenAIConfigurationError("Add your API key in Account settings before generating.")
        self.client = None
        if self.model_settings.api_mode != "ollama_native":
            client_kwargs = {"api_key": self.model_settings.api_key or "ollama"}
            if self.model_settings.base_url:
                client_kwargs["base_url"] = _openai_compatible_base_url(self.model_settings.base_url)
            self.client = OpenAI(**client_kwargs)

    def test_connection(self) -> str:
        text = self._responses_create(
            model=self.model_settings.model,
            instructions="Reply with a short confirmation that the model connection works.",
            content=[{"type": "input_text", "text": "Connection test. Reply with: connection ok."}],
            max_output_tokens=64,
        )
        return text.strip() or "Connection succeeded."

    def _responses_create(
        self,
        *,
        model: str,
        instructions: str,
        content: list[dict],
        max_output_tokens: int | None = None,
        on_progress: ProgressCallback | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> str:
        token_budget = max_output_tokens or self.model_settings.max_output_tokens
        if self.model_settings.api_mode == "ollama_native":
            return self._ollama_native_create(
                model=model,
                instructions=instructions,
                content=content,
                on_progress=on_progress,
                on_text_delta=on_text_delta,
            )
        if self.model_settings.api_mode == "chat":
            return self._chat_completions_create(
                model=model,
                instructions=instructions,
                content=content,
                max_output_tokens=token_budget,
                on_progress=on_progress,
                on_text_delta=on_text_delta,
            )

        kwargs = {
            "model": model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": token_budget,
        }
        if self.model_settings.reasoning_effort and self.model_settings.reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": self.model_settings.reasoning_effort}

        if on_progress or on_text_delta:
            return self._responses_stream(
                kwargs=kwargs,
                on_progress=on_progress,
                on_text_delta=on_text_delta,
            )

        response = self.client.responses.create(**kwargs)
        text = _extract_response_text(response)
        _raise_if_response_not_complete(response, text)
        return text

    def _chat_completions_create(
        self,
        *,
        model: str,
        instructions: str,
        content: list[dict],
        max_output_tokens: int,
        on_progress: ProgressCallback | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": _chat_content(content)},
        ]
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
        }
        if on_progress:
            on_progress(f"Connecting to {self.model_settings.label}.")

        if on_text_delta:
            deltas: list[str] = []
            finish_reason = None
            stream = self.client.chat.completions.create(**kwargs, stream=True)
            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                delta = getattr(choice.delta, "content", None) or ""
                if delta:
                    deltas.append(delta)
                    on_text_delta(delta)
                finish_reason = choice.finish_reason or finish_reason
            text = "".join(deltas).strip()
            if finish_reason == "length":
                raise OpenAIResponseIncompleteError("length", text)
            if on_progress:
                on_progress(f"{self.model_settings.label} response completed.")
            return text

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0] if response.choices else None
        text = choice.message.content.strip() if choice and choice.message.content else ""
        if choice and choice.finish_reason == "length":
            raise OpenAIResponseIncompleteError("length", text)
        return text

    def _ollama_native_create(
        self,
        *,
        model: str,
        instructions: str,
        content: list[dict],
        on_progress: ProgressCallback | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> str:
        text_content, images = _ollama_content_and_images(content)
        user_message: dict[str, Any] = {"role": "user", "content": text_content}
        if images:
            user_message["images"] = images
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                user_message,
            ],
            "stream": bool(on_text_delta),
        }
        url = _ollama_native_url(self.model_settings.base_url)
        if on_progress:
            on_progress("Connecting to Ollama.")

        try:
            response = requests.post(url, json=payload, stream=bool(on_text_delta), timeout=(10, None))
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed at {url}: {exc}") from exc

        if on_text_delta:
            deltas: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                message = event.get("message") or {}
                delta = message.get("content") or ""
                if delta:
                    deltas.append(delta)
                    on_text_delta(delta)
                if event.get("done"):
                    break
            if on_progress:
                on_progress("Ollama response completed.")
            return "".join(deltas).strip()

        data = response.json()
        message = data.get("message") or {}
        text = str(message.get("content") or "").strip()
        if on_progress:
            on_progress("Ollama response completed.")
        return text

    def _responses_stream(
        self,
        *,
        kwargs: dict,
        on_progress: ProgressCallback | None,
        on_text_delta: TextDeltaCallback | None,
    ) -> str:
        deltas: list[str] = []
        incomplete_response = None

        if on_progress:
            on_progress(f"Connecting to {self.model_settings.label}.")
        with self.client.responses.stream(**kwargs) as stream:
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.created" and on_progress:
                    on_progress(f"{self.model_settings.label} accepted the request.")
                elif event_type == "response.in_progress" and on_progress:
                    on_progress(f"{self.model_settings.label} is analyzing the material.")
                elif event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        deltas.append(delta)
                        if on_text_delta:
                            on_text_delta(delta)
                elif event_type == "response.output_text.done" and on_progress:
                    on_progress(f"{self.model_settings.label} finished writing text.")
                elif event_type == "response.completed" and on_progress:
                    on_progress(f"{self.model_settings.label} response completed.")
                elif event_type == "response.incomplete":
                    incomplete_response = getattr(event, "response", None)
                    if on_progress:
                        on_progress(f"{self.model_settings.label} stopped early; preparing continuation.")
                    break
                elif event_type in {"response.failed", "error"}:
                    message = getattr(event, "message", None)
                    response = getattr(event, "response", None)
                    if response is not None:
                        message = getattr(response, "error", None) or getattr(response, "status", None) or message
                    raise RuntimeError(message or f"OpenAI stream ended with {event_type}.")

            if incomplete_response is not None:
                partial_text = _extract_response_text(incomplete_response) or "".join(deltas)
                raise OpenAIResponseIncompleteError(_incomplete_reason(incomplete_response), partial_text)

            response = stream.get_final_response()

        text = _extract_response_text(response)
        _raise_if_response_not_complete(response, text)
        return text

    def generate_voice_print(
        self,
        *,
        sample_context: str,
        image_inputs: list[tuple[str, str]],
        author_notes: str,
        sample_count: int,
        on_progress: ProgressCallback | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> str:
        instructions = """
You are a senior newsroom editor and style analyst. Create a reusable Voice Print for an authorized journalist from supplied samples.

The Voice Print must be practical drafting guidance, not a literary imitation. Extract durable patterns in structure, pacing, paragraphing, sourcing, evidence handling, lede style, transitions, sentence rhythm, vocabulary level, quote handling, skepticism, and endings.

Safety and integrity rules:
- Do not copy memorable phrases from the samples.
- Do not instruct future drafts to invent facts, quotes, sources, scenes, or statistics.
- Preserve attribution discipline and flag uncertainty.
- Keep the guidance usable as a prompt wrapper for first-draft article generation.
""".strip()
        prompt = f"""
Analyze {sample_count} uploaded sample file(s) and produce a Voice Print.

Author notes:
{author_notes.strip() or "None provided."}

Return the result in this exact structure:

# Voice Print
## Editorial Identity
## Article Architecture
## Sentence and Paragraph Rhythm
## Tone, Distance, and Point of View
## Reporting and Attribution Habits
## Vocabulary and Diction
## Lede Patterns
## Transitions and Endings
## Do
## Do Not
## Prompt Wrapper

The Prompt Wrapper section must be concise instructions that can be inserted into future generation prompts.

Sample text:
{sample_context or "[No extractable text. Use image inputs if provided.]"}
""".strip()
        content = [{"type": "input_text", "text": prompt}]
        for name, data_url in image_inputs:
            content.append({"type": "input_text", "text": f"Image sample: {name}"})
            content.append({"type": "input_image", "image_url": data_url, "detail": "auto"})
        return self._responses_create(
            model=self.model_settings.voice_model,
            instructions=instructions,
            content=content,
            max_output_tokens=max(self.config.max_output_tokens, 9000),
            on_progress=on_progress,
            on_text_delta=on_text_delta,
        )

    def generate_article(
        self,
        *,
        title: str,
        source_context: str,
        image_inputs: list[tuple[str, str]],
        user_instructions: str,
        word_count: int,
        tone_strength: int,
        voice_print: str | None,
        include_editor_notes: bool,
        on_progress: ProgressCallback | None = None,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> str:
        word_count = max(250, min(self.config.article_max_words, int(word_count)))
        if word_count >= self.config.article_parallel_threshold_words:
            return self._generate_article_parallel(
                title=title,
                source_context=source_context,
                image_inputs=image_inputs,
                user_instructions=user_instructions,
                word_count=word_count,
                tone_strength=tone_strength,
                voice_print=voice_print,
                include_editor_notes=include_editor_notes,
                on_progress=on_progress,
                on_text_delta=on_text_delta,
            )

        try:
            return self._generate_article_single(
                title=title,
                source_context=source_context,
                image_inputs=image_inputs,
                user_instructions=user_instructions,
                word_count=word_count,
                tone_strength=tone_strength,
                voice_print=voice_print,
                include_editor_notes=include_editor_notes,
                on_progress=on_progress,
                on_text_delta=on_text_delta,
            )
        except OpenAIResponseIncompleteError as exc:
            if on_progress:
                on_progress("Draft stopped early; asking OpenAI to continue.")
            return self._continue_article(
                partial_article=exc.partial_text,
                title=title,
                source_context=source_context,
                user_instructions=user_instructions,
                word_count=word_count,
                tone_strength=tone_strength,
                voice_print=voice_print,
                include_editor_notes=include_editor_notes,
                on_progress=on_progress,
                on_text_delta=on_text_delta,
            )

    def _generate_article_single(
        self,
        *,
        title: str,
        source_context: str,
        image_inputs: list[tuple[str, str]],
        user_instructions: str,
        word_count: int,
        tone_strength: int,
        voice_print: str | None,
        include_editor_notes: bool,
        on_progress: ProgressCallback | None,
        on_text_delta: TextDeltaCallback | None,
    ) -> str:
        instructions, prompt = self._article_prompt(
            title=title,
            source_context=source_context,
            user_instructions=user_instructions,
            word_count=word_count,
            tone_strength=tone_strength,
            voice_print=voice_print,
            include_editor_notes=include_editor_notes,
        )
        content = [{"type": "input_text", "text": prompt}]
        for name, data_url in image_inputs:
            content.append({"type": "input_text", "text": f"Source image: {name}"})
            content.append({"type": "input_image", "image_url": data_url, "detail": "auto"})
        return self._responses_create(
            model=self.model_settings.model,
            instructions=instructions,
            content=content,
            max_output_tokens=self._article_output_budget(word_count),
            on_progress=on_progress,
            on_text_delta=on_text_delta,
        )

    def _generate_article_parallel(
        self,
        *,
        title: str,
        source_context: str,
        image_inputs: list[tuple[str, str]],
        user_instructions: str,
        word_count: int,
        tone_strength: int,
        voice_print: str | None,
        include_editor_notes: bool,
        on_progress: ProgressCallback | None,
        on_text_delta: TextDeltaCallback | None,
    ) -> str:
        section_count = max(2, math.ceil(word_count / max(500, self.config.article_section_target_words)))
        section_count = min(section_count, 12)
        workers = max(1, min(self.config.article_parallel_max_workers, section_count))

        if on_progress:
            on_progress(f"Building source brief and {section_count}-section plan.")
        plan = self._build_parallel_plan(
            title=title,
            source_context=source_context,
            image_inputs=image_inputs,
            user_instructions=user_instructions,
            word_count=word_count,
            tone_strength=tone_strength,
            voice_print=voice_print,
            section_count=section_count,
        )
        sections = _normalize_sections(plan.get("sections"), section_count, word_count)
        source_brief = str(plan.get("source_brief") or source_context[:12000])
        verification_gaps = plan.get("verification_gaps") or []
        outline_text = _outline_text(sections)

        if on_progress:
            on_progress(f"Parallel drafting {len(sections)} section(s) with {workers} worker(s).")

        completed: dict[int, str] = {}
        ordered_flush_index = 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._draft_article_section,
                    title=title,
                    user_instructions=user_instructions,
                    tone_strength=tone_strength,
                    voice_print=voice_print,
                    source_brief=source_brief,
                    outline_text=outline_text,
                    section=section,
                    section_count=len(sections),
                ): section
                for section in sections
            }
            for future in as_completed(futures):
                section = futures[future]
                section_number = int(section["number"])
                try:
                    completed[section_number] = future.result()
                except OpenAIResponseIncompleteError as exc:
                    completed[section_number] = exc.partial_text
                if on_progress:
                    on_progress(f"Section {section_number}/{len(sections)} drafted.")

                while ordered_flush_index in completed:
                    text = completed[ordered_flush_index]
                    if on_text_delta:
                        on_text_delta(("\n\n" if ordered_flush_index > 1 else "") + text)
                    ordered_flush_index += 1

        article = "\n\n".join(completed[index] for index in range(1, len(sections) + 1)).strip()
        word_total = _count_words(article)
        minimum, maximum = _word_range(word_count, self.config.article_word_tolerance_pct)
        if word_total < minimum:
            if on_progress:
                on_progress(f"Draft is short at {word_total:,} words; extending toward target.")
            article = self._continue_article(
                partial_article=article,
                title=title,
                source_context=source_brief,
                user_instructions=user_instructions,
                word_count=word_count,
                tone_strength=tone_strength,
                voice_print=voice_print,
                include_editor_notes=False,
                on_progress=on_progress,
                on_text_delta=on_text_delta,
            )
        elif word_total > maximum and on_progress:
            on_progress(f"Draft is above target range at {word_total:,} words; leaving full copy intact.")

        if include_editor_notes and verification_gaps:
            notes = "\n".join(f"- {gap}" for gap in verification_gaps[:12])
            article = f"{article}\n\n## Editor Notes\n{notes}"

        return article.strip()

    def _build_parallel_plan(
        self,
        *,
        title: str,
        source_context: str,
        image_inputs: list[tuple[str, str]],
        user_instructions: str,
        word_count: int,
        tone_strength: int,
        voice_print: str | None,
        section_count: int,
    ) -> dict[str, Any]:
        minimum, maximum = _word_range(word_count, self.config.article_word_tolerance_pct)
        instructions = """
You are a senior assigning editor. Build a factual source brief and section plan for a long-form first draft.

Use only the uploaded/source material and explicit user instructions. Do not invent facts, quotes, names, dates, statistics, or scenes.
Return JSON only.
""".strip()
        prompt = f"""
Article title or topic:
{title.strip() or "Untitled draft"}

Target length:
{word_count} words, acceptable range {minimum}-{maximum} words.

Number of sections:
{section_count}

User instructions:
{user_instructions.strip() or "Write the strongest news-style first draft supported by the uploaded material."}

Voice Print strength:
{tone_strength}/100

Voice Print:
{voice_print.strip() if voice_print else "No Voice Print selected. Use a neutral professional newsroom voice."}

Source material:
{source_context or "[No extractable text. Use image inputs if provided.]"}

Return JSON in this shape:
{{
  "source_brief": "A compact but detailed factual brief with key claims, timeline, named entities, quotes, data, caveats, and image observations.",
  "verification_gaps": ["Important uncertainty or missing source item"],
  "sections": [
    {{
      "number": 1,
      "heading": "Section heading",
      "purpose": "What this section accomplishes",
      "target_words": 900,
      "must_include": ["Fact or point from the source brief"]
    }}
  ]
}}
""".strip()
        content = [{"type": "input_text", "text": prompt}]
        for name, data_url in image_inputs:
            content.append({"type": "input_text", "text": f"Source image: {name}"})
            content.append({"type": "input_image", "image_url": data_url, "detail": "auto"})
        try:
            raw = self._responses_create(
                model=self.model_settings.model,
                instructions=instructions,
                content=content,
                max_output_tokens=max(7000, min(self.config.max_output_tokens, 14000)),
            )
            return _json_object(raw)
        except Exception:
            return {
                "source_brief": source_context[:20000],
                "verification_gaps": ["Plan generation failed; verify structure and source coverage manually."],
                "sections": [],
            }

    def _draft_article_section(
        self,
        *,
        title: str,
        user_instructions: str,
        tone_strength: int,
        voice_print: str | None,
        source_brief: str,
        outline_text: str,
        section: dict[str, Any],
        section_count: int,
    ) -> str:
        section_number = int(section["number"])
        target_words = int(section["target_words"])
        minimum, maximum = _word_range(target_words, self.config.article_word_tolerance_pct)
        heading = str(section["heading"])
        purpose = str(section["purpose"])
        must_include = section.get("must_include") or []
        first_last = "This is the opening section; include the lede." if section_number == 1 else ""
        if section_number == section_count:
            first_last = "This is the closing section; include a strong ending without a separate editor notes section."

        instructions = f"""
You are a newsroom drafting assistant writing one section of a longer article.

Rules:
- Write only section {section_number} of {section_count}.
- Target {target_words} words, acceptable range {minimum}-{maximum} words.
- Use only the source brief and user instructions.
- Do not invent facts, quotes, names, dates, statistics, or scenes.
- Match the Voice Print at strength {tone_strength}/100 without copying sample phrases.
- Keep transitions compatible with the full outline.
- Do not include editor notes.
{first_last}
""".strip()
        prompt = f"""
Full article title/topic:
{title.strip() or "Untitled draft"}

User instructions:
{user_instructions.strip() or "Write the strongest news-style first draft supported by the uploaded material."}

Voice Print:
{voice_print.strip() if voice_print else "No Voice Print selected. Use a neutral professional newsroom voice."}

Full outline:
{outline_text}

Section to write:
Number: {section_number}
Heading: {heading}
Purpose: {purpose}
Must include: {json.dumps(must_include, ensure_ascii=False)}

Source brief:
{source_brief}

Return this section only. Use a markdown heading for the section unless this is section 1 and the lede reads better without one.
""".strip()
        try:
            return self._responses_create(
                model=self.model_settings.model,
                instructions=instructions,
                content=[{"type": "input_text", "text": prompt}],
                max_output_tokens=max(4000, int(target_words * 4) + 2000),
            )
        except OpenAIResponseIncompleteError as exc:
            continuation = self._continue_section(
                partial_section=exc.partial_text,
                source_brief=source_brief,
                section=section,
                section_count=section_count,
            )
            return f"{exc.partial_text}\n\n{continuation}".strip()

    def _continue_section(
        self,
        *,
        partial_section: str,
        source_brief: str,
        section: dict[str, Any],
        section_count: int,
    ) -> str:
        prompt = f"""
Continue this incomplete section from where it stopped. Do not restart or repeat earlier paragraphs.

Section {section["number"]} of {section_count}: {section["heading"]}
Target section words: {section["target_words"]}

Partial section:
{partial_section}

Source brief:
{source_brief}
""".strip()
        return self._responses_create(
            model=self.model_settings.model,
            instructions="Continue an incomplete article section without repeating previous text.",
            content=[{"type": "input_text", "text": prompt}],
            max_output_tokens=max(3000, int(section["target_words"]) * 3 + 2000),
        )

    def _continue_article(
        self,
        *,
        partial_article: str,
        title: str,
        source_context: str,
        user_instructions: str,
        word_count: int,
        tone_strength: int,
        voice_print: str | None,
        include_editor_notes: bool,
        on_progress: ProgressCallback | None,
        on_text_delta: TextDeltaCallback | None,
    ) -> str:
        article = partial_article.strip()
        minimum, _ = _word_range(word_count, self.config.article_word_tolerance_pct)
        for attempt in range(1, 4):
            current_words = _count_words(article)
            if current_words >= minimum:
                break
            remaining_words = max(400, min(1800, word_count - current_words))
            if on_progress:
                on_progress(f"Continuation pass {attempt}: {current_words:,}/{word_count:,} words drafted.")
            notes_line = (
                "Add editor notes only after the article is complete."
                if include_editor_notes and attempt == 3
                else "Do not add editor notes in this continuation."
            )
            prompt = f"""
Continue the incomplete article from exactly where it stopped. Do not restart, summarize, or repeat prior paragraphs.

Title/topic:
{title.strip() or "Untitled draft"}

Target total length:
{word_count} words.

Current draft:
{article}

Write about {remaining_words} additional words.

User instructions:
{user_instructions.strip() or "Write the strongest news-style first draft supported by the uploaded material."}

Voice Print strength:
{tone_strength}/100

Voice Print:
{voice_print.strip() if voice_print else "No Voice Print selected. Use a neutral professional newsroom voice."}

Source material:
{source_context}

{notes_line}
""".strip()
            try:
                addition = self._responses_create(
                    model=self.model_settings.model,
                    instructions="Continue an incomplete newsroom article draft without repeating previous copy.",
                    content=[{"type": "input_text", "text": prompt}],
                    max_output_tokens=max(2500, int(remaining_words * 3)),
                )
            except OpenAIResponseIncompleteError as exc:
                addition = exc.partial_text
            if not addition.strip():
                break
            article = f"{article}\n\n{addition.strip()}"
            if on_text_delta:
                on_text_delta("\n\n" + addition.strip())
        return article.strip()

    def _article_prompt(
        self,
        *,
        title: str,
        source_context: str,
        user_instructions: str,
        word_count: int,
        tone_strength: int,
        voice_print: str | None,
        include_editor_notes: bool,
    ) -> tuple[str, str]:
        closeness = _voice_closeness_label(tone_strength)
        minimum, maximum = _word_range(word_count, self.config.article_word_tolerance_pct)
        instructions = f"""
You are an expert newsroom first-draft assistant. Generate a clean, usable article draft from the provided source material and instructions.

Editorial requirements:
- Use only facts supported by uploaded material or explicit user instructions.
- Never fabricate quotes, names, dates, statistics, locations, or source claims.
- If a claim is important but uncertain, mark it for verification instead of presenting it as fact.
- Keep the piece publication-ready in structure but clearly treat it as a first draft.
- Use AP-style clarity where no user instruction conflicts.
- Target {word_count} words and stay within {minimum}-{maximum} words unless source material is too thin.
- Voice Print strength is {tone_strength}/100 ({closeness}).
- When a Voice Print is supplied, use its general drafting guidance without copying distinctive phrases from samples.
""".strip()
        notes_instruction = (
            "After the article, add a short 'Editor Notes' section listing verification gaps and assumptions."
            if include_editor_notes
            else "Return only the article draft, with no notes section."
        )
        prompt = f"""
Article title or topic:
{title.strip() or "Untitled draft"}

Target word count:
{word_count} words. Acceptable range: {minimum}-{maximum} words.

User instructions:
{user_instructions.strip() or "Write the strongest news-style first draft supported by the uploaded material."}

Voice Print:
{voice_print.strip() if voice_print else "No Voice Print selected. Use a neutral professional newsroom voice."}

Source material:
{source_context or "[No extractable text. Use image inputs if provided.]"}

Output instruction:
{notes_instruction}
""".strip()
        return instructions, prompt

    def _article_output_budget(self, word_count: int) -> int:
        estimated = int(word_count * 3.2) + 5000
        return max(self.config.max_output_tokens, estimated)


def _extract_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content_item in getattr(item, "content", []) or []:
            value = getattr(content_item, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts).strip()


def _chat_content(content: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for item in content:
        item_type = item.get("type")
        if item_type == "input_text":
            converted.append({"type": "text", "text": item.get("text", "")})
        elif item_type == "input_image":
            converted.append({"type": "image_url", "image_url": {"url": item.get("image_url", "")}})
    return converted


def _ollama_content_and_images(content: list[dict]) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    images: list[str] = []
    for item in content:
        item_type = item.get("type")
        if item_type == "input_text":
            text_parts.append(str(item.get("text") or ""))
        elif item_type == "input_image":
            data_url = str(item.get("image_url") or "")
            if "," in data_url:
                images.append(data_url.split(",", 1)[1])
    return "\n\n".join(part for part in text_parts if part.strip()), images


def _ollama_native_url(base_url: str | None) -> str:
    url = (base_url or "http://localhost:11434/api/chat").strip().rstrip("/")
    if url.endswith("/api/chat"):
        return url
    if url.endswith("/api"):
        return f"{url}/chat"
    return f"{url}/api/chat"


def _openai_compatible_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        return url[: -len(suffix)]
    return url


def _raise_if_response_not_complete(response, partial_text: str) -> None:
    status = getattr(response, "status", None)
    if status == "incomplete":
        raise OpenAIResponseIncompleteError(_incomplete_reason(response), partial_text)
    if status == "failed":
        error = getattr(response, "error", None)
        raise RuntimeError(error or "OpenAI response failed.")


def _incomplete_reason(response) -> str:
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    if reason:
        return str(reason)
    return "incomplete"


def _json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object.")
    return value


def _normalize_sections(raw_sections: Any, section_count: int, word_count: int) -> list[dict[str, Any]]:
    if not isinstance(raw_sections, list):
        raw_sections = []
    targets = _section_word_targets(word_count, section_count)
    fallback_headings = [
        "The Stakes",
        "What Happened",
        "The Background",
        "The Evidence",
        "The People Affected",
        "The Response",
        "What Comes Next",
        "The Wider Context",
        "The Unanswered Questions",
        "The Ending",
    ]
    sections: list[dict[str, Any]] = []
    for index in range(section_count):
        source = raw_sections[index] if index < len(raw_sections) and isinstance(raw_sections[index], dict) else {}
        heading = str(source.get("heading") or fallback_headings[index % len(fallback_headings)])
        purpose = str(source.get("purpose") or f"Develop section {index + 1} of the article.")
        must_include = source.get("must_include")
        if not isinstance(must_include, list):
            must_include = []
        sections.append(
            {
                "number": index + 1,
                "heading": heading,
                "purpose": purpose,
                "target_words": targets[index],
                "must_include": must_include,
            }
        )
    return sections


def _section_word_targets(word_count: int, section_count: int) -> list[int]:
    base = word_count // section_count
    remainder = word_count % section_count
    return [base + (1 if index < remainder else 0) for index in range(section_count)]


def _outline_text(sections: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{section['number']}. {section['heading']} ({section['target_words']} words): {section['purpose']}"
        for section in sections
    )


def _word_range(word_count: int, tolerance_pct: int) -> tuple[int, int]:
    tolerance = max(0, tolerance_pct) / 100
    return int(word_count * (1 - tolerance)), int(word_count * (1 + tolerance))


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _voice_closeness_label(value: int) -> str:
    if value < 35:
        return "light touch"
    if value < 70:
        return "balanced"
    return "close"
