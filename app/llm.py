"""OpenRouter adapter for evidence-grounded bilingual medicine answers."""

from dataclasses import dataclass, field
import json
import os
import re
from pathlib import Path

import requests

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"
KANNADA_RE = re.compile(r"[\u0c80-\u0cff]")


def load_local_env(path=Path(__file__).resolve().parents[1] / ".env"):
    """Load simple KEY=VALUE entries without overriding exported variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()


@dataclass
class OpenRouterGenerator:
    api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    )
    endpoint: str = DEFAULT_ENDPOINT
    timeout: int = 90
    last_model: str | None = field(default=None, init=False)

    backend = "openrouter"

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")

    def answer(self, medicine_name, question, evidence):
        language = "Kannada" if KANNADA_RE.search(question or "") else "English"
        facts = "\n".join(
            f"- {key}: {value}" for key, value in evidence.items() if value
        )
        system_prompt = (
            f"Answer entirely in clear {language} using only the supplied medicine evidence. "
            "Translate descriptive English evidence into the requested language. Keep medicine "
            "names, manufacturer names, and scientific ingredient names unchanged when no safe "
            "translation exists. Never add a dose, diagnosis, interaction, contraindication, "
            "warning, or medical claim absent from the evidence. If evidence cannot answer, "
            f"state in {language} that the information is unavailable. Never reveal reasoning. "
            "Return JSON containing only the answer_text field."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Medicine: {medicine_name}\nQuestion: {question}\n"
                    f"Complete matched CSV row:\n{facts}"
                ),
            },
        ]
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 600,
            "reasoning": {"effort": "none", "exclude": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "grounded_medicine_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer_text": {"type": "string"}},
                        "required": ["answer_text"],
                        "additionalProperties": False,
                    },
                },
            },
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Kannada Medicine Assistant",
        }
        response = requests.post(
            self.endpoint, headers=headers, json=payload, timeout=self.timeout
        )
        # Some free routed models reject JSON Schema or reasoning controls. Retry once
        # with a plain-text contract while keeping the same evidence-only constraints.
        if getattr(response, "status_code", None) == 400:
            retry_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"response_format", "reasoning"}
            }
            retry_messages = [dict(item) for item in messages]
            retry_messages[0]["content"] = system_prompt.replace(
                "Return JSON containing only the answer_text field.",
                "Return only the final answer text without JSON or reasoning.",
            )
            retry_payload["messages"] = retry_messages
            response = requests.post(
                self.endpoint, headers=headers, json=retry_payload, timeout=self.timeout
            )
        response.raise_for_status()
        payload = response.json()
        self.last_model = payload.get("model")
        choices = payload.get("choices") or []
        raw_content = choices[0].get("message", {}).get("content") if choices else ""
        content = raw_content.strip() if isinstance(raw_content, str) else ""
        try:
            answer = json.loads(content)["answer_text"].strip()
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            answer = content
        valid_script = (
            bool(KANNADA_RE.search(answer))
            if language == "Kannada"
            else bool(re.search(r"[A-Za-z]", answer)) and not KANNADA_RE.search(answer)
        )
        if not answer or not valid_script:
            raise RuntimeError(f"OpenRouter did not return a valid {language} answer")
        return answer
