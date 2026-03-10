"""Persistence service for saving and loading audit forms."""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any


class FormStore:
    """Manage local JSON-backed persistence for audit forms."""

    def __init__(self, forms_dir: str) -> None:
        """Store the forms directory and ensure it exists.

        Args:
            forms_dir: Absolute path to the local forms directory.
        """
        self.forms_dir = forms_dir
        os.makedirs(self.forms_dir, exist_ok=True)

    def iso_now(self) -> str:
        """Return the current UTC timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()

    def form_file_path(self, form_id: str) -> str:
        """Build a safe JSON file path for a form identifier.

        Args:
            form_id: Form identifier.

        Returns:
            Full path to the backing JSON file.
        """
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", form_id)
        return os.path.join(self.forms_dir, f"{safe_id}.json")

    def atomic_write_json(self, path: str, payload: dict[str, Any]) -> None:
        """Write JSON atomically by replacing the destination file.

        Args:
            path: Destination file path.
            payload: JSON-serializable payload.
        """
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)

    def validate_form_payload(self, payload: dict[str, Any]) -> str | None:
        """Validate the canonical audit form payload.

        Args:
            payload: Candidate audit form payload.

        Returns:
            `None` when valid, otherwise an error message.
        """
        required_fields = [
            "peril",
            "questions",
            "overall_outcome",
            "outcome_justification",
        ]
        missing = [field for field in required_fields if field not in payload]
        if missing:
            return f"Missing required fields: {', '.join(missing)}"

        if not isinstance(payload.get("questions"), list):
            return "Field 'questions' must be a list."

        if not isinstance(payload.get("peril"), dict):
            return "Field 'peril' must be an object."

        return None

    def build_form_title(self, form_payload: dict[str, Any]) -> str:
        """Build a fallback saved-form title.

        Args:
            form_payload: Canonical form payload.

        Returns:
            Human-friendly title string.
        """
        peril = form_payload.get("peril", {}).get("peril", "Unknown")
        outcome = form_payload.get("overall_outcome", "Unknown")
        return f"{peril} - {outcome} - {self.iso_now()[:10]}"

    def save_form(
        self,
        form_id: str,
        payload: dict[str, Any],
        title: str | None = None,
        source_docs: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Persist an audit form and return the stored record.

        Args:
            form_id: Form identifier to save.
            payload: Canonical audit form payload.
            title: Optional explicit title.
            source_docs: Optional source document metadata.

        Returns:
            Persisted form record.
        """
        path = self.form_file_path(form_id)
        existing_created_at = None
        if os.path.exists(path):
            try:
                existing = self.read_form(form_id)
                existing_created_at = existing.get("created_at")
            except Exception:
                existing_created_at = None

        record = {
            "id": form_id,
            "schema_version": "1.0",
            "created_at": existing_created_at or self.iso_now(),
            "updated_at": self.iso_now(),
            "title": title or self.build_form_title(payload),
            "source_docs": source_docs or [],
            "peril": payload["peril"],
            "questions": payload["questions"],
            "overall_outcome": payload["overall_outcome"],
            "outcome_justification": payload["outcome_justification"],
            "additional_analysis": payload.get("additional_analysis"),
            "follow_ups": payload.get("follow_ups"),
        }
        self.atomic_write_json(path, record)
        return record

    def list_forms(self) -> list[dict[str, Any]]:
        """Return lightweight summaries for all saved forms."""
        forms: list[dict[str, Any]] = []
        for name in os.listdir(self.forms_dir):
            if not name.endswith(".json"):
                continue
            file_path = os.path.join(self.forms_dir, name)
            try:
                with open(file_path, "r", encoding="utf-8") as file_obj:
                    data = json.load(file_obj)
                forms.append(
                    {
                        "id": data.get("id"),
                        "title": data.get("title"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "peril": data.get("peril", {}).get("peril"),
                        "overall_outcome": data.get("overall_outcome"),
                        "question_count": len(data.get("questions", [])),
                    }
                )
            except Exception as exc:
                print(f"[FORMS] Failed reading {file_path}: {exc}", flush=True)
        forms.sort(key=lambda form: form.get("updated_at") or "", reverse=True)
        return forms

    def list_forms_full(self) -> list[dict[str, Any]]:
        """Return full saved-form records sorted newest-first."""
        forms: list[dict[str, Any]] = []
        for name in os.listdir(self.forms_dir):
            if not name.endswith(".json"):
                continue
            file_path = os.path.join(self.forms_dir, name)
            try:
                with open(file_path, "r", encoding="utf-8") as file_obj:
                    forms.append(json.load(file_obj))
            except Exception as exc:
                print(f"[FORMS] Failed reading {file_path}: {exc}", flush=True)
        forms.sort(key=lambda form: form.get("updated_at") or "", reverse=True)
        return forms

    def read_form(self, form_id: str) -> dict[str, Any]:
        """Read one saved form record.

        Args:
            form_id: Form identifier.

        Returns:
            Saved record dictionary.

        Raises:
            FileNotFoundError: If the record does not exist.
        """
        path = self.form_file_path(form_id)
        if not os.path.exists(path):
            raise FileNotFoundError(form_id)
        with open(path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def delete_form(self, form_id: str) -> None:
        """Delete one saved form record.

        Args:
            form_id: Form identifier.

        Raises:
            FileNotFoundError: If the record does not exist.
        """
        path = self.form_file_path(form_id)
        if not os.path.exists(path):
            raise FileNotFoundError(form_id)
        os.remove(path)

    def to_form_payload(self, record: dict[str, Any]) -> dict[str, Any]:
        """Normalize a saved form record into the canonical in-memory payload."""
        return {
            "peril": record.get("peril", {}),
            "questions": record.get("questions", []),
            "overall_outcome": record.get("overall_outcome", ""),
            "outcome_justification": record.get("outcome_justification", ""),
            "additional_analysis": record.get("additional_analysis"),
            "follow_ups": record.get("follow_ups"),
        }
