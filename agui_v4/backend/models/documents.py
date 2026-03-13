"""Document source models shared across workflows and services."""

import datetime
import uuid
from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer
from pydantic.json_schema import SkipJsonSchema


class DocBaseConfig(BaseModel):
    """Base model for documents with common configuration."""

    model_config = ConfigDict(
        validate_by_alias=True,
        validate_by_name=True,
        json_schema_extra={"additionalProperties": False},
        json_schema_serialization_defaults_required=True,
    )


class Document(DocBaseConfig):
    """A document related to a claim."""

    id: SkipJsonSchema[uuid.UUID] = Field(default_factory=uuid.uuid4)
    claim_number: str = Field(alias="claimNumber")
    content_id: str = Field(alias="contentId")
    mime_type: str = Field(alias="mimeType")
    content_url: str = Field(alias="contentURL")
    presigned_url: str = Field(alias="presignedURL", default="")
    domain: Literal["claim", "policy"] = "claim"
    document_type: Optional[str] = Field(alias="documentType", default=None)
    upload_time: Optional[str] = Field(alias="uploadTime", default=None)
    source_system: Optional[str] = Field(alias="sourceSystem", default=None)
    text: Optional[str] = ""
    content: SkipJsonSchema[Optional[bytes | str]] = None
    document_sub_type: str | None = Field(alias="documentSubType", default=None)
    document_description: str | None = Field(alias="documentDescription", default=None)
    create_date: datetime.datetime = Field(alias="createDate", default=None)
    company_name: str | None = Field(alias="companyName", default=None)
    list_of_contents: list[dict[str, str]] | None = Field(alias="listOfContents", default=None)
    token_count: Optional[int] = Field(alias="tokenCount", default=None)

    @field_serializer("create_date")
    def format_date(self, value: datetime.date) -> str:
        """Serialize create_date as a UTC string."""
        if value:
            return value.strftime("%Y-%m-%d %H:%M:%S %Z")

    @computed_field
    @property
    def file_name(self) -> str:
        """Derive the file name from content_url, falling back to content_id."""
        return Path(self.content_url).name if self.content_url else self.content_id

    def as_string(self) -> str:
        """Format the document details as a human-readable string."""
        return (
            f"Document Name: {self.file_name}\n"
            f"Type: {self.document_type or 'N/A'}\n"
            f"MIME Type: {self.mime_type}\n"
            f"Text: {self.text or 'No text content available'}"
        )


class Documents(DocBaseConfig):
    """A collection of documents related to a claim."""

    documents: list[Document] = Field(default_factory=list)

    @property
    def valid_ids(self) -> list[str]:
        """Get content IDs for all documents that have one."""
        return [doc.content_id for doc in self.documents if doc.content_id]

    def as_string(self) -> str:
        """Format all documents as a single string separated by dividers."""
        return (
            "\n----------\n".join(document.as_string() for document in self.documents)
            if self.documents
            else "No documents available."
        )

    def as_summary_string(self, max_len: int = 50) -> str:
        """Return truncated text for each document."""
        summaries = []
        for doc in self.documents:
            truncated_text = (
                (doc.text[:max_len] + "...") if (doc.text and len(doc.text) > max_len) else doc.text
            )
            summaries.append(
                f"Document Name: {doc.file_name}\n"
                f"Type: {doc.document_type or 'N/A'}\n"
                f"MIME Type: {doc.mime_type}\n"
                f"Text (truncated): {truncated_text or 'No text content available'}"
            )
        return "\n----------\n".join(summaries) if summaries else "No documents available."

    def as_metadata_string(self) -> str:
        """Return metadata-only representation sorted by create_date (newest first)."""
        self.documents.sort(key=lambda doc: doc.create_date, reverse=True)
        metadata_list = []
        for doc in self.documents:
            meta_string = ""
            meta_string += f"CONTENT ID: {doc.content_id}\n"
            meta_string += f"Created: {doc.create_date.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            meta_string += f"MIME Type: {doc.mime_type}\n"
            meta_string += f"Type: {doc.document_type or 'N/A'}\n"
            if doc.document_sub_type:
                meta_string += f"Sub-Type: {doc.document_sub_type}\n"
            if doc.document_description:
                meta_string += f"Description: {doc.document_description}\n"
            meta_string += f"Source System: {doc.source_system or 'N/A'}\n"
            if doc.company_name:
                meta_string += f"Company Name: {doc.company_name}\n"
            if doc.source_system == "GRMUC_ROOTS_PLCYPCKT" and doc.list_of_contents:
                meta_string += "Policy Forms:\n"
                for form in doc.list_of_contents:
                    form_id = form.get("formID", "N/A")
                    form_name = form.get("formName", "").strip()
                    if form_name:
                        meta_string += f"  • {form_id}: {form_name}\n"
                    else:
                        meta_string += f"  • {form_id}\n"
            metadata_list.append(meta_string)

        return "\n----------\n".join(metadata_list) if metadata_list else "No documents available."

    def get_doc_by_content_id(self, content_id: str) -> Optional[Document]:
        """Find a document by its content_id."""
        if content_id not in self.valid_ids:
            raise ValueError(
                f"Content ID {content_id} is not valid. Valid IDs are: {', '.join(self.valid_ids)}"
            )
        return next((doc for doc in self.documents if doc.content_id == content_id), None)
