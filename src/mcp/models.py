from datetime import date
from typing import List, Optional, TypeAlias

from pydantic_ai.format_prompt import format_as_xml
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)

try:
    from utils import html_to_text
except ImportError:
    from src.mcp.utils import html_to_text

COURTLISTENER_WEB_URL = "https://www.courtlistener.com"


class BaseFilteredModel(BaseModel):
    """Base model that filters out None values and empty lists."""

    @model_validator(mode="before")
    def empty_to_none(cls, data):
        """Convert empty lists to None."""
        for k, v in data.items():
            if v == "":
                data[k] = None
        return data

    @model_serializer(mode="wrap")
    def serialize_filtered(self, handler):
        """Exclude None values and empty lists from serialization."""
        result = handler(self)
        return {k: v for k, v in result.items() if v is not None and v != []}


class Meta(BaseFilteredModel):
    timestamp: Optional[str] = None
    date_created: Optional[str] = None


class Opinion(BaseFilteredModel):
    """
    Individual opinion entry within the case/opinion cluster.
    """

    id: int
    author_id: Optional[int] = None
    cites: List[int] = Field(default_factory=list)
    joined_by_ids: List[int] = Field(default_factory=list)
    snippet: Optional[str] = None
    type: Optional[str] = None


Opinions: TypeAlias = list[Opinion]


class OpinionSearchResult(BaseFilteredModel):
    absolute_url: Optional[str] = None
    attorney: Optional[str] = None
    caseName: Optional[str] = None
    caseNameFull: Optional[str] = None
    citation: List[str] = Field(default_factory=list)
    citeCount: int = 0
    cluster_id: int
    court: Optional[str] = None
    court_citation_string: Optional[str] = None
    court_id: Optional[str] = None

    dateArgued: Optional[date] = None
    dateFiled: Optional[date] = None
    dateReargued: Optional[date] = None
    dateReargumentDenied: Optional[date] = None

    docketNumber: Optional[str] = None
    docket_id: Optional[int] = None
    judge: Optional[str] = None
    lexisCite: Optional[str] = None
    neutralCite: Optional[str] = None

    non_participating_judge_ids: List[int] = Field(default_factory=list)
    opinions: Opinions = Field(default_factory=list)

    panel_ids: List[int] = Field(default_factory=list)
    panel_names: List[str] = Field(default_factory=list)

    posture: Optional[str] = None
    procedural_history: Optional[str] = None
    sibling_ids: List[int] = Field(default_factory=list)
    source: Optional[str] = None
    status: Optional[str] = None
    suitNature: Optional[str] = None
    syllabus: Optional[str] = None

    @computed_field
    @property
    def web_link(self) -> str:
        return f"{COURTLISTENER_WEB_URL}{self.absolute_url}"

    @property
    def primary_opinion(self) -> Optional[Opinion]:
        return self.opinions[0] if self.opinions else None

    @property
    def primary_opinion_id(self) -> Optional[int]:
        return self.primary_opinion.id if self.primary_opinion else None


class OpinionSearchResults(BaseFilteredModel):
    count: int
    next: Optional[str] = None
    previous: Optional[str] = None
    results: list[OpinionSearchResult]

    def to_xml(self) -> str:
        return format_as_xml(
            self.results, root_tag=self.__class__.__name__, include_field_info="once"
        )


class Position(BaseFilteredModel):
    appointer: Optional[str] = None
    court: Optional[str] = None
    court_citation_string: Optional[str] = None
    court_exact: Optional[str] = None
    court_full_name: Optional[str] = None
    date_confirmation: Optional[str] = None
    date_elected: Optional[str] = None
    date_hearing: Optional[str] = None
    date_judicial_committee_action: Optional[str] = None
    date_nominated: Optional[str] = None
    date_recess_appointment: Optional[str] = None
    date_referred_to_judicial_committee: Optional[str] = None
    date_retirement: Optional[str] = None
    date_start: Optional[str] = None
    date_termination: Optional[str] = None
    job_title: Optional[str] = None
    judicial_committee_action: Optional[str] = None
    nomination_process: Optional[str] = None
    organization_name: Optional[str] = None
    position_type: Optional[str] = None
    predecessor: Optional[str] = None
    selection_method: Optional[str] = None
    selection_method_id: Optional[str] = None
    supervisor: Optional[str] = None
    termination_reason: Optional[str] = None


class PersonSearchResult(BaseFilteredModel):
    id: int
    aba_rating: Optional[List[str]] = None
    absolute_url: Optional[str] = None
    alias: Optional[List[str]] = None
    alias_ids: Optional[List[int]] = None
    dob: Optional[str] = None
    dob_city: Optional[str] = None
    dob_state: Optional[str] = None
    dob_state_id: Optional[str] = None
    dod: Optional[str] = None
    fjc_id: Optional[str] = None
    gender: Optional[str] = None
    meta: Optional[Meta] = None
    name: Optional[str] = None
    political_affiliation: Optional[List[str]] = None
    political_affiliation_id: Optional[List[str]] = None
    positions: Optional[List[Position]] = None
    races: Optional[List[str]] = None
    religion: Optional[str] = None
    school: Optional[List[str]] = None

    @computed_field
    @property
    def web_link(self) -> str:
        return f"{COURTLISTENER_WEB_URL}{self.absolute_url}"


class PersonSearchResults(BaseFilteredModel):
    count: int
    next: Optional[str] = None
    previous: Optional[str] = None
    results: list[PersonSearchResult]

    def to_xml(self) -> str:
        return format_as_xml(
            self.results, root_tag=self.__class__.__name__, include_field_info="once"
        )


class Opinion(BaseFilteredModel):
    id: int
    resource_uri: Optional[str] = None
    absolute_url: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster: Optional[str] = None
    author_id: Optional[int] = None
    author: Optional[str] = None
    author_str: Optional[str] = None
    joined_by_str: Optional[str] = None
    per_curiam: Optional[bool] = None
    type: Optional[str]
    page_count: Optional[int] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    download_url: Optional[str] = None
    local_path: Optional[str] = None

    @computed_field
    @property
    def web_link(self) -> str:
        return f"{COURTLISTENER_WEB_URL}{self.absolute_url}"

    plain_text: Optional[str] = Field(
        default=None,
        exclude=True,
    )
    html: Optional[str] = Field(
        default=None,
        exclude=True,
    )
    html_lawbox: Optional[str] = Field(
        default=None,
        exclude=True,
    )
    html_columbia: Optional[str] = Field(
        default=None,
        exclude=True,
    )
    html_anon_2020: Optional[str] = Field(
        default=None,
        exclude=True,
    )
    xml_harvard: Optional[str] = Field(
        default=None,
        exclude=True,
    )
    html_with_citations: Optional[str] = Field(
        default=None,
        exclude=True,
    )
    opinions_cited: Optional[List[str]] = Field(
        default=None,
        exclude=True,
    )
    full_text_flag: Optional[bool] = Field(
        default=None,
        exclude=True,
    )

    @field_validator("date_created", "date_modified")
    def parse_date(cls, v: str) -> Optional[str]:
        """Parse datetime string to ISO format date string."""
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]

    @computed_field
    def opinions_cited_ids(self) -> List[int]:
        ids = []
        for opinion_id in self.opinions_cited:
            chunks = opinion_id.split("/")
            identifier = chunks[-2]
            ids.append(int(identifier))
        return ids

    @computed_field
    def text(self) -> str:
        # First try plain text (return as-is)
        if self.plain_text is not None and self.full_text_flag:
            return self.plain_text

        if self.plain_text is not None and not self.full_text_flag:
            return self.plain_text[:50] + "..."

        # Try HTML fields (convert to text)
        html_fields = [
            self.html,
            self.html_lawbox,
            self.html_columbia,
            self.html_anon_2020,
            self.html_with_citations,
        ]

        for html_content in html_fields:
            if html_content is not None:
                if self.full_text_flag:
                    return html_to_text(html_content)
                else:
                    return html_to_text(html_content)[:150] + "..."

        # Fall back to XML (return as-is, assuming it's text-like)
        if self.xml_harvard is not None:
            if self.full_text_flag:
                return self.xml_harvard
            else:
                return self.xml_harvard[:150] + "..."

        # Return empty string if no content available
        return ""

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
            item_tag="cite",
        )


class OpinionExcerpt(BaseFilteredModel):
    score: float
    index_range: str
    text: str


class OpinionExcerpts(BaseFilteredModel):
    id: str
    excerpts: list[OpinionExcerpt] | None | str = None

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
        )


class PersonSource(BaseFilteredModel):
    resource_uri: Optional[str] = None
    id: Optional[int] = None
    person: Optional[str] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    url: Optional[str] = None
    date_accessed: Optional[str] = None  # Added missing field
    notes: Optional[str] = None

    @field_validator("date_created", "date_modified")
    def parse_date(cls, v: str) -> Optional[str]:
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]


class AbaRating(BaseFilteredModel):
    """Model for ABA rating entries."""

    resource_uri: Optional[str] = None
    id: Optional[int] = None
    person: Optional[str] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    year_rated: Optional[int] = None
    rating: Optional[str] = None

    @field_validator("date_created", "date_modified")
    def parse_date(cls, v: str) -> Optional[str]:
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]


class School(BaseFilteredModel):
    """Model for educational institution."""

    resource_uri: Optional[str] = None
    id: Optional[int] = None
    is_alias_of: Optional[int] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    name: Optional[str] = None
    ein: Optional[int] = None

    @field_validator("date_created", "date_modified")
    def parse_date(cls, v: str) -> Optional[str]:
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]


class Education(BaseFilteredModel):
    """Model for education entries."""

    resource_uri: Optional[str] = None
    id: Optional[int] = None
    school: Optional[School] = None
    person: Optional[str] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    degree_level: Optional[str] = None
    degree_detail: Optional[str] = None
    degree_year: Optional[int] = None

    @field_validator("date_created", "date_modified")
    def parse_date(cls, v: str) -> Optional[str]:
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]


class PoliticalAffiliation(BaseFilteredModel):
    """Model for political affiliation entries."""

    resource_uri: Optional[str] = None
    id: Optional[int] = None
    person: Optional[str] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    political_party: Optional[str] = None
    source: Optional[str] = None
    date_start: Optional[str] = None
    date_granularity_start: Optional[str] = None
    date_end: Optional[str] = None
    date_granularity_end: Optional[str] = None

    @field_validator("date_created", "date_modified")
    def parse_date(cls, v: str) -> Optional[str]:
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]


class Person(BaseFilteredModel):
    id: int

    resource_uri: Optional[str] = None
    race: Optional[List[str]] = None
    sources: Optional[List[PersonSource]] = None
    aba_ratings: Optional[List[AbaRating]] = None
    educations: Optional[List[Education]] = None
    positions: Optional[List[str]] = None
    political_affiliations: Optional[List[PoliticalAffiliation]] = None
    is_alias_of: Optional[int] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    date_completed: Optional[str] = None
    fjc_id: Optional[int] = None
    slug: Optional[str] = None
    name_first: Optional[str] = None
    name_middle: Optional[str] = None
    name_last: Optional[str] = None
    name_suffix: Optional[str] = None
    date_dob: Optional[str] = None
    date_granularity_dob: Optional[str] = None
    date_dod: Optional[str] = None
    date_granularity_dod: Optional[str] = None
    dob_city: Optional[str] = None
    dob_state: Optional[str] = None
    dob_country: Optional[str] = None
    dod_city: Optional[str] = None
    dod_state: Optional[str] = None
    dod_country: Optional[str] = None
    gender: Optional[str] = None
    religion: Optional[str] = None
    ftm_total_received: Optional[float] = None
    ftm_eid: Optional[str] = None
    has_photo: Optional[bool] = None

    @field_validator(
        "date_created", "date_modified", "date_completed", "date_dob", "date_dod"
    )
    def parse_date(cls, v: str) -> Optional[str]:
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]

    def to_xml(self) -> str:
        return format_as_xml(
            self, root_tag=self.__class__.__name__, include_field_info="once"
        )


class DocketMeta(BaseFilteredModel):
    """
    Metadata for docket search results.

    Attributes:
        timestamp: ISO timestamp of when the result was generated
        date_created: ISO timestamp of when the docket was created
        score: Search relevance scoring information
    """

    timestamp: Optional[str] = None
    date_created: Optional[str] = None
    score: Optional[dict] = None

    @field_validator("timestamp", "date_created")
    def parse_date(cls, v: str) -> Optional[str]:
        """Parse datetime string to ISO format date string."""
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]


class DocketSearchResult(BaseFilteredModel):
    """
    Individual docket search result from CourtListener API.

    Represents a single case/docket with all associated metadata including
    parties, attorneys, court information, and case details.
    """

    # Core identifiers
    docket_id: int
    docket_absolute_url: Optional[str] = None
    docketNumber: Optional[str] = None
    pacer_case_id: Optional[str] = None

    # Case information
    caseName: Optional[str] = None
    case_name_full: Optional[str] = None
    cause: Optional[str] = None
    suitNature: Optional[str] = None
    chapter: Optional[str] = None

    # Court information
    court: Optional[str] = None
    court_citation_string: Optional[str] = None
    court_id: Optional[str] = None

    # Dates
    dateArgued: Optional[str] = None
    dateFiled: Optional[str] = None
    dateTerminated: Optional[str] = None

    # Judges and assignment
    assignedTo: Optional[str] = None
    assigned_to_id: Optional[int] = None
    referredTo: Optional[str] = None
    referred_to_id: Optional[int] = None

    # Parties
    party: List[str] = Field(default_factory=list)
    party_id: List[int] = Field(default_factory=list)

    # Attorneys and firms
    attorney: List[str] = Field(default_factory=list)
    attorney_id: List[int] = Field(default_factory=list)
    firm: List[str] = Field(default_factory=list)
    firm_id: List[int] = Field(default_factory=list)

    # Case characteristics
    jurisdictionType: Optional[str] = None
    juryDemand: Optional[str] = None
    trustee_str: Optional[str] = None

    # Metadata
    meta: Optional[DocketMeta] = None

    @computed_field
    @property
    def web_link(self) -> str:
        return f"{COURTLISTENER_WEB_URL}{self.docket_absolute_url}"


class DocketSearchResults(BaseFilteredModel):
    """
    Collection of docket search results from CourtListener API.

    Contains pagination information and a list of docket results.
    """

    count: int
    next: Optional[str] = None
    previous: Optional[str] = None
    results: List[DocketSearchResult]

    def to_xml(self) -> str:
        """
        Convert the docket search results to XML format.

        Returns:
            str: XML representation of the search results
        """
        return format_as_xml(
            self.results,
            root_tag=self.__class__.__name__,
            include_field_info="once",
            item_tag=None,
        )


class Docket(BaseFilteredModel):
    """Represents a CourtListener docket record."""

    resource_uri: str
    id: int
    court: Optional[str]
    court_id: Optional[str]
    original_court_info: Optional[str]
    idb_data: Optional[str]
    clusters: list[str]
    audio_files: list[str]
    assigned_to: Optional[str]
    referred_to: Optional[str]
    absolute_url: str
    date_created: str
    date_modified: str
    source: Optional[int]
    appeal_from_str: Optional[str]
    assigned_to_str: Optional[str]
    referred_to_str: Optional[str]
    panel_str: Optional[str]
    date_last_index: Optional[str]
    date_cert_granted: Optional[str]
    date_cert_denied: Optional[str]
    date_argued: Optional[str]
    date_reargued: Optional[str]
    date_reargument_denied: Optional[str]
    date_filed: Optional[str]
    date_terminated: Optional[str]
    date_last_filing: Optional[str]
    case_name_short: Optional[str]
    case_name: Optional[str]
    case_name_full: Optional[str]
    slug: Optional[str]
    docket_number: Optional[str]
    docket_number_core: Optional[str]
    docket_number_raw: Optional[str]
    federal_dn_office_code: Optional[str]
    federal_dn_case_type: Optional[str]
    federal_dn_judge_initials_assigned: Optional[str]
    federal_dn_judge_initials_referred: Optional[str]
    federal_defendant_number: Optional[int]
    pacer_case_id: Optional[str]
    cause: Optional[str]
    nature_of_suit: Optional[str]
    jury_demand: Optional[str]
    jurisdiction_type: Optional[str]
    appellate_fee_status: Optional[str]
    appellate_case_type_information: Optional[str]
    mdl_status: Optional[str]
    filepath_ia: Optional[str]
    filepath_ia_json: Optional[str]
    ia_upload_failure_count: Optional[int]
    ia_needs_upload: Optional[bool]
    ia_date_first_change: Optional[str]
    date_blocked: Optional[str]
    blocked: Optional[bool]
    appeal_from: Optional[str]
    parent_docket: Optional[str]
    tags: list[str]
    panel: list[str]

    @computed_field
    @property
    def web_link(self) -> str:
        return f"{COURTLISTENER_WEB_URL}{self.absolute_url}"

    def __str__(self) -> str:
        """Concise string useful for LLM or UI summaries."""
        return (
            f"Docket {self.id} | {self.case_name or 'Unknown Case'} "
            f"({self.web_link})"
            f"({self.court_id or 'N/A'}) — {self.docket_number or ''} | "
            f"Filed: {self.date_filed or 'N/A'} | "
            f"Judge: {self.assigned_to_str or 'N/A'} | "
            f"Cause: {self.cause or 'N/A'}"
        )

    def to_xml(self) -> str:
        """
        Convert the docket to XML format.

        Returns:
            str: XML representation of the docket
        """
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
            item_tag=None,
        )


class ScoreInfo(BaseModel):
    """Scoring details, typically from BM25 or other relevance ranking."""

    bm25: Optional[float] = Field(None, description="BM25 score or ranking metric.")


class MetaInfo(BaseModel):
    """Metadata for the search result, including timestamps and scoring."""

    timestamp: Optional[str] = Field(
        None, description="Timestamp of the search result."
    )
    date_created: Optional[str] = Field(
        None, description="Original creation date of the record."
    )
    score: Optional[ScoreInfo] = Field(
        None, description="Scoring information such as BM25."
    )


class RECAPSearchResult(BaseModel):
    """Represents a RECAP docket-level search result."""

    assignedTo: Optional[str]
    assigned_to_id: Optional[int]
    attorney: Optional[List[str]]
    attorney_id: Optional[List[int]]
    caseName: Optional[str]
    case_name_full: Optional[str]
    cause: Optional[str]
    chapter: Optional[str]
    court: Optional[str]
    court_citation_string: Optional[str]
    court_id: Optional[str]
    dateArgued: Optional[str]
    dateFiled: Optional[str]
    dateTerminated: Optional[str]
    docketNumber: Optional[str]
    docket_absolute_url: Optional[str]
    docket_id: Optional[int]
    firm: Optional[List[str]]
    firm_id: Optional[List[int]]
    jurisdictionType: Optional[str]
    juryDemand: Optional[str]
    meta: Optional[MetaInfo]
    pacer_case_id: Optional[str]
    party: Optional[List[str]]
    party_id: Optional[List[int]]
    referredTo: Optional[str]
    referred_to_id: Optional[int]
    suitNature: Optional[str]
    trustee_str: Optional[str]

    @computed_field
    @property
    def web_link(self) -> str:
        return f"{COURTLISTENER_WEB_URL}{self.docket_absolute_url}"

    def __str__(self) -> str:
        """
        Compact text representation suitable for LLM consumption.
        Focuses on case identification, key parties, attorneys, and docket info.
        """
        parts = [
            f"Case: {self.caseName or 'Unknown'} ({self.docketNumber or 'No docket'})",
            f"({self.web_link})",
            f"Court: {self.court_citation_string or self.court_id or 'N/A'}",
            f"Filed: {self.dateFiled or 'N/A'}",
            f"Terminated: {self.dateTerminated or 'N/A'}",
        ]
        if self.attorney:
            parts.append(
                f"Attorneys: {', '.join(self.attorney[:3])}{'...' if len(self.attorney) > 3 else ''}"
            )
        if self.firm:
            parts.append(
                f"Firms: {', '.join(self.firm[:2])}{'...' if len(self.firm) > 2 else ''}"
            )
        if self.party:
            parts.append(
                f"Parties: {', '.join(self.party[:3])}{'...' if len(self.party) > 3 else ''}"
            )
        return " | ".join(parts)

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
        )


class RECAPSearchResults(BaseModel):
    count: int
    next: Optional[str] = None
    previous: Optional[str] = None
    results: List[RECAPSearchResult]

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
        )


class PartyRepresented(BaseModel):
    """Represents a party the attorney is linked to in a docket."""

    role: Optional[int] = None
    docket: Optional[str] = None
    party: Optional[str] = None
    date_action: Optional[str] = None

    @field_validator("date_action")
    def parse_date(cls, v: str) -> Optional[str]:
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]

    def __str__(self) -> str:
        return f"Role: {self.role}, Docket: {self.docket}, Party: {self.party}"


class Attorney(BaseModel):
    """Represents an attorney in the CourtListener dataset."""

    resource_uri: Optional[str] = None
    id: int
    parties_represented: Optional[List[PartyRepresented]] = []
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    name: Optional[str] = None
    contact_raw: Optional[str] = None
    phone: Optional[str] = None
    fax: Optional[str] = None
    email: Optional[str] = None

    @field_validator("date_created", "date_modified")
    def parse_date(cls, v: str) -> Optional[str]:
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]

    def __str__(self) -> str:
        """Concise summary useful for LLM responses or logs."""
        parts_repr = (
            f"{len(self.parties_represented)} represented"
            if self.parties_represented
            else "No represented parties"
        )
        return (
            f"Attorney {self.name or 'Unknown'} "
            f"({self.email or 'no email'}) — {parts_repr}, "
            f"Phone: {self.phone or 'N/A'}"
        )

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
        )


class PacerFilingSearchResult(BaseFilteredModel):
    """Represents a PACER filing document from the CourtListener dataset."""

    id: int
    absolute_url: Optional[str] = None
    attachment_number: Optional[int] = None
    cites: List[int] = Field(default_factory=list)
    description: Optional[str] = None
    docket_entry_id: Optional[int] = None
    docket_id: Optional[int] = None
    document_number: Optional[int] = None
    document_type: Optional[str] = None
    entry_date_filed: Optional[date] = None
    entry_number: Optional[int] = None
    filepath_local: Optional[str] = None
    is_available: Optional[bool] = None
    meta: Optional[Meta] = None
    pacer_doc_id: Optional[str] = None
    page_count: Optional[int] = None
    short_description: Optional[str] = None
    snippet: Optional[str] = None

    @computed_field
    @property
    def web_link(self) -> str:
        return f"{COURTLISTENER_WEB_URL}{self.absolute_url}"

    def __str__(self) -> str:
        """Concise summary useful for LLM responses or logs."""
        return (
            f"PACER Filing #{self.document_number or 'N/A'} "
            f"({self.web_link})"
            f"(ID: {self.id}) — {self.short_description or 'No description'}, "
            f"Filed: {self.entry_date_filed or 'Unknown'}, "
            f"Pages: {self.page_count or 0}"
        )

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
        )


class PacerFilingSearchResults(BaseFilteredModel):
    count: int
    next: Optional[str] = None
    previous: Optional[str] = None
    results: List[PacerFilingSearchResult]

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
        )


class OralArgumentSearchResult(BaseFilteredModel):
    """
    Oral argument audio recording search result from CourtListener.
    """

    id: int
    absolute_url: Optional[str] = None
    caseName: Optional[str] = None
    case_name_full: Optional[str] = None
    court: Optional[str] = None
    court_citation_string: Optional[str] = None
    court_id: Optional[str] = None

    dateArgued: Optional[date] = None
    dateReargued: Optional[date] = None
    dateReargumentDenied: Optional[date] = None

    docketNumber: Optional[str] = None
    docket_id: Optional[int] = None
    download_url: Optional[str] = None
    duration: Optional[int] = None
    file_size_mp3: Optional[int] = None
    judge: Optional[str] = None
    local_path: Optional[str] = None
    meta: Optional[Meta] = None
    pacer_case_id: Optional[str] = None
    panel_ids: List[int] = Field(default_factory=list)
    sha1: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

    @computed_field
    @property
    def web_link(self) -> str:
        return f"{COURTLISTENER_WEB_URL}{self.absolute_url}"

    @computed_field
    @property
    def mp3_url(self) -> str:
        local_path_mp3 = self.local_path
        if local_path_mp3:
            base_url = COURTLISTENER_WEB_URL
            return f"{base_url}/{local_path_mp3}"
        else:
            return "No MP3 URL available"

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
        )


class OralArgumentSearchResults(BaseFilteredModel):
    count: int
    next: Optional[str] = None
    previous: Optional[str] = None
    results: List[OralArgumentSearchResult]

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
        )


class OralArgument(BaseFilteredModel):
    """
    Represents an oral argument record from the CourtListener API.
    """

    id: int
    resource_uri: Optional[str] = None
    absolute_url: Optional[str] = None
    panel: Optional[list[str]] = Field(default_factory=list)
    docket: Optional[str] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    source: Optional[str] = None
    case_name_short: Optional[str] = None
    case_name: Optional[str] = None
    case_name_full: Optional[str] = None
    judges: Optional[str] = None
    sha1: Optional[str] = None
    download_url: Optional[str] = None
    local_path_mp3: Optional[str] = None
    local_path_original_file: Optional[str] = None
    filepath_ia: Optional[str] = None
    ia_upload_failure_count: Optional[int] = None
    duration: Optional[int] = None
    processing_complete: Optional[bool] = None
    date_blocked: Optional[str] = None
    blocked: Optional[bool] = None
    stt_status: Optional[int] = None
    stt_source: Optional[int] = None
    stt_transcript: Optional[str] = None

    @field_validator("date_created", "date_modified")
    def parse_date(cls, v: str) -> Optional[str]:
        if v is None:
            return None
        # Parse the datetime string and extract just the date portion
        return v.split("T")[0]

    def to_xml(self) -> str:
        return format_as_xml(
            self,
            root_tag=self.__class__.__name__,
            include_field_info="once",
        )
