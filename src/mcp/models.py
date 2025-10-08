from datetime import date
from typing import List, Optional, TypeAlias

from pydantic_ai.format_prompt import format_as_xml
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)

from utils import html_to_text


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
                    return html_to_text(html_content)[:50] + "..."
        
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
            self.results, root_tag=self.__class__.__name__, include_field_info="once", item_tag=None,
        )


# Add Docket and related classes