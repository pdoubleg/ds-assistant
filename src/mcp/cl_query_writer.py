from typing import Annotated, Dict, Literal, Optional
import os
import httpx
from textwrap import dedent
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pydantic_ai import Agent

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.server import logger

try:
    from utils import get_citation_context, get_context_with_bm25

except ImportError:
    from src.mcp.utils import get_citation_context, get_context_with_bm25
    
try:
    from models import (
        OralArgumentSearchResults,
        PacerFilingSearchResults,
        OpinionSearchResults,
        DocketSearchResults,
        PersonSearchResults,
        RECAPSearchResults,
        Attorney, Docket, Opinion, OpinionExcerpt, OpinionExcerpts, OralArgument, Person
    )
except ImportError:
    from src.mcp.models import (
        OralArgumentSearchResults,
        PacerFilingSearchResults,
        OpinionSearchResults,
        DocketSearchResults,
        PersonSearchResults,
        RECAPSearchResults,
        Attorney, Docket, Opinion, OpinionExcerpt, OpinionExcerpts, OralArgument, Person
    )

load_dotenv()

API_KEY = os.getenv("COURT_LISTENER_API_KEY")
if not API_KEY:
    raise ValueError("COURT_LISTENER_API_KEY not found in environment variables")

query_server = FastMCP(
    name="cl_query_writer",
    instructions=dedent("""\
        A set of tools for constructing and executing advanced search queries for the CourtListener API.
        Use the get_court_listener_query tool to construct a query or series of queries using an AI agent to address the user's request.
        Use the execute_court_listener_search_query tool to execute the query or series of queries and return the search results as an XML string.
        Note that get_court_listener_query may include multiple steps with some steps depending on the results of prior steps. In these cases
        you will need to use the execute_court_listener_search_query tool to get the results of the prior step and use those results to construct the next step. 
        Finally, use the "get tools", e.g., get_opinion, get_opinion_excerpt, get_opinion_excerpt_by_citation, get_person, get_docket, get_attorney, 
        get_oral_argument tools to retrieve the data you need.
    """),
)

class CourtListenerSearchQuery(BaseModel):
    """Search query or series of queries for the CourtListener API"""

    rationale: str = Field(
        ...,
        description=(
            "Brief explanation of the chosen type, core fielded terms/operators, "
            "sort choice, and any extra params. Include any additional instructions for the next step(s)."
        ),
    )
    step_number: int = Field(
        default=1,
        description=("The step number in the multi-step query. Defaults to 1."),
    )

    q: str = Field(
        ...,
        description=(
            "Lucene/eDisMax query string combining free text and fielded terms "
            "(camelCase fields), Boolean operators, ranges, phrases, wildcards, and "
            "advanced operators. Examples: "
            "'cites:(111722)', "
            "'court_id:ca1 AND status:published AND dateFiled:[2015-01-01 TO 2020-12-31]'."
        ),
    )
    type: Literal["o", "r", "rd", "d", "p", "oa"] = Field(
        default="o",
        description=(
            "Corpus selector: 'o' opinions, 'r' RECAP-style federal dockets, "
            "'rd' RECAP filing documents, 'd' PACER dockets, 'p' judges, 'oa' oral arguments."
        ),
    )
    order_by: Optional[str] = Field(
        None,
        description=(
            "Optional sort as 'field asc' or 'field desc'. Must be valid for the chosen type. "
            "Examples: 'dateFiled desc' (o), 'entry_date_filed desc' (rd), 'dateArgued asc' (oa), "
            "'dateTerminated desc' (r/d), 'name asc' (p). Omit if not specified."
        ),
    )
    params: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Optional extra GET parameters corresponding to sidebar filters when not easily expressed in `q`. "
            "Example for precedential-only opinions: {'stat_Precedential': 'on'}."
        ),
    )


QUERY_PROMPT = """\
You are a world class query-construction AI for the CourtListener Legal Search API.
Given a user’s request, produce a list of CourtListenerSearchQuery objects that address the request. For simple requests return a single object. \
For more multi-step requests return a list of sequential search queries that will address the request. For example, if the user wants to find opinions involving a \
specific firm or attorney, we first need to find the dockets associated with that firm or attorney, then get the party IDs, and then use those IDs to find the opinions. \
For any unknown values that depend on prior steps, use var_name wrapped in curly braces, e.g., {var_name}.

Output format (strict)
Return a list of CourtListenerSearchQuery objects with these keys:
- q (string, required) – The Lucene/eDisMax query string, including any fielded terms, Boolean operators, ranges, wildcards, and advanced operators.
- type (one of: o, r, rd, d, p, oa; required) – Search corpus:
    - o = opinion clusters (+ nested opinions)
    - r = federal dockets (with up to 3 nested docs, RECAP search UI style)
    - rd = federal filing documents from PACER (RECAP documents)
    - d = federal dockets from PACER
    - p = judges
    - oa = oral arguments
- order_by (string, optional) – One sortable field and direction: "field asc" or "field desc". Omit if not needed.
- rationale (string, required) – A short explanation of how you mapped the user’s request to q, type, and sorting/filters.

Optionally, if sidebar-style filters are required that cannot be expressed cleanly in q, you may also include:
- params (object, optional) – Extra GET params (e.g., {"stat_Precedential": "on"}).

Do not return a URL, curl, or code to run the request. Do not include authentication.

General construction rules:
1. Pick type first based on the user’s corpus intent:
    - Opinions/caselaw → o  
    - Federal dockets (case-level) → r or d
    - Federal filing documents (document-level RECAP) → rd
    - Judges/people → p
    - Oral arguments → oa

2. Prefer fielded queries in q for precision. Use camelCase field names provided by CourtListener’s search layer (e.g., caseName, docketNumber, court_id, dateFiled, citeCount, status, author_id, panel_ids, party, attorney, document_number, description, etc.).
Examples:
    - court_id:ca1 status:published         
    - dateFiled:[2018-01-01 TO 2018-12-31]
    - citation:("410 U.S. 113")
    - cites:(111722) (opinions that cite opinion id 111722)
    - firm:("Kirkland Ellis LLP") (dockets with Kirkland Ellis LLP as a firm)
    - attorney:("John Doe") (dockets with John Doe as an attorney)

3. Use Boolean logic & grouping to refine your query:
    - Default operator is AND, but make it explicit in complex queries: (term1 AND term2).
    - Use OR for unions; - or NOT/% for negation: immigration AND border AND -"border patrol".
    - Group thoughtfully with parentheses.

4. Phrases and exact matches
    - Use quotes for phrases or to avoid stemming/synonyms: "summary judgment", "Information" vs "Inform".  

5. Wildcards, fuzzy, proximity
    - * and ? for wildcards (observe CourtListener wildcard limits; avoid leading * on short terms).
    - Fuzzy terms with ~ (e.g., immigrant~1).
    - Proximity on phrases with ~k, e.g., "class certification"~4.

6. Ranges and dates
    - Numeric/date ranges with [] and TO, e.g., dateFiled:[2019-01-01 TO 2020-12-31], citeCount:[10 TO *].
    - Dates must be ISO-8601 YYYY-MM-DD.

7. Use sidebar/extra params sparingly
    - Prefer encoding filters in q.
    - If the website exposes sidebar switches (e.g., precedential) that map to GET params, you may set params, e.g., {"stat_Precedential": "on"} for opinions.

8. Sorting (order_by)
    - Only use fields valid for the chosen type. Typical, corpus-appropriate choices include:
        - o (opinions): dateFiled asc|desc, citeCount desc, sometimes relevance (omit direction if not supported)
        - rd (RECAP documents): entry_date_filed asc|desc, dateFiled asc|desc, page_count desc
        - r / d (dockets): dateFiled asc|desc, dateTerminated asc|desc
        - oa (oral arguments): dateArgued asc|desc
        - p (judges): name asc|desc, or position-date fields like date_start asc|desc if supported
    - If the user specifies “newest/oldest,” map to the appropriate date field and direction. If unspecified, omit order_by or use a sensible default for that corpus.

9. Validation & minimalism
    - Keep q compact but complete.
    - Avoid fields not supported by the selected type.
    - Don’t include null/empty filters.
    - Ensure parentheses and quotes are balanced.
    - Prefer structured q over free text.

Advanced operator reminders
    - Intersections: AND / &
    - Unions: OR
    - Negation: -term, NOT term, % term
    - Phrases: "..." (exact term behavior inside phrase)
    - Wildcards: *, ? (avoid disallowed patterns; no leading * on very short tokens)
    - Fuzzy: term~1 or term~2
    - Proximity: "phrase"~k
    - Ranges: field:[A TO B], * for open bound
    
Different search interfaces support **different fields** according to the following:

## OPINIONS
- id: CourtListener system ID
- docket_id: Associated docket ID 
- scdb_id: Supreme Court Database ID
- cluster_id: ID of opinion cluster (dissents, concurrences etc)
- sibling_ids: IDs of other opinions in cluster
- court_id: Abbreviated court ID (see jurisdictions page for abbreviations)
- attorney: Case attorneys
- author_id: Judge author ID
- panel_ids: Panel judge IDs
- panel_names: Panel judge names
- joined_by_ids: IDs of joining judges
- judge: Full-text searchable judge name (useful for judges without IDs)
- per_curiam: Whether per curiam opinion
- dateFiled: Decision issue date
- dateArgued: First argument date
- dateReargued: Reargument date
- dateReargumentDenied: Reargument denial date
- caseName: Case name
- docketNumber: Case docket number
- citation: All opinion citations
- neutralCite: Neutral citation if known
- lexisCite: LexisNexis citation
- suitNature: Nature of suit
- citeCount: Times cited (accepts range queries)
- status: Precedential status (valid: published, unpublished, errata, separate, in-chambers, relating-to, unknown)
- cites: IDs of citing opinions
- type: Opinion type (valid: combined-opinion, unanimous-opinion, lead-opinion, plurality-opinion, concurrence-opinion, in-part-opinion, dissent, addendum, remittitur, rehearing, on-the-merits, on-motion-to-strike)
- procedural_history: Court-to-court history
- posture: Procedural posture
- syllabus: Case summary and outcome
- non_participating_judge_ids: Non-participating judges

## PARENTHETICALS
- docket_id: Associated docket ID
- cluster_id: ID of opinion cluster (dissents, concurrences etc)
- court_id: Abbreviated court ID (see jurisdictions page)
- author_id: Judge author ID
- panel_ids: Panel judge IDs
- panel_names: Panel judge names
- judge: Full-text searchable judge name (for judges without IDs)
- dateFiled: Decision issue date
- caseName: Case name
- docketNumber: Case docket number
- citation: All opinion citations
- neutralCite: Neutral citation if known
- lexisCite: LexisNexis citation
- suitNature: Nature of suit
- citeCount: Times cited (accepts range queries)
- status: Precedential status (valid: published, unpublished, errata, separate, in-chambers, relating-to, unknown)
- cites: IDs of citing opinions

## RECAP
- id: CourtListener system ID
- docket_id: Associated docket ID
- docket_entry_id: Docket entry ID
- court_id: Abbreviated court ID (see jurisdictions page)
- party: Party names (multiple possible)*
- attorney: Attorney names (multiple possible)*
- firm_id: Law firm IDs (multiple possible)
- firm: Law firm names (multiple possible)
- assigned_to_id: Assigned judge IDs
- referred_to_id: Referred judge ID
- assignedTo: Assigned judge name (useful for judges without IDs)
- referredTo: Referred judge name (useful for judges without IDs)
- dateFiled: Case initiation date
- entry_date_filed: Document entry date
- dateArgued: First argument date
- dateTerminated: Case termination date
- caseName: Case name
- docketNumber: Docket number
- suitNature: Nature of suit
- document_type: Valid values: PACER Document or Attachment
- document_number: PACER docket document number
- attachment_number: PACER docket attachment number
- is_available: Whether in RECAP Archive
- page_count: Document page count (supports range queries)
- description: PACER document description
- short_description: Download page description
- cause: PACER cause of action
- juryDemand: PACER jury demand
- jurisdictionType: PACER jurisdiction (e.g. 'Diversity' or 'U.S. Government Defendant')
- chapter: Bankruptcy chapter
- trustee_str: Bankruptcy trustee name
- entry_number: PACER docket entry number/internal ID
- pacer_doc_id: Internal PACER document ID
- plain_text: Extracted document text

*Important Note: party and attorney fields are docket-level only. Avoid combining with non-docket fields in main query as it won't yield results. Instead use sidebar filters:
- Document Description: "description"
- Party Name: "party name" 
- Attorney Name: "attorney name"

## ORAL ARGUMENTS
- id: Item ID in CourtListener system
- docket_id: Associated docket ID 
- court_id: Abbreviated court ID
- panel_ids: Judge panel IDs for case
- judge: Judge name (searchable text field)
- dateArgued: First argument date
- dateReargued: Reargument date
- dateReargumentDenied: Date reargument denied
- caseName: Case name
- docketNumber: Case docket number

## JUDGES
- id: Judge ID in CourtListener system
- fjc_id: Federal Judicial Center ID
- name: Full name
- races: Known races
- gender: Gender identity (Female/Male/Other)
- religion: Known religion
- dob/dod: Birth/death dates
- birth_location: City, state, state ID of birth
- court: Courts where positions held
- position_type: Titles held (e.g. Chief Judge, Special Chairman)
- dates: Key dates including nomination, confirmation, start, retirement
- relationships: Appointer, supervisor, predecessor names
- selection: Nomination process and selection method
- background: School, political affiliation, ABA rating
"""


agent = Agent(
    model="openai:gpt-5-mini",
    output_type=list[CourtListenerSearchQuery],
    retries=5,
    deps_type=None,
    system_prompt=QUERY_PROMPT,
)


@query_server.tool()
async def get_court_listener_query(
    query: Annotated[
        str,
        Field(
            description="The user's request that we need to translate into a query or series of queries for the CourtListener API"
        ),
    ],
    ctx: Context | None = None,
) -> list[CourtListenerSearchQuery]:
    """Delegate a user query to a specialized query construction agent. This agent has access to the CourtListener advanced search options.

    Args:
        query: The user's request that we need to translate into a query or series of queries for the CourtListener API
        ctx: The context of the server session

    Returns:
        list[CourtListenerSearchQuery]: The query or series of queries for the CourtListener API
    """

    if ctx:
        await ctx.info(f"Getting CL query for with context: {query}")
    else:
        logger.info(f"Getting CL query for: {query}")
        
        result = await agent.run(query)

    return result.output


    
@query_server.tool()
async def execute_court_listener_search_query(
    q: Annotated[
        str,
        Field(description="The query string to execute"),
    ],
    type: Annotated[
        Literal["o", "r", "rd", "d", "p", "oa"],
        Field(
            description="The type of search to execute. Options are as follows: o = opinions, r = List of Federal cases (dockets), rd = Federal filing documents from PACER, d = Federal cases (dockets) from PACER, p = judges, oa = oral arguments"
        ),
    ],
    
    order_by: Annotated[
        str,
        Field(description="Sort by 'score desc', 'dateFiled desc', or 'dateFiled asc'"),
    ] = "score desc",
    
    params: Annotated[
        Dict[str, str] | None,
        Field(description="The params to use"),
    ] = None,
    
    limit: Annotated[
        int,
        Field(description="The limit to use"),
    ] = 10,
    ctx: Context | None = None,
) -> str:
    """Execute a single search query on the CourtListener API and return the results as an XML string.

    Args:
        q: The query string to execute
        type: The type of search to execute. Options are as follows: o = opinions, r = List of Federal cases (dockets), rd = Federal filing documents from PACER, d = Federal cases (dockets) from PACER, p = judges, oa = oral arguments
        order_by: The order by to use
        params: The params to use
        limit: The limit to use. Default is 10.
        ctx: The context of the server session

    Returns:
        str: The search results as returned by the CourtListener API
    """
    if ctx:
        await ctx.info(f"Executing search query with context: {q}")
    else:
        logger.info(f"Executing search query: {q}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}
    
    if not params:
        params = {}
    
    if limit:
        params["limit"] = limit

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.courtlistener.com/api/rest/v4/search/",
                params=dict(q=q, type=type, order_by=order_by, params=params),
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            data['results'] = data['results'][:limit]
            
            if type == "o":
                data = OpinionSearchResults(**data)
            elif type == "r":
                data = RECAPSearchResults(**data)
            elif type == "rd":
                data = PacerFilingSearchResults(**data)
            elif type == "d":
                data = DocketSearchResults(**data)
            elif type == "p":
                data = PersonSearchResults(**data)
            elif type == "oa":
                data = OralArgumentSearchResults(**data)
                
            return data.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error executing search query: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    
@query_server.tool()
async def get_opinion(
    opinion_id: Annotated[str, Field(description="The opinion ID to retrieve")],
    full_text: Annotated[
        bool,
        Field(
            default=False, description="Whether to return the full text of the opinion"
        ),
    ] = False,
    ctx: Context | None = None,
) -> str:
    """Get a specific court opinion by ID from CourtListener.

    Args:
        opinion_id: The opinion ID to retrieve.
        full_text: Whether to return the full text of the opinion. Default is False.

    Returns:
        str: The opinion data as returned by the CourtListener API.
    """

    if ctx:
        await ctx.info(f"Getting opinion with ID: {opinion_id}")
    else:
        logger.info(f"Getting opinion with ID: {opinion_id}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.courtlistener.com/api/rest/v4/opinions/{opinion_id}/",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            if ctx:
                await ctx.info(f"Successfully retrieved opinion {opinion_id}")
            else:
                logger.info(f"Successfully retrieved opinion {opinion_id}")

            result = Opinion(full_text_flag=full_text, **data)

            return result.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting opinion: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting opinion: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    
@query_server.tool()
async def get_opinion_excerpt(
    opinion_id: Annotated[str, Field(description="The opinion ID to retrieve")],
    search_query: Annotated[
        str, Field(description="The query to retrieve excerpt(s) from the opinion")
    ],
    ctx: Context | None = None,
) -> str:
    """Given a search query and opinion ID, retrieve excerpt(s) from the opinion text. Uses BM25 to retrieve excerpt(s).
    Useful for retrieving excerpts from a specific opinion that are relevant to a given search query.

    Args:
        opinion_id: The opinion ID to retrieve.
        search_query: The query to retrieve excerpts from the opinion.

    Returns:
        str: The opinion excerpts as returned by the CourtListener API.
    """

    if ctx:
        await ctx.info(
            f"Getting opinion with ID: {opinion_id} and query: {search_query}"
        )
    else:
        logger.info(f"Getting opinion with ID: {opinion_id} and query: {search_query}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.courtlistener.com/api/rest/v4/opinions/{opinion_id}/",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            if ctx:
                await ctx.info(f"Successfully retrieved opinion {opinion_id}")
            else:
                logger.info(f"Successfully retrieved opinion {opinion_id}")

            result = Opinion(full_text_flag=True, **data)

            bm25_results = get_context_with_bm25(
                search_query, result.text, 500, 1000, adjust_to_sentences=True
            )

            if bm25_results:
                excerpts = [
                    OpinionExcerpt(
                        score=round(score, 3),
                        index_range=f"{start}-{end}",
                        text=context,
                    )
                    for context, start, end, score in bm25_results
                ]

            else:
                excerpts = f"No excerpts found in opinion `{opinion_id}` for query `{search_query}`"

            opinion_excerpts = OpinionExcerpts(id=opinion_id, excerpts=excerpts)

            return opinion_excerpts.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting opinion: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting opinion: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    
@query_server.tool()
async def get_opinion_excerpt_by_citation(
    opinion_id: Annotated[str, Field(description="The opinion ID to retrieve")],
    citation: Annotated[
        str, Field(description="The citation to retrieve excerpt(s) from the opinion")
    ],
    ctx: Context | None = None,
) -> str:
    """Given an input citation and opinion ID, retrieve excerpt(s) from the opinion text. Uses citation lookup engine to detect citations.
    Useful for retrieving excerpts from a specific opinion concerning a given citation.

    Args:
        opinion_id: The opinion ID to retrieve.
        citation: The citation to retrieve excerpt(s) from the opinion.

    Returns:
        str: The opinion excerpt(s) as returned by the CourtListener API.
    """

    if ctx:
        await ctx.info(
            f"Getting opinion with ID: {opinion_id} and citation: {citation}"
        )
    else:
        logger.info(f"Getting opinion with ID: {opinion_id} and citation: {citation}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.courtlistener.com/api/rest/v4/opinions/{opinion_id}/",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            if ctx:
                await ctx.info(f"Successfully retrieved opinion {opinion_id}")
            else:
                logger.info(f"Successfully retrieved opinion {opinion_id}")

            result = Opinion(full_text_flag=True, **data)

            citation_results = get_citation_context(
                result.text, citation, words_before=500, words_after=1000
            )

            if citation_results:
                excerpts = [
                    OpinionExcerpt(
                        score=round(score, 3),
                        index_range=f"{start}-{end}",
                        text=context,
                    )
                    for context, start, end, score in citation_results
                ]

            else:
                excerpts = f"No excerpts found in opinion `{opinion_id}` for citation `{citation}`"

            opinion_excerpts = OpinionExcerpts(id=opinion_id, excerpts=excerpts)

            return opinion_excerpts.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting opinion: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting opinion: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    
@query_server.tool()
async def get_person(
    person_id: Annotated[str, Field(description="The person (judge) ID to retrieve")],
    ctx: Context | None = None,
) -> str:
    """Get judge or legal professional information by ID from CourtListener.

    Args:
        person_id: The person ID to retrieve.
        ctx: Optional context for logging and error reporting.

    Returns:
        str: The person data as returned by the CourtListener API.
    """

    if ctx:
        await ctx.info(f"Getting person with ID: {person_id}")
    else:
        logger.info(f"Getting person with ID: {person_id}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.courtlistener.com/api/rest/v4/people/{person_id}/",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            if ctx:
                await ctx.info(f"Successfully retrieved person {person_id}")
            else:
                logger.info(f"Successfully retrieved person {person_id}")

            result = Person(**data)

            return result.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting person: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting person: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    
@query_server.tool()
async def get_docket(
    docket_id: Annotated[str, Field(description="The docket ID to retrieve")],
    ctx: Context | None = None,
) -> str:
    """Get a specific court docket by ID from CourtListener.

    Args:
        docket_id: The docket ID to retrieve.
        ctx: Optional context for logging and error reporting.

    Returns:
        str: The docket data as returned by the CourtListener API.

    Raises:
        ValueError: If the COURT_LISTENER_API_KEY is not found in environment variables.

    """
    if ctx:
        await ctx.info(f"Getting docket with ID: {docket_id}")
    else:
        logger.info(f"Getting docket with ID: {docket_id}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.courtlistener.com/api/rest/v4/dockets/{docket_id}/",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            result = Docket(**data)

            if ctx:
                await ctx.info(f"Successfully retrieved docket {docket_id}")
            else:
                logger.info(f"Successfully retrieved docket {docket_id}")

            return result.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting docket: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting docket: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    
@query_server.tool()
async def get_attorney(
    attorney_id: Annotated[str, Field(description="The attorney ID to retrieve")],
    ctx: Context | None = None,
) -> str:
    """Get attorney information by ID from CourtListener.

    Args:
        attorney_id: The attorney ID to retrieve.
        ctx: Optional context for logging and error reporting.

    Returns:
        str: The attorney data as returned by the CourtListener API.
    """

    if ctx:
        await ctx.info(f"Getting attorney with ID: {attorney_id}")
    else:
        logger.info(f"Getting attorney with ID: {attorney_id}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.courtlistener.com/api/rest/v4/attorneys/{attorney_id}/",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            result = Attorney(**data)

            if ctx:
                await ctx.info(f"Successfully retrieved person {attorney_id}")
            else:
                logger.info(f"Successfully retrieved person {attorney_id}")

            return result.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting attorney: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting attorney: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    
@query_server.tool()
async def get_oral_argument(
    audio_id: Annotated[str, Field(description="The audio recording, i.e., oral argument, ID to retrieve")],
    ctx: Context | None = None,
) -> str:
    """Get oral argument information by ID from CourtListener. Typically contains transcript text.

    Args:
        audio_id: The audio recording, i.e., oral argument, ID to retrieve.
        ctx: Optional context for logging and error reporting.

    Returns:
        str: The oral argument data as returned by the CourtListener API.

    Raises:
        ValueError: If the COURT_LISTENER_API_KEY is not found in environment variables.

    """
    if ctx:
        await ctx.info(f"Getting oral argument with ID: {audio_id}")
    else:
        logger.info(f"Getting oral argument with ID: {audio_id}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.courtlistener.com/api/rest/v4/audio/{audio_id}/",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

            if ctx:
                await ctx.info(f"Successfully retrieved audio {audio_id}")
            else:
                logger.info(f"Successfully retrieved audio {audio_id}")

            result = OralArgument(**response.json())

            return result.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting audio: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting audio: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    

if __name__ == "__main__":
    # Initialize and run the server
    query_server.run(transport="stdio")
