import logging
import os
from typing import Annotated, Literal
from textwrap import dedent

import httpx
from dotenv import load_dotenv
from models import (
    Docket,
    DocketSearchResults,
    Opinion,
    OpinionExcerpt,
    OpinionExcerpts,
    OpinionSearchResults,
    Person,
    Attorney,
    PersonSearchResults,
    RECAPSearchResults,
    OralArgumentSearchResults,  
    OralArgument,
)
from pydantic import Field

from mcp.server.fastmcp import Context, FastMCP
from utils import (
    and_join,
    build_firm_query,
    date_range,
    get_citation_context,
    get_context_with_bm25,
)

load_dotenv()

API_KEY = os.getenv("COURT_LISTENER_API_KEY")
if not API_KEY:
    raise ValueError("COURT_LISTENER_API_KEY not found in environment variables")

COURTLISTENER_WEB_URL = "https://www.courtlistener.com"

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="court_listener",
    instructions=dedent("""\
        A set of tools for searching and retrieving data from the CourtListener API. 
        Provides three main patterns:
        - Search tools: search_opinions, search_opinions_by_citation, search_people, search_dockets_by_firm_name, search_recap_docs_by_attorney_name, search_oral_arguments
        - Get tools: get_opinion, get_opinion_excerpt, get_opinion_excerpt_by_citation, get_docket, get_attorney, get_person, get_oral_argument
        - Fetch tools: fetch_forward_citations
        Use the search tools to search for data, e.g., ID and metadata, and the get tools to retrieve data, e.g., opinion full text and excerpt(s), people, dockets, oral arguments.
        Use the fetch tools to fetch additional data that is not directly available from the search or get tools, e.g., forward citations.
    """))


@mcp.tool()
async def search_opinions(
    q: Annotated[str, Field(description="Search query for full text of opinions")],
    court: Annotated[
        str | None, Field(description="Court ID filter (e.g., 'scotus', 'ca9')")
    ] = None,
    case_name: Annotated[str | None, Field(description="Filter by case name")] = None,
    judge: Annotated[str | None, Field(description="Filter by judge name")] = None,
    filed_after: Annotated[
        str | None,
        Field(description="Only show opinions filed after this date (YYYY-MM-DD)"),
    ] = None,
    filed_before: Annotated[
        str | None,
        Field(description="Only show opinions filed before this date (YYYY-MM-DD)"),
    ] = None,
    order_by: Annotated[
        str,
        Field(description="Sort by 'score desc', 'dateFiled desc', or 'dateFiled asc'"),
    ] = "score desc",
    limit: Annotated[
        int, Field(description="Maximum results to return", ge=1, le=100)
    ] = 10,
    ctx: Context | None = None,
) -> str:
    """Full text search for case law opinions in CourtListener.

    Args:
        q: Search query for full text of opinions.
        court (optional): Court ID filter (e.g., 'scotus', 'ca9').
        case_name (optional): Filter by case name.
        judge (optional): Filter by judge name.
        filed_after (optional): Only show opinions filed after this date (YYYY-MM-DD).
        filed_before (optional): Only show opinions filed before this date (YYYY-MM-DD).
        order_by: Sort by 'score desc', 'dateFiled desc', or 'dateFiled asc'. Defaults to 'score desc'.
        limit: Maximum results to return. Defaults to 10.

    Returns:
        A string containing search results with opinion clusters and nested opinions.
    """

    if ctx:
        await ctx.info(f"Searching opinions with query: {q}")
    else:
        logger.info(f"Searching opinions with query: {q}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    params = {
        "q": q,
        "order_by": order_by,
        "type": "o",  # Opinion type for V4 API
    }

    # Add optional filters
    if court:
        params["court"] = court
    if case_name:
        params["case_name"] = case_name
    if judge:
        params["judge"] = judge
    if filed_after:
        params["filed_after"] = filed_after
    if filed_before:
        params["filed_before"] = filed_before
    if limit:
        params["hit"] = limit  # V4 uses 'hit' instead of 'limit'

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.courtlistener.com/api/rest/v4/search/",
                params=params,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            data['results'] = data['results'][:limit]
            
            if ctx:
                await ctx.info(f"Found {data.get('count', 0)} opinions")
            else:
                logger.info(f"Found {data.get('count', 0)} opinions")

            results = OpinionSearchResults(**data)

            return results.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Search error: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e


@mcp.tool()
async def search_opinions_by_citation(
    citation: Annotated[str, Field(description="The citation to search for")],
    limit: Annotated[int, Field(description="The maximum number of results to return", ge=1, le=100)] = 10,
    ctx: Context | None = None,
) -> str:
    """Search for court opinions by citation string from CourtListener. Useful for finding the opinion ID of a given citation
    along with other metadata.

    Args:
        citation: The citation to search for, e.g., '410 U.S. 113' or '2023 WL 12345'.
        limit: The maximum number of results to return. Default is 10.

    Returns:
        str: The opinion search results as returned by the CourtListener API.
    """

    if ctx:
        await ctx.info(f"Getting opinion with citation: {citation}")
    else:
        logger.info(f"Getting opinion with citation: {citation}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}

    params = {
        "citation": citation,
        "type": "o",  # Opinion type for V4 API
        "limit": limit,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.courtlistener.com/api/rest/v4/search/",
                params=params,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            data['results'] = data['results'][:limit]
            results = OpinionSearchResults(**data)

            if ctx:
                await ctx.info(
                    f"Successfully retrieved opinions for citation {citation}"
                )
            else:
                logger.info(f"Successfully retrieved opinions for citation {citation}")

            return results.to_xml()

    except Exception as e:
        if ctx:
            await ctx.error(f"Error retrieving opinions for citation {citation}: {e}")
        else:
            logger.error(f"Error retrieving opinions for citation {citation}: {e}")
        raise e


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
async def fetch_forward_citations(
    opinion_id: Annotated[
        int, Field(description="The target opinion ID to find forward citations for")
    ],
    ctx: Context | None = None,
) -> list[int]:
    """
    Calls Court Listener's front end citation lookup engine to search all records for
        a given case id that cites the target opinion.

    Args:
        opinion_id (int): The unique identifier for the opinion to find forward citations for.
        ctx: Optional context for logging and error reporting.

    Returns:
        List[int]: A list of ids for cases that cite the target opinion.
    """

    ep = f"https://www.courtlistener.com/api/rest/v4/search/?q=cites%3A({opinion_id})&type=o&order_by=dateFiled%20asc&stat_Precedential=on"

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                ep,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            results = OpinionSearchResults(**data)
            forward_citations = [
                result.primary_opinion_id for result in results.results
            ]
            return forward_citations

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting forward citation ids: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting forward citation ids: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e


@mcp.tool()
async def search_people(
    q: Annotated[
        str, Field(description="Search query for judges and legal professionals")
    ],
    name: Annotated[str, Field(description="Optional filter by person's name")] = "",
    position_type: Annotated[
        str | None,
        Field(description="Optional filter by position type (e.g., 'jud' for judge)"),
    ] = None,
    political_affiliation: Annotated[
        str | None, Field(description="Optional filter by political affiliation")
    ] = None,
    school: Annotated[
        str | None, Field(description="Optional filter by school attended")
    ] = None,
    appointed_by: Annotated[
        str | None, Field(description="Optional filter by appointing authority")
    ] = "",
    selection_method: Annotated[
        str | None, Field(description="Optional filter by selection method")
    ] = "",
    order_by: Annotated[
        str,
        Field(
            description="Sort by 'score desc' or 'name asc'. Default is 'score desc'"
        ),
    ] = "score desc",
    limit: Annotated[
        int,
        Field(description="Maximum results to return. Default is 10.", ge=1, le=100),
    ] = 10,
    ctx: Context | None = None,
) -> str:
    """Search judges and related legal professionals in the CourtListener database. Note this tool is typically not useful for retrieving attorneys, 
    but rather for retrieving judges and related legal professionals. 

    Args:
        q: Search query for judges and legal professionals.
        name (optional): Optional filter by person's name.
        position_type (optional): Optional filter by position type (e.g., 'jud' for judge).
        political_affiliation (optional): Optional filter by political affiliation.
        school (optional): Optional filter by school attended.
        appointed_by (optional): Optional filter by appointing authority.
        selection_method (optional): Optional filter by selection method.
        order_by: Sort by 'score desc' or 'name asc'. Default is 'score desc'.
        limit: Maximum results to return. Default is 10.

    Returns:
        A string containing search results with people information.
    """

    if ctx:
        await ctx.info(f"Searching people with query: {q}")
    else:
        logger.info(f"Searching people with query: {q}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    params = {
        "q": q,
        "order_by": order_by,
        "type": "p",  # People type for V4 API
    }

    # Add optional filters
    if name:
        params["name"] = name
    if position_type:
        params["position_type"] = position_type
    if political_affiliation:
        params["political_affiliation"] = political_affiliation
    if school:
        params["school"] = school
    if appointed_by:
        params["appointed_by"] = appointed_by
    if selection_method:
        params["selection_method"] = selection_method
    if limit:
        params["hit"] = limit  # V4 uses 'hit' instead of 'limit'

    headers = {"Authorization": f"Token {API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.courtlistener.com/api/rest/v4/search/",
                params=params,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            if ctx:
                await ctx.info(f"Found {data.get('count', 0)} people")
            else:
                logger.info(f"Found {data.get('count', 0)} people")

            data['results'] = data['results'][:limit]
            results = PersonSearchResults(**data)

            return results.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Search error: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e


@mcp.tool()
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


@mcp.tool()
async def search_dockets_by_firm_name(
    firm_name: Annotated[str, Field(description="The name of the firm to search for")],
    court_id: Annotated[
        str | None, Field(description="Optional ID of the court to search in")
    ] = None,
    date_filed_start: Annotated[
        str | None,
        Field(description="Optional start date of the date range to search in (YYYY-MM-DD)"),
    ] = None,
    date_filed_end: Annotated[
        str | None,
        Field(description="Optional end date of the date range to search in (YYYY-MM-DD)"),
    ] = None,
    order_by: Annotated[
        Literal["dateFiled desc", "dateFiled asc"],
        Field(description="Optional sort by 'dateFiled desc' or 'dateFiled asc'"),
    ] = "dateFiled desc",
    proximity: Annotated[
        int, Field(description="Word proximity for proximity search mode", ge=1)
    ] = 2,
    ctx: Context | None = None,
) -> str:
    """Search for court dockets by firm name from CourtListener.

    Args:
        firm_name: The name of the firm to search for.
        court_id: Optional ID of the court to search in.
        date_filed_start: Optional start date of the date range to search in.
        date_filed_end: Optional end date of the date range to search in.
        order_by: Optional order by which to sort the results. Must be one of {'dateFiled desc', 'dateFiled asc'}.
        proximity: The proximity of the search. Default is 2.
        ctx: Optional context for logging and error reporting.

    Returns:
        str: The docket search results as returned by the CourtListener API.
    """

    if ctx:
        await ctx.info(f"Searching dockets with firm name: {firm_name}")
    else:
        logger.info(f"Searching dockets with firm name: {firm_name}")

    firm_clause = build_firm_query(firm_name, mode="proximity", proximity=proximity)
    clauses = [
        firm_clause,
        f"court_id:{court_id}" if court_id else None,
        date_range("dateFiled", date_filed_start, date_filed_end),
    ]
    q = and_join(clauses)

    if ctx:
        await ctx.info(f"Searching dockets with firm name: {firm_name}")
    else:
        logger.info(f"Searching dockets with firm name: {firm_name}")

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
                "https://www.courtlistener.com/api/rest/v4/search/",
                params=dict(q=q, type="d", order_by=order_by),
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            results = DocketSearchResults(**data)

            if ctx:
                await ctx.info(
                    f"Successfully retrieved dockets for firm name {firm_name}"
                )
            else:
                logger.info(f"Successfully retrieved dockets for firm name {firm_name}")

            return results.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting dockets for firm name: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting dockets for firm name: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e


@mcp.tool()
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
    
    
@mcp.tool()
async def search_recap_docs_by_attorney_name(
    attorney_name: Annotated[str, Field(description="The name of the attorney to search for")],
    max_results: Annotated[int, Field(description="The maximum number of results to return", ge=1, le=20)] = 5,
    court_id: Annotated[str | None, Field(description="Optional ID of the court to search in")] = None,
    date_filed_start: Annotated[str | None, Field(description="Optional start date of the date range to search in (YYYY-MM-DD)")] = None,
    date_filed_end: Annotated[str | None, Field(description="Optional end date of the date range to search in (YYYY-MM-DD)")] = None,
    order_by: Annotated[
        Literal["dateFiled desc", "dateFiled asc"],
        Field(description="Sort by 'dateFiled desc' or 'dateFiled asc'. Default is 'dateFiled desc'"),
    ] = "dateFiled desc",
    ctx: Context | None = None,
) -> str:
    """Search for RECAP documents by attorney name from CourtListener. Useful retrieving documents by attorney name and the 
    attorney ID for further querying.

    Args:
        attorney_name: The name of the attorney to search for.
        max_results: The maximum number of results to return. Default is 5.
        court_id: Optional ID of the court to search in.
        date_filed_start: Optional start date of the date range to search in.
        date_filed_end: Optional end date of the date range to search in.
        order_by: Sort by 'dateFiled desc' or 'dateFiled asc'. Default is 'dateFiled desc'.
        ctx: Optional context for logging and error reporting.

    Returns:
        str: The RECAP document search results as returned by the CourtListener API.
    """

    def _quote(s: str) -> str:
        # Escape quotes by replacing " with "" and wrap in quotes
        escaped = s.replace('"', '""')
        return '"' + escaped + '"'

    clauses = [
        f"attorney:{_quote(attorney_name)}",
        f"court_id:{court_id}" if court_id else None,
        date_range("dateFiled", date_filed_start, date_filed_end),
    ]
    q = and_join(clauses)

    headers = {"Authorization": f"Token {API_KEY}"}

    if ctx:
        await ctx.info(f"Searching RECAP documents with attorney name: {attorney_name}")
    else:
        logger.info(f"Searching RECAP documents with attorney name: {attorney_name}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.courtlistener.com/api/rest/v4/search/",
                params=dict(q=q, type="r", order_by=order_by),
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            data['results'] = data['results'][:max_results]
            results = RECAPSearchResults(**data)

            if ctx:
                await ctx.info(
                    f"Successfully retrieved dockets for attorney name {attorney_name}"
                )
            else:
                logger.info(
                    f"Successfully retrieved dockets for attorney name {attorney_name}"
                )

            return results.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting dockets for attorney name: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting dockets for attorney name: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    
@mcp.tool()
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
    
    
@mcp.tool()
async def search_oral_arguments(
    q: Annotated[str, Field(description="Search query for oral arguments")],
    limit: Annotated[int, Field(description="The maximum number of results to return", ge=1, le=100)] = 10,
    ctx: Context | None = None,
) -> str:
    """Search for oral arguments from CourtListener.

    Args:
        q: The search query to execute.
        limit: The maximum number of results to return. Default is 10.
        ctx: Optional context for logging and error reporting.

    Returns:
        str: The oral argument search results as returned by the CourtListener API.
    """
    
    if ctx:
        await ctx.info(f"Searching oral arguments with query: {q}")
    else:
        logger.info(f"Searching oral arguments with query: {q}")

    if not API_KEY:
        error_msg = "COURT_LISTENER_API_KEY not found in environment variables"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise ValueError(error_msg)

    headers = {"Authorization": f"Token {API_KEY}"}
    params = {
        "q": q,
        "type": "oa",
        "limit": limit,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.courtlistener.com/api/rest/v4/search/",
                params=params,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            data['results'] = data['results'][:limit]
            results = OralArgumentSearchResults(**data)

            if ctx:
                await ctx.info(f"Successfully retrieved oral arguments with query {q}")
            else:
                logger.info(f"Successfully retrieved oral arguments with query {q}")

            return results.to_xml()

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error getting oral arguments: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    except Exception as e:
        error_msg = f"Error getting oral arguments: {e}"
        if ctx:
            await ctx.error(error_msg)
        else:
            logger.error(error_msg)
        raise e
    
    
@mcp.tool()
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
    mcp.run(transport="stdio")
