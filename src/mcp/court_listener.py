import logging
import os
from typing import Annotated

import httpx
from dotenv import load_dotenv
from pydantic import Field
from mcp.server.fastmcp import Context, FastMCP

from models import OpinionSearchResults, PersonSearchResults, Opinion, Person, OpinionExcerpt, OpinionExcerpts
from utils import get_context_with_bm25


load_dotenv()

API_KEY = os.getenv("COURT_LISTENER_API_KEY")
if not API_KEY:
    raise ValueError("COURT_LISTENER_API_KEY not found in environment variables")

logger = logging.getLogger(__name__)

mcp = FastMCP("court_listener")


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
    """Search case law opinion clusters with nested Opinion documents in CourtListener.

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
    """Search judges and legal professionals in the CourtListener database.

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
async def get_opinion(
    opinion_id: Annotated[str, Field(description="The opinion ID to retrieve")],
    full_text: Annotated[bool | None, Field(default=None, description="Whether to return the full text of the opinion")],
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

            result = Opinion(
                full_text_flag=full_text,
                **data
            )
            
            return result

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
async def get_opinion_excerpts(
    opinion_id: Annotated[str, Field(description="The opinion ID to retrieve")],
    query: Annotated[str, Field(description="The query to retrieve excerpts from the opinion")],
    ctx: Context | None = None,
) -> str:
    """Given an input query, retrieve excerpts from a specific court opinion by ID from CourtListener.

    Args:
        opinion_id: The opinion ID to retrieve.
        query: The query to retrieve excerpts from the opinion.

    Returns:
        str: The opinion excerpts as returned by the CourtListener API.
    """

    if ctx:
        await ctx.info(f"Getting opinion with ID: {opinion_id} and query: {query}")
    else:
        logger.info(f"Getting opinion with ID: {opinion_id} and query: {query}")

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

            result = Opinion(
                full_text_flag=True,
                **data
            )
            
            bm25_results = get_context_with_bm25(query, result.text, 50, 100, adjust_to_sentences=True)
            
            if bm25_results:
                excerpts = [OpinionExcerpt(
                    score=round(score, 3),
                    index_range=f"{start}-{end}",
                    text=context
                ) for context, start, end, score in bm25_results]
                
            else:
                excerpts = f"No excerpts found in opinion `{opinion_id}` for query `{query}`"
                
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
async def get_person(
    person_id: Annotated[str, Field(description="The person (judge) ID to retrieve")],
    ctx: Context | None = None,
) -> str:
    """Get judge or legal professional information by ID from CourtListener.

    Args:
        person_id: The person (judge) ID to retrieve.
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


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
