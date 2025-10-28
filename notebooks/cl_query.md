court_listener_agent ➤ how can we use court listener to find all of the attorneys that work with the firm Morgan and Morgan?


Calling tool 'get_court_listener_query' with args: {'query': '"Find all attorneys who work for the firm Morgan and Morgan (including variations \'Morgan & Morgan\', \'Morgan & Morgan, P.A.\', \'Morgan &
Morgan, P.A., Management\', etc.). Provide CourtListener advanced search queries to locate attorney records affiliated with that firm, including suggestions for handling punctuation/ampersand variants,
state-specific offices, and common false positives (e.g., attorneys with last name Morgan but different firms). Return a series of queries covering different approaches (attorney name/firm field if
available, text search across opinions/dockets/attorney records) and include exact parameters for each query."'}


Finished calling tool 'get_court_listener_query'


Good — below are practical, ready-to-run CourtListener advanced-search queries and a short plan for finding and verifying attorneys who work for Morgan & Morgan (including common name variants). I’ll not
execute them unless you ask.

High-level approach

 • Prefer structured fields (firm:, attorney:) in the dockets/people indices to avoid false positives (attorneys named “Morgan” at other firms).
 • Also run full-text searches across opinions/filings to catch mentions that aren’t captured in structured fields.
 • Use multiple firm-name variants with OR to capture punctuation/ampersand/abbreviation differences.
 • Post-process results to deduplicate (name + bar number or address/email) and filter out false positives.
 • If you want, I can run these queries and return the raw results and a deduplicated list.

Queries to run

 • The JSON parameter blocks below are ready to pass to the execute_court_listener_search_query tool (or to CourtListener’s API). Each block shows the exact parameters to run that search. Adjust the limit as   needed (many results may require a larger limit).

 1 Search dockets (RECAP dockets) by firm name variants

 • Use this to find dockets where Morgan & Morgan is listed as a firm (captures firm-affiliation lines in dockets). Code: { "q": "firm:("Morgan and Morgan" OR "Morgan & Morgan" OR "Morgan & Morgan, P.A." OR
   "Morgan & Morgan PA" OR "Morgan & Morgan, P.A., Management" OR "Morgan & Morgan, P.A., Management Company")", "type": "r", "order_by": "dateFiled desc", "limit": 500 }

 2 Search dockets by attorney field (attorneys linked to cases)

 • This locates docket attorney entries that mention the firm in the attorney block. Code: { "q": "attorney:("Morgan and Morgan" OR "Morgan & Morgan" OR "Morgan & Morgan, P.A." OR "Morgan & Morgan PA" OR
   "Morgan & Morgan, P.A., Management")", "type": "r", "order_by": "dateFiled desc", "limit": 500 }

 3 Search persons (attorney records / people index) by firm

 • This tries to find person records where the firm field contains Morgan & Morgan (best for creating a list of attorneys). Code: { "q": "firm:("Morgan and Morgan" OR "Morgan & Morgan" OR "Morgan & Morgan,
   P.A." OR "Morgan & Morgan PA" OR "Morgan & Morgan, P.A., Management" OR "Morgan & Morgan, P.A., Management Company")", "type": "p", "order_by": "name asc", "limit": 500 }

 4 Full-text search of opinions (catch textual mentions)

 • Useful for older cases or places where structured attorney metadata is missing. Code: { "q": ""Morgan and Morgan" OR "Morgan & Morgan" OR "Morgan & Morgan, P.A." OR "Morgan & Morgan PA"", "type": "o",
   "order_by": "dateFiled desc", "limit": 500 }

 5 Search RECAP filing documents (pdfs/filings) text/description

 • Filings often include a firm block with full attorney contact that might not surface elsewhere. Code: { "q": "description:("Morgan and Morgan" OR "Morgan & Morgan" OR "Morgan & Morgan, P.A." OR "Morgan &
   Morgan PA" OR "Morgan & Morgan, P.A., Management")", "type": "rd", "order_by": "entry_date_filed desc", "limit": 500 }

Tips to reduce false positives and improve precision

 • Prefer queries that search the firm: field in the people index (type=p) or the firm: field when available — this avoids results that merely contain the word Morgan.
 • Use the phrase "Morgan & Morgan" (with ampersand) and "Morgan and Morgan" (spelled out) both in OR — some records use one or the other.
 • If you get many false positives (e.g., attorneys whose last name is Morgan), filter further by requiring the firm phrase in the same record (the firm: field) or search for known Morgan & Morgan email
   domains or addresses.
 • To focus by geography/office, add a court or jurisdiction filter (e.g., limit by district court code or include a state name in the query). Example: add AND court:(flmd) or AND jurisdiction:fl (you can
   first look up court codes on CourtListener).
 • Use a higher limit and then deduplicate by (full name + bar number / address / attorney ID) because results can repeat across dockets.

Next steps I can do for you

 • Run these queries and return raw results (I can then extract attorney names and person IDs).
 • From the results, fetch person/attorney records using the get_attorney or get_person tools to retrieve contact info and verify firm affiliation.
 • Deduplicate, sort by state/office, and produce a clean list (with confidence flags for matches that are exact firm matches vs. textual mentions).

Would you like me to run these searches now and produce a deduplicated list of attorneys for Morgan & Morgan? If yes, tell me any scope limits (e.g., specific states or federal courts to prioritize).
court_listener_agent ➤ /markdown
Markdown output of last question:

Good — below are practical, ready-to-run CourtListener advanced-search queries and a short plan for finding and verifying attorneys who work for Morgan & Morgan (including common name variants). I’ll not
execute them unless you ask.

High-level approach

- Prefer structured fields (firm:, attorney:) in the dockets/people indices to avoid false positives (attorneys named “Morgan” at other firms).
- Also run full-text searches across opinions/filings to catch mentions that aren’t captured in structured fields.
- Use multiple firm-name variants with OR to capture punctuation/ampersand/abbreviation differences.
- Post-process results to deduplicate (name + bar number or address/email) and filter out false positives.
- If you want, I can run these queries and return the raw results and a deduplicated list.

Queries to run

- The JSON parameter blocks below are ready to pass to the execute_court_listener_search_query tool (or to CourtListener’s API). Each block shows the exact parameters to run that search. Adjust the limit as
  needed (many results may require a larger limit).

1) Search dockets (RECAP dockets) by firm name variants

- Use this to find dockets where Morgan & Morgan is listed as a firm (captures firm-affiliation lines in dockets).
  Code:
  {
  "q": "firm:(\"Morgan and Morgan\" OR \"Morgan & Morgan\" OR \"Morgan & Morgan, P.A.\" OR \"Morgan & Morgan PA\" OR \"Morgan & Morgan, P.A., Management\" OR \"Morgan & Morgan, P.A., Management Company\")",
  "type": "r",
  "order_by": "dateFiled desc",
  "limit": 500
  }

2) Search dockets by attorney field (attorneys linked to cases)

- This locates docket attorney entries that mention the firm in the attorney block.
  Code:
  {
  "q": "attorney:(\"Morgan and Morgan\" OR \"Morgan & Morgan\" OR \"Morgan & Morgan, P.A.\" OR \"Morgan & Morgan PA\" OR \"Morgan & Morgan, P.A., Management\")",
  "type": "r",
  "order_by": "dateFiled desc",
  "limit": 500
  }

3) Search persons (attorney records / people index) by firm

- This tries to find person records where the firm field contains Morgan & Morgan (best for creating a list of attorneys).
  Code:
  {
  "q": "firm:(\"Morgan and Morgan\" OR \"Morgan & Morgan\" OR \"Morgan & Morgan, P.A.\" OR \"Morgan & Morgan PA\" OR \"Morgan & Morgan, P.A., Management\" OR \"Morgan & Morgan, P.A., Management Company\")",
  "type": "p",
  "order_by": "name asc",
  "limit": 500
  }

4) Full-text search of opinions (catch textual mentions)

- Useful for older cases or places where structured attorney metadata is missing.
  Code:
  {
  "q": "\"Morgan and Morgan\" OR \"Morgan & Morgan\" OR \"Morgan & Morgan, P.A.\" OR \"Morgan & Morgan PA\"",
  "type": "o",
  "order_by": "dateFiled desc",
  "limit": 500
  }

5) Search RECAP filing documents (pdfs/filings) text/description

- Filings often include a firm block with full attorney contact that might not surface elsewhere.
  Code:
  {
  "q": "description:(\"Morgan and Morgan\" OR \"Morgan & Morgan\" OR \"Morgan & Morgan, P.A.\" OR \"Morgan & Morgan PA\" OR \"Morgan & Morgan, P.A., Management\")",
  "type": "rd",
  "order_by": "entry_date_filed desc",
  "limit": 500
  }

Tips to reduce false positives and improve precision

- Prefer queries that search the firm: field in the people index (type=p) or the firm: field when available — this avoids results that merely contain the word Morgan.
- Use the phrase "Morgan & Morgan" (with ampersand) and "Morgan and Morgan" (spelled out) both in OR — some records use one or the other.
- If you get many false positives (e.g., attorneys whose last name is Morgan), filter further by requiring the firm phrase in the same record (the firm: field) or search for known Morgan & Morgan emaildomains or addresses.
- To focus by geography/office, add a court or jurisdiction filter (e.g., limit by district court code or include a state name in the query). Example: add AND court:(flmd) or AND jurisdiction:fl (you canfirst look up court codes on CourtListener).
- Use a higher limit and then deduplicate by (full name + bar number / address / attorney ID) because results can repeat across dockets.

Next steps I can do for you

- Run these queries and return raw results (I can then extract attorney names and person IDs).
- From the results, fetch person/attorney records using the get_attorney or get_person tools to retrieve contact info and verify firm affiliation.
- Deduplicate, sort by state/office, and produce a clean list (with confidence flags for matches that are exact firm matches vs. textual mentions).

Would you like me to run these searches now and produce a deduplicated list of attorneys for Morgan & Morgan? If yes, tell me any scope limits (e.g., specific states or federal courts to prioritize).
