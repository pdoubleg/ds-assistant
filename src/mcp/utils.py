import difflib
import re
from typing import Iterable, List, Optional, Tuple

import eyecite
import html2text

try:
    import nltk
    import spacy
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
    SPACY_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    SPACY_AVAILABLE = False


def build_firm_query(
    firm_name: str,  # e.g., "Kirkland & Ellis LLP - Chicago"
    mode: str = "proximity",  # "proximity" | "and" | "mixed"
    proximity: int = 4,
) -> str:
    """
    Build a firm: query that survives punctuation (&, -, en dash) and exact-phrase pitfalls.
    """
    # Basic normalization: replace ampersand and dash variants with spaces, collapse whitespace
    normalized = firm_name.replace("&", " ").replace("–", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    if mode == "proximity":
        # firm:("Kirkland Ellis LLP Chicago"~4)
        return f'firm:("{normalized}"~{proximity})'

    if mode == "and":
        # firm:Kirkland AND firm:Ellis AND firm:LLP AND firm:Chicago
        tokens = normalized.split()
        return " AND ".join(f"firm:{t}" for t in tokens)

    if mode == "mixed":
        # firm:("Kirkland Ellis LLP"~2) AND firm:Chicago
        parts = normalized.split()
        if len(parts) >= 2:
            head = " ".join(parts[:-1])
            tail = parts[-1]
            return f'firm:("{head}"~2) AND firm:{tail}'
        else:
            return f'firm:("{normalized}"~2)'

    raise ValueError("mode must be one of {'proximity','and','mixed'}")


def date_range(field: str, start: Optional[str], end: Optional[str]) -> Optional[str]:
    if not start and not end:
        return None
    lo = start if start else "*"
    hi = end if end else "*"
    return f"{field}:[{lo} TO {hi}]"


def paren(expr: str) -> str:
    return f"({expr})"


def and_join(clauses: Iterable[Optional[str]]) -> str:
    parts: list[str] = []
    for c in clauses:
        if not c:
            continue
        needs_paren = any(
            tok in c for tok in [" ", " AND ", " OR ", " TO ", ":", "[", "]", '"']
        )
        parts.append(
            paren(c)
            if needs_paren and not (c.startswith("(") and c.endswith(")"))
            else c
        )
    return " AND ".join(parts) if parts else "*:*"


def html_to_text(html_string: str) -> str:
    """Converts an HTML string to plain text using html2text.

    Args:
        html_string (str): The HTML string to convert.

    Returns:
        str: The converted plain text string.

    Example:
        >>> html_to_text("<p>Hello <strong>world</strong>!</p>")
        'Hello **world**!'
    """
    # Create html2text instance with desired settings
    h = html2text.HTML2Text()
    h.ignore_links = False  # Keep links as markdown
    h.ignore_images = True  # Remove images
    h.body_width = 0  # Don't wrap lines

    # Convert HTML to text
    text = h.handle(html_string)

    # Clean up extra whitespace
    return text.strip()


def get_context(
    query: str,
    text: str,
    words_before: int = 100,
    words_after: int = 500,
) -> Tuple[str, int, int]:
    """
    Returns a portion of text containing the best approximate match of the query,
    including b words before and a words after the match.

    Args:
    query (str): The string to search for.
    text (str): The body of text in which to search.
    b (int): The number of words before the query to return.
    a (int): The number of words after the query to return.

    Returns:
    str: A string containing b words before, the match, and a words after
        the best approximate match position of the query in the text.
        The text is extracted from the original `text`, preserving formatting,
        whitespace, etc, so it does not disturb any downstream processing.
        If no match is found, returns empty string.
    int: The start position of the match in the text.
    int: The end position of the match in the text.
    """

    # Find best matching position of query in text
    sequence_matcher = difflib.SequenceMatcher(None, text, query)
    match = sequence_matcher.find_longest_match(0, len(text), 0, len(query))

    if match.size == 0:
        return "", 0, 0

    # Count words before match point
    segments = text.split()
    n_segs = len(segments)
    start_segment_pos = len(text[: match.a].split())

    # Calculate word window boundaries
    words_before = words_before or n_segs
    words_after = words_after or n_segs
    start_pos = max(0, start_segment_pos - words_before)
    end_pos = min(len(segments), start_segment_pos + words_after + len(query.split()))

    # Find character positions where words start
    word_positions = [m.start() for m in re.finditer(r"\S+", text)]

    # Convert word positions to character positions
    start_char = word_positions[start_pos] if start_pos < len(word_positions) else 0
    end_char = word_positions[min(end_pos, len(word_positions) - 1)] + len(
        text.split()[min(end_pos - 1, len(word_positions) - 1)]
    )

    # return exact substring with original formatting
    return text[start_char:end_char], start_pos, end_pos


def download_nltk_resource(resource: str) -> None:
    """
    Downloads an NLTK resource if it's not already available.

    Args:
        resource (str): The NLTK resource identifier (e.g., 'tokenizers/punkt').

    Example:
        >>> download_nltk_resource('tokenizers/punkt')
    """
    if not BM25_AVAILABLE:
        raise ImportError(
            "NLTK is not available. Please install nltk to use BM25 functionality."
        )

    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)


def preprocess_text(text: str) -> str:
    """
    Preprocesses the given text by:
    1. Lowercasing all words.
    2. Tokenizing (splitting the text into words).
    3. Removing punctuation.
    4. Removing stopwords.
    5. Lemmatizing words.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.

    Example:
        >>> preprocess_text("The quick brown fox jumps over the lazy dog!")
        'quick brown fox jump lazy dog'
    """
    if not BM25_AVAILABLE:
        raise ImportError(
            "BM25Okapi is not available. Please install rank-bm25 to use BM25 functionality."
        )

    # Ensure the NLTK resources are available
    for resource in ["punkt", "wordnet", "stopwords"]:
        download_nltk_resource(resource)

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer

    # Lowercase the text
    text = text.lower()

    # Tokenize the text and remove punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Join the words back into a string
    text = " ".join(tokens)

    return text


def get_context_with_bm25(
    query: str,
    text: str,
    words_before: int = 100,
    words_after: int = 500,
    k: int = 5,
    spacy_model: str = "en_core_web_sm",
    adjust_to_sentences: bool = True,
) -> List[Tuple[str, int, int, float]]:
    """
    Returns k best matches using BM25 on sentence-level chunks from a single text,
    with surrounding context for each match. Uses spaCy for robust sentence segmentation.

    This function uses spaCy to split the input text into sentences, applies BM25 to find
    the most relevant sentences based on the query, then extracts surrounding context.
    Optionally adjusts context boundaries to complete sentences.

    Args:
        query (str): The string to search for.
        text (str): The input text to search within.
        words_before (int, optional): Number of words before each match to include. Defaults to 100.
        words_after (int, optional): Number of words after each match to include. Defaults to 500.
        k (int, optional): Number of best matches to return. Defaults to 5.
        spacy_model (str, optional): spaCy model to use for sentence segmentation. Defaults to "en_core_web_sm".
        adjust_to_sentences (bool, optional): Whether to adjust context to complete sentences. Defaults to True.

    Returns:
        List[Tuple[str, int, int, float]]: List of tuples, each containing:
            - str: Text containing words before, the match, and words after (adjusted to complete sentences if enabled).
            - int: Start word position in the original text.
            - int: End word position in the original text.
            - float: BM25 score of the match.

        Returns empty list if no matches are found.
    """

    if not BM25_AVAILABLE:
        raise ImportError(
            "BM25 dependencies not available. Please install rank-bm25 and nltk."
        )

    if not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy not available. Please install spacy and download the language model."
        )

    if not text.strip():
        return []

    try:
        # Load spaCy model
        nlp = spacy.load(spacy_model)
    except OSError:
        raise ImportError(
            f"spaCy model '{spacy_model}' not found. Please install it with: python -m spacy download {spacy_model}"
        )

    # Process text with spaCy to get sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return []

    # Preprocess sentences for BM25
    sentences_clean = [preprocess_text(sentence) for sentence in sentences]

    # Filter out empty preprocessed sentences
    valid_indices = [i for i, s in enumerate(sentences_clean) if s.strip()]
    if not valid_indices:
        return []

    sentences = [sentences[i] for i in valid_indices]
    sentences_clean = [sentences_clean[i] for i in valid_indices]

    # Prepare BM25
    query_clean = preprocess_text(query)
    if not query_clean.strip():
        return []

    sentence_words = [sentence.split() for sentence in sentences_clean]
    bm25 = BM25Okapi(sentence_words)
    query_words = query_clean.split()

    # Get BM25 scores for all sentences
    doc_scores = bm25.get_scores(query_words)

    # Get indices of top k scores
    top_indices = sorted(range(len(doc_scores)), key=lambda i: -doc_scores[i])[:k]

    results = []

    for idx in top_indices:
        if doc_scores[idx] <= 0:  # Skip sentences with no relevance
            continue

        sentence = sentences[idx]
        score = doc_scores[idx]

        # Use difflib to find the exact match position
        sequence_matcher = difflib.SequenceMatcher(None, text, sentence)
        match_info = sequence_matcher.find_longest_match(0, len(text), 0, len(sentence))

        if match_info.size == 0:
            continue

        # Calculate word positions using the matched position
        segments = text.split()
        n_segs = len(segments)
        start_segment_pos = len(text[: match_info.a].split())

        # Calculate word window boundaries
        words_before_adj = words_before or n_segs
        words_after_adj = words_after or n_segs
        start_pos = max(0, start_segment_pos - words_before_adj)
        end_pos = min(
            n_segs, start_segment_pos + words_after_adj + len(sentence.split())
        )

        # Extract initial context
        context = " ".join(segments[start_pos:end_pos])

        # Use spaCy to adjust to complete sentences if requested
        if adjust_to_sentences and context.strip():
            try:
                context_doc = nlp(context)
                context_sentences = list(context_doc.sents)

                # If we have more than 2 sentences, drop the first and last if they might be incomplete
                if len(context_sentences) > 2:
                    adjusted_context = " ".join(
                        sentence.text for sentence in context_sentences[1:-1]
                    )
                    if adjusted_context.strip():
                        context = adjusted_context
                # If we have exactly 2 sentences, keep them both
                elif len(context_sentences) == 2:
                    context = " ".join(sentence.text for sentence in context_sentences)
                # If we have 1 sentence, keep it as is
                # (context remains unchanged)

            except Exception:
                # If spaCy processing fails, use the original context
                pass

        results.append((context, start_pos, end_pos, score))

    return results


# def get_citation_context(
#     text: str,
#     citation: str,
#     words_before: int = 1000,
#     words_after: int = 1000,
#     max_excerpts: int = 10,
# ) -> str:
#     """
#     Finds a citation in text and return the context around it, with the start and end positions
#     adjusted to complete sentences.

#     Args:
#         text (str): The text to search.
#         citation (str): The citation to find.
#         words_before (int): Maximum number of words to include before the citation. Defaults to 1000.
#         words_after (int): Maximum number of words to include after the citation. Defaults to 1000.
#         max_excerpts (int): Maximum number of excerpts to return. Defaults to 10.

#     Returns:
#         str: The context around the citation, adjusted to complete sentences.
#     """
#     if words_after is None and words_before is None:
#         # Return entire text since we're not asked to return a bounded context
#         return text

#     found_citations = eyecite.get_citations(text)

#     output = []

#     for cit in found_citations:
#         # found_citations is a list of all the citations in the text
#         # for each match, we need to find the context of the citation
#         if cit.corrected_citation() == citation:
#             match = cit.matched_text()
#             sequence_matcher = difflib.SequenceMatcher(None, text, match)
#             match_info = sequence_matcher.find_longest_match(0, len(text), 0, len(match))

#             if match_info.size == 0:
#                 return ""

#             segments = text.split()
#             n_segs = len(segments)

#             start_segment_pos = len(text[:match_info.a].split())

#             words_before = words_before or n_segs
#             words_after = words_after or n_segs
#             start_pos = max(0, start_segment_pos - words_before)
#             end_pos = min(n_segs, start_segment_pos + words_after + len(citation.split()))

#             context = " ".join(segments[start_pos:end_pos])

#             # Use spaCy to adjust to complete sentences
#             nlp = spacy.load("en_core_web_sm")
#             doc = nlp(context)
#             sentences = list(doc.sents)

#             # Drop the first and last sentences if they are likely incomplete
#             adjusted_context = " ".join(sentence.text for sentence in sentences[1:-1])

#             output.append(adjusted_context)

#     return "\n\n".join(output[:max_excerpts])


def get_citation_context(
    text: str,
    citation: str,
    words_before: int = 100,
    words_after: int = 500,
    max_excerpts: int = 10,
) -> list[tuple[str, int, int, float]]:
    """Find contexts around a target legal citation and return spans as tuples.

    This function scans `text` for citations using eyecite, finds those that match
    the provided `citation` (based on `corrected_citation()`), then extracts
    surrounding context as word-based windows. It optionally (always, here) tries
    to align the context to complete sentences using spaCy, similar to
    `get_context_with_bm25`. The return type mirrors `get_context_with_bm25`:
    a list of tuples `(context, start_pos, end_pos, score)`. Since there is no
    BM25 scoring here, `score` is set to `1.0` for each matched excerpt.

    Args:
        text (str): The input text to search.
        citation (str): The normalized citation string to match against
            `cit.corrected_citation()`.
        words_before (int, optional): Max words before the match to include.
            Defaults to 1000.
        words_after (int, optional): Max words after the match to include.
            Defaults to 1000.
        max_excerpts (int, optional): Maximum number of excerpts to return.
            Defaults to 10.

    Returns:
        List[Tuple[str, int, int, float]]: Each tuple contains:
            - context (str): Extracted context, adjusted to full sentences when possible.
            - start_pos (int): Start word index in the original `text`.
            - end_pos (int): End word index in the original `text`.
            - score (float): Fixed to 1.0 for citation matches.

        Returns an empty list if no matches are found.

    """

    # If both bounds are None, return entire text as a single span (if non-empty)
    if words_after is None and words_before is None:
        segments = text.split()
        return [(text, 0, len(segments), 1.0)] if text.strip() else []

    found_citations = eyecite.get_citations(text)

    results: list[tuple[str, int, int, float]] = []

    for cit in found_citations:
        # Filter only the citations that match the requested normalized citation
        if cit.corrected_citation() != citation:
            continue

        # Use the actual matched text from eyecite to localize the span in `text`
        match = cit.matched_text()

        # Find the position of `match` within `text` using difflib
        sequence_matcher = difflib.SequenceMatcher(None, text, match)
        match_info = sequence_matcher.find_longest_match(0, len(text), 0, len(match))
        if match_info.size == 0:
            continue  # Skip if we cannot localize match

        # Compute word-based positions for the context window
        segments = text.split()
        n_segs = len(segments)

        # Words before the start of the matched region
        start_segment_pos = len(text[: match_info.a].split())

        # Adjust window sizes when None/zero-like
        words_before_adj = words_before or n_segs
        words_after_adj = words_after or n_segs

        # Word-based window boundaries
        start_pos = max(0, start_segment_pos - words_before_adj)
        end_pos = min(
            n_segs, start_segment_pos + words_after_adj + len(citation.split())
        )

        # Initial context
        context = " ".join(segments[start_pos:end_pos])

        # Try to align context to full sentences using spaCy
        if context.strip():
            try:
                nlp = spacy.load("en_core_web_sm")
                context_doc = nlp(context)
                context_sentences = list(context_doc.sents)

                # If >2 sentences, drop potentially incomplete first/last
                if len(context_sentences) > 2:
                    adjusted_context = " ".join(
                        s.text for s in context_sentences[1:-1]
                    ).strip()
                    if adjusted_context:
                        context = adjusted_context
                # If exactly 2 sentences, keep both
                elif len(context_sentences) == 2:
                    context = " ".join(s.text for s in context_sentences)
                # If 1 sentence, leave as-is
            except Exception:
                # On any spaCy error, fall back to the original context
                pass

        # Append tuple with a fixed score of 1.0 for citation matches
        results.append((context, start_pos, end_pos, 1.0))

    # Enforce max_excerpts limit
    return results[:max_excerpts]
