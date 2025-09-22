import difflib
import re
from typing import List, Tuple

try:
    import nltk
    from rank_bm25 import BM25Okapi
    import spacy
    BM25_AVAILABLE = True
    SPACY_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    SPACY_AVAILABLE = False


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
        raise ImportError("NLTK is not available. Please install nltk to use BM25 functionality.")
    
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
        raise ImportError("NLTK is not available. Please install nltk to use BM25 functionality.")
    
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
    
    Example:
        >>> text = "The quick brown fox jumps. This is sentence two. The fox is clever."
        >>> results = get_context_with_bm25("fox jumps", text, words_before=3, words_after=5, k=2)
        >>> for context, start, end, score in results:
        ...     print(f"Score: {score:.3f}, Context: {context}")
    """
    if not BM25_AVAILABLE:
        raise ImportError("BM25 dependencies not available. Please install rank-bm25 and nltk.")
    
    if not SPACY_AVAILABLE:
        raise ImportError("spaCy not available. Please install spacy and download the language model.")
    
    if not text.strip():
        return []

    try:
        # Load spaCy model
        nlp = spacy.load(spacy_model)
    except OSError:
        raise ImportError(f"spaCy model '{spacy_model}' not found. Please install it with: python -m spacy download {spacy_model}")

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
        start_segment_pos = len(text[:match_info.a].split())
        
        # Calculate word window boundaries
        words_before_adj = words_before or n_segs
        words_after_adj = words_after or n_segs
        start_pos = max(0, start_segment_pos - words_before_adj)
        end_pos = min(n_segs, start_segment_pos + words_after_adj + len(sentence.split()))
        
        # Extract initial context
        context = " ".join(segments[start_pos:end_pos])
        
        # Use spaCy to adjust to complete sentences if requested
        if adjust_to_sentences and context.strip():
            try:
                context_doc = nlp(context)
                context_sentences = list(context_doc.sents)
                
                # If we have more than 2 sentences, drop the first and last if they might be incomplete
                if len(context_sentences) > 2:
                    adjusted_context = " ".join(sentence.text for sentence in context_sentences[1:-1])
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