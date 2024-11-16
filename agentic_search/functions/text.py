from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from nltk import data as nltk_data, download as nltk_download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from typing import List


def get_pdf_pages_docs(pdf_file_path: str) -> List[Document]:
    """Get the pages of a PDF document as a list of documents."""
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    return docs


def normalize_and_lemmatize_text(input_text: str) -> str:
    """
    Normalize and lemmatize the input text.
    """
    # check if required NLTK data files are already downloaded, if not, download them
    try:
        nltk_data.find("corpora/wordnet")
    except LookupError:
        nltk_download("wordnet")
    try:
        nltk_data.find("tokenizers/punkt")
    except LookupError:
        nltk_download("punkt")
    try:
        nltk_data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk_download("punkt_tab")
    try:
        nltk_data.find("corpora/stopwords")
    except LookupError:
        nltk_download("stopwords")

    # initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # step 1: lowercase the text
    # special handling for "US" before lowercase conversion
    input_text = re.sub(r"\bUS\b", "us", input_text)
    input_text = input_text.lower()

    # step 2: keep main currency symbols and join numbers with units, remove punctuation
    input_text = re.sub(
        r"(\d+)\s+(billion|million|thousand|B|M|Billion|Million|Thousand)",
        r"\1-\2",
        input_text,
    )
    # replace special characters with spaces before removing other punctuation
    input_text = re.sub(r"[/\-_*+=@#%|\\<>{}[\]~`]", " ", input_text)
    # remove punctuation but preserve currency symbols
    input_text = re.sub(r"[^\w\s$€£¥]", "", input_text)

    # step 3: tokenize the text
    tokens = word_tokenize(input_text)

    # step 4: remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # step 5: lemmatization - preserve "US" before lemmatizing
    tokens = [
        "us" if token == "us" else lemmatizer.lemmatize(token) for token in tokens
    ]

    # step 6: normalize whitespaces (join tokens back into a string)
    res = " ".join(tokens)

    # additional step: ensure currency symbols stick to numbers
    res = re.sub(r"(\$|\€|\£|\¥)\s+(\d+)", r"\1\2", res)

    return res
