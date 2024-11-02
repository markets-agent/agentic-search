from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List


def get_pdf_pages_docs(pdf_file_path: str) -> List[Document]:
    """Get the pages of a PDF document as a list of documents."""
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    return docs
