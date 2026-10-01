from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving meaningful content.
    """

    # Replace repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Remove spaces around brackets
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return text.strip()


def load_and_split_pdf(pdf_path: str):
    """
    Load a PDF and split it into overlapping chunks.

    Returns:
        documents: extracted page documents
        chunks: split document chunks
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    # Load PDF
    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    if not documents:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    # Clean extracted text
    for document in documents:
        document.page_content = clean_text(
            document.page_content
        )

    # Remove empty pages
    documents = [
        document
        for document in documents
        if document.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "The PDF does not contain extractable text."
        )

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    # Split documents
    chunks = text_splitter.split_documents(documents)

    # Add useful metadata
    for index, chunk in enumerate(chunks, start=1):

        chunk.metadata["chunk_id"] = index

        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = (
                chunk.metadata["page"] + 1
            )

    return documents, chunks