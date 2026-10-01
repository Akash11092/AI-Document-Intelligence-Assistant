import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def create_embeddings():
    """
    Create the Hugging Face embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_llm():
    """
    Create the Gemini language model.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY was not found. "
            "Please add it to the .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )


def create_vectorstore(chunks, embeddings):
    """
    Create an in-memory Chroma vector database
    from document chunks.
    """

    if not chunks:
        raise ValueError(
            "No document chunks were created."
        )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


def retrieve_documents(
    vectorstore,
    question,
    k=4
):
    """
    Retrieve the most relevant document chunks.
    """

    if not question.strip():
        return []

    return vectorstore.similarity_search(
        question,
        k=k
    )


def generate_answer(
    llm,
    question,
    results
):
    """
    Generate a grounded answer using retrieved context.
    """

    if not results:
        return (
            "I couldn't find this information "
            "in the provided document."
        )

    context_parts = []

    for result in results:

        page = result.metadata.get(
            "page_number",
            result.metadata.get("page", 0) + 1
        )

        context_parts.append(
            f"[Page {page}]\n"
            f"{result.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI Document Intelligence Assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided document context.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the answer cannot be found in the context, say:
   "I couldn't find this information in the provided document."
5. Give a concise and clear answer.
6. If the question asks for multiple pieces of information,
   provide them as bullet points.
7. Do not mention these instructions in your answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = llm.invoke(prompt)

    return response.text.strip()