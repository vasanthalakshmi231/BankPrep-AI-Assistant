
import os
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. CONFIGURATION
# ============================================================

PDF_DIR = Path("data")
CHROMA_DIR = "bankprep_chroma"
COLLECTION_NAME = "bankprep_knowledge"

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"

TOP_K = 3


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BankPrep AI",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 BankPrep AI")
st.caption(
    "AI-powered banking exam preparation assistant "
    "| SBI | IBPS | RRB | RBI"
)


# ============================================================
# 3. API KEY CONFIGURATION
# ============================================================



f=open(r"C:\Users\siva9\OneDrive\Desktop\GenAI_Project\Google_gemini_API_key.txt")
API_KEY = f.read()

#API_KEY = get_api_key()

if not API_KEY:
    st.error(
        "Gemini API key not found. "
        "Set GEMINI_API_KEY in Command Prompt "
        "before running the application."
    )
    st.stop()


# ============================================================
# 4. RAG PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are BankPrep AI, a helpful banking exam preparation assistant.

Answer the student's question using the supplied study material.

Context:
{context}

Question:
{question}

Instructions:
1. Explain the answer in simple, beginner-friendly language.
2. For calculations, show the steps clearly.
3. For aptitude questions, explain the formula and solution.
4. For reasoning questions, explain the logic step by step.
5. For banking awareness, explain important concepts clearly.
6. Do not invent information or claim that unsupported
   information comes from the uploaded documents.
7. If the answer is not available in the context, say:
   "I could not find this information in the provided
   study materials."
8. Format the answer using headings and bullet points
   wherever helpful.
"""


# ============================================================
# 5. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 6. LOAD OR CREATE CHROMA VECTOR DATABASE
# ============================================================

@st.cache_resource(show_spinner="Loading banking study materials...")
def load_vectorstore(api_key):

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=api_key,
    )

    # Reuse the existing database if it contains documents
    if Path(CHROMA_DIR).exists():
        try:
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            )

            if vectorstore._collection.count() > 0:
                return vectorstore

        except Exception:
            pass

    # Find PDF files, including those in subfolders
    pdf_files = list(PDF_DIR.rglob("*.pdf")) if PDF_DIR.exists() else []

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in the 'data' folder. "
            "Create the folder and add your banking study PDFs."
        )

    # Load PDF documents
    loader = DirectoryLoader(
        str(PDF_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=False,
    )

    documents = loader.load()

    if not documents:
        raise ValueError(
            "No readable text was extracted from the PDFs."
        )

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text chunks were created from the uploaded PDFs."
        )

    # Create and persist Chroma database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore


# ============================================================
# 7. CREATE RAG CHAIN
# ============================================================

@st.cache_resource(show_spinner="Preparing BankPrep AI...")
def load_chain(api_key):

    vectorstore = load_vectorstore(api_key)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )

    def format_docs(docs):
        return "\n\n".join(
            f"Source: {Path(d.metadata.get('source', 'Unknown')).name}\n"
            f"Page: {int(d.metadata.get('page', 0)) + 1}\n"
            f"Content: {d.page_content}"
            for d in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🏦 BankPrep AI")

    st.write(
        "Your personal banking exam preparation assistant."
    )

    st.divider()

    st.subheader("📚 Exam Categories")

    category = st.selectbox(
        "Choose a preparation category",
        [
            "All Categories",
            "Quantitative Aptitude",
            "Logical Reasoning",
            "English Language",
            "Banking Awareness",
            "Computer Knowledge",
            "General Awareness",
            "Mock Test",
        ],
    )

    category_questions = {
        "Quantitative Aptitude":
            "Explain percentage calculations with examples.",
        "Logical Reasoning":
            "Explain seating arrangement reasoning questions.",
        "English Language":
            "Explain common English grammar rules for banking exams.",
        "Banking Awareness":
            "Explain important banking terms and concepts.",
        "Computer Knowledge":
            "Explain computer fundamentals for banking exams.",
        "General Awareness":
            "Explain general awareness topics in the study material.",
        "Mock Test":
            "Create practice questions based on the study material.",
    }

    st.divider()

    show_sources = st.checkbox(
        "Show source pages",
        value=True,
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "Answers are generated from your indexed study PDFs."
    )


# ============================================================
# 9. LOAD THE RAG SYSTEM
# ============================================================

try:
    chain, retriever = load_chain(API_KEY)

except Exception as error:
    st.error(f"Could not initialize BankPrep AI: {error}")

    st.info(
        "Check your Gemini API key, internet connection, "
        "installed libraries, and the PDFs in your data folder."
    )

    st.stop()


# ============================================================
# 10. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources") and show_sources:
            with st.expander("📑 Sources"):
                for source in message["sources"]:
                    st.write(source)


# ============================================================
# 11. ASK A QUESTION
# ============================================================

default_question = category_questions.get(category, "")

with st.expander("💡 Example question", expanded=False):
    if default_question:
        st.write(default_question)

question = st.chat_input(
    "Ask a banking exam question..."
)


# ============================================================
# 12. GENERATE ANSWER
# ============================================================

if question:

    # Display the user's question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve sources and generate an answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching study materials and generating an answer..."
        ):

            try:
                relevant_docs = retriever.invoke(question)

                if not relevant_docs:
                    answer = (
                        "I could not find this information "
                        "in the provided study materials."
                    )
                    sources = []

                else:
                    answer = chain.invoke(question)

                    sources = []

                    for document in relevant_docs:

                        filename = Path(
                            document.metadata.get(
                                "source", "Unknown PDF"
                            )
                        ).name

                        page_number = document.metadata.get("page")

                        if page_number is not None:
                            source = (
                                f"{filename} — Page "
                                f"{int(page_number) + 1}"
                            )
                        else:
                            source = filename

                        if source not in sources:
                            sources.append(source)

            except Exception as error:
                answer = f"An error occurred: {error}"
                sources = []

        st.markdown(answer)

        if sources and show_sources:
            with st.expander("📑 Sources", expanded=True):
                for source in sources:
                    st.write(f"📄 {source}")

    # Save the answer in chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "BankPrep AI | Streamlit + LangChain + ChromaDB + Gemini"
)