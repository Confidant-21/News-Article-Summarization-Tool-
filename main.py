import os
import streamlit as st
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

# Streamlit UI Configuration
st.set_page_config(page_title="News Research Tool", page_icon="📈", layout="wide")
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs from sidebar
urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}") for i in range(3)]

process_url_clicked = st.sidebar.button("Process URLs", type="primary")
main_placeholder = st.empty()

# Securely retrieve Gemini API Key from Streamlit Secrets or environment variables
try:
    gemini_api_key = st.secrets["GOOGLE_GEMINI_KEY"]
except Exception:
    gemini_api_key = os.environ.get("GOOGLE_GEMINI_KEY", "")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_api_key,
    temperature=0.7
)

# Custom wrapper for SentenceTransformer optimized with Streamlit caching
class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = self.load_model(model_name)

    @staticmethod
    @st.cache_resource
    def load_model(model_name: str) -> SentenceTransformer:
        return SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()

# Manage vector store persistence within Streamlit session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if process_url_clicked:
    valid_urls = [u.strip() for u in urls if u.strip()]
    if not valid_urls:
        st.sidebar.error("Please enter at least one valid URL.")
    else:
        try:
            with st.spinner("Scraping and processing URLs..."):
                # Use WebBaseLoader for lightweight, reliable, cloud-friendly parsing
                loader = WebBaseLoader(web_paths=valid_urls)
                main_placeholder.text("Data Loading via WebBaseLoader... Started... ✅✅✅")
                data = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000,
                    chunk_overlap=200
                )
                main_placeholder.text("Text Splitting... Started... ✅✅✅")
                docs = text_splitter.split_documents(data)

                embeddings = MiniLMEmbeddings()
                main_placeholder.text("Creating Embeddings & Building FAISS Index... ✅✅✅")
                
                # Build and store vector store in memory session state
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

                main_placeholder.text("FAISS Index Built Successfully in Memory! 🚀")
        except Exception as e:
            st.error(f"An error occurred during pipeline execution: {e}")

# User Query Interface
query = st.text_input("Ask a question about the articles:")
if query:
    if st.session_state.vectorstore is not None:
        try:
            with st.spinner("Synthesizing answer with Gemini..."):
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, 
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                result = chain({"question": query}, return_only_outputs=True)

                st.header("Answer")
                st.write(result.get("answer", "No answer found."))

                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    for source in sources.split("\n"):
                        if source.strip():
                            st.markdown(f"- {source}")
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {e}")
    else:
        st.warning("⚠️ Please enter URLs in the sidebar and click 'Process URLs' first to build the knowledge base.")