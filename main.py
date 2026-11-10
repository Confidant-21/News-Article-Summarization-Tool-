import os
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

# Streamlit UI Configuration
st.set_page_config(page_title="News Research Tool", page_icon="📈")
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Retrieve Gemini API Key securely from Streamlit Secrets
# (Make sure to add GOOGLE_GEMINI_KEY in your Streamlit Cloud app settings)
try:
    gemini_api_key = st.secrets["GOOGLE_GEMINI_KEY"]
except Exception:
    gemini_api_key = os.environ.get("GOOGLE_GEMINI_KEY", "")

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_api_key,
    temperature=0.7
)

# Custom wrapper for SentenceTransformer with Streamlit Caching for RAM optimization
class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = self.load_model(model_name)

    @staticmethod
    @st.cache_resource
    def load_model(model_name):
        return SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Initialize session state for vectorstore so it persists across UI interactions
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if process_url_clicked:
    # Filter out empty URLs
    valid_urls = [u for u in urls if u.strip()]
    if not valid_urls:
        st.error("Please enter at least one valid URL.")
    else:
        try:
            with st.spinner("Processing URLs..."):
                loader = UnstructuredURLLoader(urls=valid_urls)
                main_placeholder.text("Data Loading...Started...✅✅✅")
                data = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                main_placeholder.text("Text Splitting...Started...✅✅✅")
                docs = text_splitter.split_documents(data)

                embeddings = MiniLMEmbeddings()
                main_placeholder.text("Creating Embeddings and Building Index...✅✅✅")
                
                # Store directly in session state instead of local disk pickle
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

                main_placeholder.text("FAISS Index Built and Ready in Memory! 🚀")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

query = st.text_input("Question: ")
if query:
    if st.session_state.vectorstore is not None:
        try:
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
                        st.write(source)
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {e}")
    else:
        st.warning("Please enter URLs and click 'Process URLs' first to build the knowledge base.")