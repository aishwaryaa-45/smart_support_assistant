# Correct imports
import os
import streamlit as st
from typing import Optional, List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Updated LangChain imports (new structure)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.chains import RetrievalQA        # ✔️ updated
from langchain_core.language_models import LLM
from pydantic import PrivateAttr


# Load secrets (Streamlit Cloud)
HF_TOKEN = st.secrets.get("HF_TOKEN")

# Local fallback if running locally
if not HF_TOKEN:
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

# If still missing, stop
if not HF_TOKEN:
    st.error("HF_TOKEN not found. Set it in Streamlit secrets or .env file.")
    st.stop()


# Load and split FAQ documents
loader = TextLoader("faqs.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Embeddings + FAISS Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)


# ---------------------------
# Custom Hugging Face LLM (Updated)
# ---------------------------
class HFInferenceLLM(LLM):
    model_id: str = "tiiuae/falcon-rw-1b"   # ✔️ free model that supports text-generation
    token: str
    _client: InferenceClient = PrivateAttr()

    def model_post_init(self, __context=None):
        self._client = InferenceClient(model=self.model_id, token=self.token)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            response = self._client.text_generation(
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.7,
                stop_sequences=stop or []
            )
            return response.strip()
        except Exception as e:
            return f"Error from HF Inference API: {e}"

    @property
    def _llm_type(self) -> str:
        return "hf-inference-llm"

    @property
    def _identifying_params(self) -> dict:
        return {"model_id": self.model_id}


# Instantiate LLM
llm = HFInferenceLLM(token=HF_TOKEN)


# Retrieval QA chain (updated)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🤖 Smart Support Assistant")

user_input = st.text_input("Ask your support question:")

greetings = ["hello", "hi", "hey", "good morning", "good evening"]

if not user_input.strip():
    st.info("Please enter a support question.")

elif user_input.lower().strip() in greetings:
    st.success("Hello! How can I help you today?")

else:
    with st.spinner("Thinking..."):
        try:
            result = qa_chain.invoke(user_input)
            faq_answer = result["result"].strip()

            # If FAQ answer is useless → fallback
            if not faq_answer or len(faq_answer) < 8 or "i don't know" in faq_answer.lower():
                try:
                    fallback_prompt = (
                        "You are a concise customer support assistant.\n"
                        "Answer only the user's question clearly.\n\n"
                        f"User: {user_input}\nSupport:"
                    )

                    fallback_response = llm._client.text_generation(
                        prompt=fallback_prompt,
                        max_new_tokens=200,
                        temperature=0.7,
                        stop_sequences=["User:", "Customer:", "Support:"]
                    ).strip()

                    st.success(fallback_response)

                except Exception as e:
                    st.error(f"LLM Fallback Error: {e}")
            else:
                st.success(faq_answer)

        except Exception as e:
            st.error(f"Unexpected error: {e}")
