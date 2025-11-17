# Correct imports
import os
import streamlit as st
from typing import Optional, List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain_core.language_models import LLM
from pydantic import PrivateAttr


load_dotenv()  
HF_TOKEN = st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN not found. Please set it in a .env file or environment variables.")
    st.stop()

# Debug print
print("HF Token Loaded:", "Yes" if HF_TOKEN else "No")

# Load and split documents
loader = TextLoader("faqs.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
for doc in documents:
    print(doc.page_content[:50])

# Embeddings and vector store

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Custom LLM using Hugging Face Inference API (chat model version)
class HFInferenceLLM(LLM):
    model_id: str = "HuggingFaceH4/zephyr-7b-beta"
    token: str
    _client: InferenceClient = PrivateAttr()

    def model_post_init(self, __context=None):
        self._client = InferenceClient(model=self.model_id, token=self.token)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            response = self._client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.7,
                stop=stop
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error from HF Inference API: {e}"

    @property
    def _llm_type(self) -> str:
        return "custom-hf-chat-inference"

    @property
    def _identifying_params(self) -> dict:
        return {"model_id": self.model_id}

# Instantiate custom LLM
llm = HFInferenceLLM(token=HF_TOKEN)

# Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Streamlit UI
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

            if not faq_answer or "i don't know" in faq_answer.lower() or len(faq_answer) < 10:
                try:
                    fallback_response = llm._client.chat_completion(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful and concise customer support assistant. "
                                    "Only respond to the user's question. Do not ask questions or add unrelated information."
                                )
                            },
                            {"role": "user", "content": user_input}
                        ],
                        max_tokens=256,
                        temperature=0.7,
                        stop=["\nUser:", "\nCustomer:", "\nSupport:"]
                    ).choices[0].message["content"].strip()

                    st.success(fallback_response)

                except Exception as e:
                    st.error(f"LLM Fallback Error: {e}")
            else:
                st.success(faq_answer)

        except Exception as e:
            st.error(f"Unexpected error: {e}")

