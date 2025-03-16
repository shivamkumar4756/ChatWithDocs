import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")  # Example: "us-west-2"
pinecone_index_name = "pdf-index"

if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.error("Google API Key not found! Please set it in the .env file.")
    st.stop()

if pinecone_api_key:
    pc = Pinecone(api_key=pinecone_api_key)
    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=768,  
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=pinecone_env)
        )
    index = pc.Index(pinecone_index_name)
else:
    st.error("Pinecone API Key not found! Please set it in the .env file.")
    st.stop()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = LangchainPinecone.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        index_name=pinecone_index_name
    )
    st.success("✅ Embeddings stored in Pinecone!")

def get_conversational_chain(retriever):
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say "Answer not available in the context."
    
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = LangchainPinecone.from_existing_index(pinecone_index_name, embeddings)
        retriever = vector_store.as_retriever()
        chain = get_conversational_chain(retriever)
        response = chain({"query": user_question})
        st.write("**Reply:**", response["result"])
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    st.set_page_config(page_title="Chat with PDF using Gemini", layout="wide")
    st.header("📄 ChatWithDocs")

    user_question = st.text_input("Ask a question from the PDF files:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload PDF Files:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("🚀 Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Files processed and stored in Pinecone!")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
