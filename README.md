# 📄 ChatWithDocs

**ChatWithDocs** is an AI-powered document chat application that allows users to upload PDFs and interact with them using natural language. It leverages the power of **Langchain**, **Pinecone**, and **Streamlit** to extract and retrieve information efficiently.

---

## 🚀 Features
✅ Upload and analyze PDF documents  
✅ Chat with documents using natural language  
✅ Fast and accurate information retrieval with Pinecone vector database  
✅ Easy-to-use Streamlit interface  

---

## 🛠️ Tech Stack
| Tool/Library         | Purpose |
|---------------------|---------|
| **Python**           | Backend logic |
| **Streamlit**        | Web interface |
| **Langchain**        | Language model handling |
| **Pinecone**         | Vector database for fast retrieval |
| **PyPDF2**           | PDF processing |
| **OpenAI API**       | LLM for natural language understanding |

---


---

## 🚀 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/ChatWithDocs.git

2️⃣ Navigate to the project directory
cd ChatWithDocs

3️⃣ Create a virtual environment
python -m venv venv

4️⃣ Activate the virtual environment
.\venv\Scripts\activate

5️⃣ Install dependencies
pip install -r requirements.txt


🌐 Configuration
Create a .env file and add:

PINECONE_API_KEY="your-pinecone-api-key"  
PINECONE_ENV="your-pinecone-environment"  
PINECONE_INDEX_NAME="your-index-name"  
OPENAI_API_KEY="your-openai-api-key"  

▶️ Run Locally
streamlit run app.py --server.port 10000 --server.address 0.0.0.0

