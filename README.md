# Intelligent Q&A RAG System

This is a local knowledge base question-answering system based on Retrieval-Augmented Generation (RAG) technology. It loads local documents and leverages advanced language models to provide users with precise, citable, and streaming answers.

## ✨ Key Features

- **Modular Design**: The project has a clean structure with decoupled core components, making it easy to extend and maintain.
- **Streaming Response**: The LLM's answer is streamed word-by-word with a typewriter effect, significantly improving the user experience.
- **Detailed Source Citation**: Every answer is backed by comprehensive source references, including original text snippets, similarity scores, and the retriever type (e.g., Hybrid), ensuring transparency and trustworthiness.
- **Advanced Retrieval Strategies**: Includes multiple built-in retrievers, such as vector search, hybrid keyword-vector search, and ensemble methods, which can be configured dynamically through the UI.
- **Interactive Frontend**: Provides a full-featured web interface for document management, parameter configuration, conversational Q&A, and system status monitoring.
- **Performance & Caching**: Supports caching for text embeddings to avoid redundant computations and accelerate document processing.
- **API Endpoints**: Offers a complete set of RESTful APIs for easy integration with other systems.

## 💻 Web Interface

The system provides an intuitive web interface that consolidates all core functionalities.

![alt text](D:\LLms\Tiny-RAG\assets\1.png)

![alt text](D:\LLms\Tiny-RAG\assets\2.png)

**Main Dialogue Interface:** Users can upload documents, ask questions, and receive LLM-generated answers here.

**Answers and Source References:** Below the generated answer, all referenced source snippets are clearly listed with their similarity scores and retrieval details.

## 📂 Project Structure

```
RAG/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── config.yaml                    # Configuration file
├── app.py                         # Flask backend main file
├── run.py                         # Startup script
│
├── src/                           # Core source code
│   ├── document_loader.py        # Document loader
│   ├── text_splitter.py          # Text splitter
│   ├── embeddings.py             # Embedding module
│   ├── vector_store.py           # Vector store
│   ├── retriever.py              # Retriever
│   ├── llm_handler.py            # LLM handler (supports normal and streaming)
│   └── rag_pipeline.py           # Main RAG pipeline
│
├── templates/                     # HTML templates
│   └── index.html                # Frontend interface
│
├── data/                          # Data directory
│   ├── documents/                # Original documents
│   └── vector_db/                # Vector database
│
├── uploads/                       # Uploaded files
└── logs/                          # Log files
```

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Clone the Project

```
git clone <your-repository-url>
cd RAG
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Configure the System (Optional)

Open `config.yaml` to adjust models and core parameters. Most settings can also be configured dynamically in the web interface.

### 5. Start the System

```
python run.py
```

Once the server is running, open your browser and navigate to `http://127.0.0.1:5000` to start.

## 📝 How to Use

1. **Upload Documents**: In the "Document Management" panel on the left side of the web UI, select and upload your `.txt` or `.pdf` files. The system will automatically process and index them.
2. **Configure Parameters (Optional)**: In the top-left configuration area, you can adjust the number of retrieved results, the similarity threshold, etc. Click "Apply Configuration" to save changes.
3. **Ask a Question**: Type your question in the input box at the bottom and click "Send".
4. **Get an Answer**: The answer will appear in the dialogue panel with a streaming typewriter effect. Below the answer, you can find detailed "Reference Sources", which you can expand to see the original text, source, and relevance score for each citation.

## 🛠️ Core Components Explained

- **`document_loader.py`**: Loads raw documents from various sources (e.g., PDF, TXT).
- **`text_splitter.py`**: Splits long documents into smaller text chunks suitable for embedding models.
- **`embeddings.py`**: Converts text chunks into vector embeddings using models like Sentence-Transformers.
- **`vector_store.py`**: Stores vectors using FAISS and provides efficient similarity search capabilities.
- **`retriever.py`**: Retrieves the most relevant text chunks from the vector store based on a user's query.
- **`llm_handler_steam.py`**: A key component that enables streaming interaction with the LLM, allowing answers to be generated token by token.
- **`rag_pipeline.py`**: Orchestrates all the components above to form the complete RAG pipeline.
- **`app.py`**: The Flask application that provides the API endpoints and web service.

## 🔮 Future Work

- [ ] Support for more document formats (e.g., .docx, .md).
- [ ] Introduce more advanced retrieval optimization techniques, such as query rewriting and HyDE.
- [ ] Integrate knowledge graphs for hybrid retrieval.
- [ ] Provide visualized reports for comprehensive system evaluation metrics.