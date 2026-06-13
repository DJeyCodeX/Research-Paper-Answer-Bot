from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import  LLMChainFilter
from langchain.chains import RetrievalQA
from langchain.globals import set_verbose
from langchain_core.documents import Document
import httpx
import logging
from typing import Any, List, Dict
import numpy as np
import re
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
import tempfile
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

# set_debug(True)
set_verbose(True)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

llm = AzureChatOpenAI(model_name="gpt-4o", temperature=0, deployment_name = 'gpt-4o', http_client=httpx.Client(verify=False))

openai_embed_model = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-large",openai_api_version="2024-02-01", http_client=httpx.Client(verify=False))

class ResearchPaperRAG:
    """
    A comprehensive RAG system for research paper question answering.
    Supports multiple embedding models, retrieval strategies, and evaluation metrics.
    """
    
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.embedding_model = None
        self.llm = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="result"
        )
        
        # Available embedding models
        self.embedding_models = {
            "BAAI/bge-small-en": "BAAI/bge-small-en",
            "openai": "text-embedding-ada-002"
        }

    def load_documents(self, pdf_directory: str) -> List[Document]:
            """Load PDF documents from directory"""
            try:
                loader = DirectoryLoader(
                    pdf_directory,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader,
                    show_progress=True
                )
                documents = loader.load()
                
                # Clean and preprocess documents
                for doc in documents:
                    # Clean text
                    doc.page_content = re.sub(r'\s+', ' ', doc.page_content)
                    doc.page_content = doc.page_content.strip()
                    
                    # Add metadata
                    if 'source' in doc.metadata:
                        doc.metadata['filename'] = os.path.basename(doc.metadata['source'])
                        doc.metadata['loaded_at'] = datetime.now().isoformat()
                
                self.documents = documents
                logger.info(f"Loaded {len(documents)} documents")
                return documents
                
            except Exception as e:
                logger.error(f"Error loading documents: {str(e)}")
                raise

    def split_documents(self, documents: List[Document], chunk_size: int = 1000, 
                        chunk_overlap: int = 200) -> List[Document]:
            """Split documents into chunks"""
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                chunks = text_splitter.split_documents(documents)
                
                # Add chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = i
                    chunk.metadata['chunk_size'] = len(chunk.page_content)
                
                logger.info(f"Split into {len(chunks)} chunks")
                return chunks
                
            except Exception as e:
                logger.error(f"Error splitting documents: {str(e)}")
                raise

    def create_embeddings(self, model_name: str) -> Any:
            """Create embedding model"""
            try:
                if model_name == "openai":
                    embeddings = openai_embed_model
                else:
                    # Use BGE embedding model from Hugging Face
                    embeddings = HuggingFaceEmbeddings(
                        model_name="BAAI/bge-small-en",
                        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                
                self.embedding_model = embeddings
                logger.info(f"Created embedding model: {model_name}")
                return embeddings
                
            except Exception as e:
                logger.error(f"Error creating embeddings: {str(e)}")
                raise

    def create_vector_store(self, chunks: List[Document], embeddings: Any) -> FAISS:
        """Create FAISS vector store"""
        try:
            vector_store = FAISS.from_documents(chunks, embeddings)
            self.vector_store = vector_store
            logger.info("Created FAISS vector store")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
        
    def save_vector_store(self, vector_store: FAISS, path: str):
        """Save vector store to disk"""
        try:
            vector_store.save_local(path)
            logger.info(f"Saved vector store to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    def load_vector_store(self, path: str, embeddings: Any) -> FAISS:
        """Load vector store from disk"""
        try:
            vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            self.vector_store = vector_store
            logger.info(f"Loaded vector store from {path}")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def create_retriever(self, k: int = 5) -> Any:
        """Create retriever based on strategy"""
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            compressor = LLMChainFilter.from_llm(llm)
            final_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
            self.retriever = final_retriever
            logger.info(f"Created retriever ")
            return retriever
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            raise

    def create_qa_chain(self, llm: Any, retriever: Any) -> RetrievalQA:
            """Create QA chain"""
            try:
                # Custom prompt template
                template = """Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Always cite the source documents used in your answer.

                Context: {context}

                Question: {question}

                Answer: """
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt},
                    memory=self.memory
                )
                
                self.qa_chain = qa_chain
                logger.info("Created QA chain")
                return qa_chain
                
            except Exception as e:
                logger.error(f"Error creating QA chain: {str(e)}")
                raise

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer question and return sources"""
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not created")
            
            result = self.qa_chain({"query": question})
            
            # Extract source information
            sources = []
            if 'source_documents' in result:
                for i, doc in enumerate(result['source_documents'][:3]):  # Top 3 sources
                    source_info = {
                        'rank': i + 1,
                        'filename': doc.metadata.get('filename', 'Unknown'),
                        'page': doc.metadata.get('page', 'Unknown'),
                        'chunk_id': doc.metadata.get('chunk_id', 'Unknown'),
                        'content_preview': doc.page_content[:200] + "..."
                    }
                    sources.append(source_info)
            
            return {
                'answer': result['result'],
                'sources': sources,
                'query': question
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise

    def evaluate_retrieval(self, test_queries: List[str], 
                            ground_truth: List[List[str]] = None) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        try:
            if not self.retriever:
                raise ValueError("Retriever not created")
            
            # Simple evaluation - count relevant documents retrieved
            results = []
            for query in test_queries:
                docs = self.retriever.get_relevant_documents(query)
                results.append(len(docs))
            
            avg_docs_retrieved = np.mean(results)
            
            return {
                'avg_documents_retrieved': avg_docs_retrieved,
                'total_queries': len(test_queries)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval: {str(e)}")
            raise

def main():
    st.set_page_config(
        page_title="Research Paper Answer Bot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("🔬 Research Paper Answer Bot")
    st.markdown("*Powered by RAG (Retrieval-Augmented Generation)*")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = ResearchPaperRAG()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # # API Key input
    # openai_api_key = st.sidebar.text_input(
    #     "OpenAI API Key (optional)",
    #     type="password",
    #     help="Required for OpenAI embeddings and GPT models"
    # )
    
    # Model selection
    embedding_model = st.sidebar.selectbox(
        "Select Embedding Model",
        options=list(st.session_state.rag_system.embedding_models.keys()),
        help="Choose the embedding model for document encoding"
    )
    
    
    # Parameters
    chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, 50)
    k_retrieval = st.sidebar.slider("Number of Documents to Retrieve", 1, 10, 5)
    
    # File upload
    st.header("📁 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload your research papers in PDF format"
    )
    
    if uploaded_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Process documents
            if st.button("🔄 Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        # Load documents
                        documents = st.session_state.rag_system.load_documents(temp_dir)
                        
                        # Split documents
                        chunks = st.session_state.rag_system.split_documents(
                            documents, chunk_size, chunk_overlap
                        )
                        
                        # Create embeddings
                        embeddings = st.session_state.rag_system.create_embeddings(
                            embedding_model
                        )
                        
                        # Create vector store
                        vector_store = st.session_state.rag_system.create_vector_store(
                            chunks, embeddings
                        )
                        
                        # Create retriever
                        retriever = st.session_state.rag_system.create_retriever(
                            k_retrieval
                        )
                        
                        # # Create LLM
                        # llm = st.session_state.rag_system.create_llm(
                        #     "gpt-3.5-turbo" if openai_api_key else "text-davinci-003",
                        #     openai_api_key
                        # )
                        
                        # Create QA chain
                        qa_chain = st.session_state.rag_system.create_qa_chain(llm, retriever)
                        
                        st.success(f"✅ Successfully processed documents into {len(chunks)} chunks!")
                        st.session_state.documents_processed = True
                        
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
    
    # Question answering section
    if st.session_state.get('documents_processed', False):
        st.header("💬 Ask Questions")
        
        # Sample questions
        sample_questions = [
            "What are the main components of a transformer architecture?",
            "How does attention mechanism work in neural networks?",
            "What are the advantages of using pre-trained language models?",
            "Explain the concept of transfer learning in NLP.",
            "What are the challenges in training large language models?"
        ]
        
        # Question input
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="Ask anything about the research papers..."
            )
        with col2:
            sample_question = st.selectbox(
                "Or choose a sample:",
                [""] + sample_questions
            )
        
        if sample_question:
            question = sample_question
        
        if question and st.button("🔍 Get Answer"):
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.rag_system.answer_question(question)
                    
                    # Display answer
                    st.subheader("📖 Answer")
                    st.write(result['answer'])
                    
                    # Display sources
                    st.subheader("📚 Sources")
                    for source in result['sources']:
                        with st.expander(f"📄 Source {source['rank']}: {source['filename']} (Page {source['page']})"):
                            st.write("**Content Preview:**")
                            st.write(source['content_preview'])
                            st.write(f"**Chunk ID:** {source['chunk_id']}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
    
    # Evaluation section
    st.header("📊 System Evaluation")
    
    if st.session_state.get('documents_processed', False):
        if st.button("🧪 Run Evaluation"):
            test_queries = [
            "What is the attention mechanism in transformers?",
            "How does BERT differ from GPT?",
            "What are the key innovations in the Transformer architecture?",
            "How does self-attention work?",
            "What is the difference between encoder and decoder architectures?"
        ]
            
            with st.spinner("Running evaluation..."):
                try:
                    eval_results = st.session_state.rag_system.evaluate_retrieval(test_queries)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Documents Retrieved", f"{eval_results['avg_documents_retrieved']:.2f}")
                    with col2:
                        st.metric("Total Test Queries", eval_results['total_queries'])
                    
                except Exception as e:
                    st.error(f"❌ Error running evaluation: {str(e)}")
    
    # System information
    st.sidebar.header("System Information")
    st.sidebar.info(f"""
    **Current Configuration:**
    - Embedding Model: {embedding_model}
    - Chunk Size: {chunk_size}
    - Chunk Overlap: {chunk_overlap}
    - Retrieval K: {k_retrieval}
    """)
    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()