from typing import List, Dict
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os

class RAGPipeline:
    def __init__(self, vector_db_path: str = "vector_db"):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_db_path: Path to store the vector database
        """
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def initialize_vector_store(self):
        """Initialize or load the vector store."""
        if os.path.exists(self.vector_db_path):
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
    
    def add_documents(self, file_paths: List[str]):
        """
        Add documents to the vector store.
        
        Args:
            file_paths: List of paths to text files
        """
        if not self.vector_store:
            self.initialize_vector_store()
        
        for file_path in file_paths:
            loader = TextLoader(file_path)
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)
            self.vector_store.add_documents(texts)
        self.vector_store.persist()
    
    def query(self, query: str, k: int = 3) -> List[Dict]:
        """
        Query the vector store for relevant context.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of relevant documents with their metadata
        """
        if not self.vector_store:
            self.initialize_vector_store()
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
    def get_context_for_translation(self, tokens: List[str], k: int = 3) -> str:
        """
        Get relevant context for ASL translation.
        
        Args:
            tokens: List of ASL tokens
            k: Number of context documents to retrieve
            
        Returns:
            Concatenated context string
        """
        query = " ".join(tokens)
        results = self.query(query, k=k)
        context = "\n".join(result["content"] for result in results)
        return context

# Create a singleton instance
rag_pipeline = RAGPipeline() 