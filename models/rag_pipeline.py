import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self, vector_store_path: str = "vector_store"):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store_path: Path to the vector store.
        """
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        
        # Initialize the vector store if it exists
        if os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the vector store.
        
        Args:
            documents: A list of documents to add.
        """
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.create_documents(documents)
        
        # Create or update the vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
        
        # Save the vector store
        self.vector_store.save_local(self.vector_store_path)
    
    def add_documents_from_directory(self, directory: str):
        """
        Add documents from a directory to the vector store.
        
        Args:
            directory: Path to the directory containing documents.
        """
        # Load documents from the directory
        loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # Extract text from documents
        texts = [doc.page_content for doc in documents]
        
        # Add documents to the vector store
        self.add_documents(texts)
    
    def query(self, query: str, k: int = 5) -> List[str]:
        """
        Query the vector store.
        
        Args:
            query: The query to search for.
            k: The number of results to return.
            
        Returns:
            A list of relevant documents.
        """
        if self.vector_store is None:
            return []
        
        # Search the vector store
        results = self.vector_store.similarity_search(query, k=k)
        
        # Extract text from results
        texts = [doc.page_content for doc in results]
        
        return texts 