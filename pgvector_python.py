import os
import pandas as pd
from typing import List

from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain_community.vectorstores import PGVector
from langchain.docstore.document import Document


CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver='psycopg2',
    host=os.environ.get('PGVECTOR_HOST'),
    port=int(os.environ.get('PGVECTOR_PORT')),
    database=os.environ.get('PGVECTOR_DATABASE'),
    user=os.environ.get('PGVECTOR_USER'),
    password=os.environ.get('PGVECTOR_PASSWORD'),
)

# Configure embeddings model
embeddings = NomicEmbeddings(
    model="nomic-embed-text-v1.5"
    )

def load_documents_from_csv(csv_path: str) -> List[Document]:
    """
    Load documents from a CSV file with 'title' and 'name' columns.
    'title' becomes page_content and 'name' becomes metadata category.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of Document objects
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = {'title', 'name'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f'Missing required columns: {missing}')
        
        # Convert DataFrame rows to Documents
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=str(row['title']),
                metadata={'category': str(row['name'])}
            )
            documents.append(doc)
            
        print(f'Successfully loaded {len(documents)} documents from {csv_path}')
        return documents
    
    except Exception as e:
        print(f'Error loading documents from CSV: {str(e)}')
        raise

def create_pgvector_store(
    docs: List[Document],
    collection_name: str = 'documents',
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE
) -> PGVector:
    """
    Create a PGVector store with the given documents.
    
    Args:
        docs: List of Document objects to store
        collection_name: Name for the collection in the database
        distance_strategy: Distance strategy for similarity search
        
    Returns:
        PGVector store instance
    """
    try:
        # Create new PGVector instance
        vector_store = PGVector.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=CONNECTION_STRING,
            distance_strategy=distance_strategy,
        )
        
        print(f'Successfully created PGVector store with {len(docs)} documents')
        return vector_store
    
    except Exception as e:
        print(f'Error creating PGVector store: {str(e)}')
        raise

def search_similar_documents(
    vector_store: PGVector,
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Search for similar documents in the vector store.
    
    Args:
        vector_store: PGVector store instance
        query: Search query string
        k: Number of results to return
        
    Returns:
        List of similar documents
    """
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    
    except Exception as e:
        print(f'Error searching documents: {str(e)}')
        raise


# Example usage
if __name__ == '__main__':
    # Sample documents
    sample_docs = load_documents_from_csv('sample_kredivo_faq.csv')
    
    # Create vector store
    vector_store = create_pgvector_store(sample_docs, collection_name='sample_kredivo_faq')
    
    # Search for similar documents
    query = 'bagaimana cara ganti pin?'
    similar_docs = search_similar_documents(vector_store, query)
    
    # Print results
    for doc in similar_docs:
        print(f'Similar document: {doc.page_content}')