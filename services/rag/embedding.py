
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-mpnet-base-v2",
        model_kwargs = model_kwargs
    )
    return embeddings