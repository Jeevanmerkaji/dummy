from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS


# Extract the Data from the PDF file 

def loader_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob ="*.pdf",
        loader_cls = PyPDFLoader
    )

    documents = loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
        Given a list of the documents objects, return a new list of document objects
        containing only source in metadata and the original page content
    """
    minimal_docs : List[Document] =[]
    for doc in docs:
        src =  doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {'source': src}
            )
        )


    return minimal_docs



# split the Data into the text chunks 
def split_text(extracted_data):
    splitter =  RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = splitter.split_documents(extracted_data)
    return text_chunks


#Downloading the embeddings from the huggingface 
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings


# Store the data into the vector database
def store_to_vectordb():
    texts_chunks =  split_text
