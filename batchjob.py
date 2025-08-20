from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss   #tells the vector size
from langchain_community.docstore.in_memory import InMemoryDocstore  #to store vector database temporary
from langchain_community.vectorstores import FAISS

#step1 load tcs report pdf file and store into documents

loader = PyPDFLoader(r"E:\finalproject\tcschatbot\documents\tcsreport1.pdf")
documents=loader.load()


print(len(documents))

#step2 convert documents into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2000,
    chunk_overlap=500,
)
chunks=text_splitter.split_documents(documents)

print("total chunks :",len(chunks))
#step3 create embedding model

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#step4 create vector fasis db
#below statement will give the total size of the vector
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

#below statement creates the vector database
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

#step5 store our chunks into vector db
vector_store.add_documents(chunks)
print("sucessfully created vector database")

#store vector db permanantly
vector_store.save_local("tcs_doc_index")

print("sucessfully store vector db locally")