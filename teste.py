from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
from langchain.embeddings import LlamaCppEmbeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

loader = TextLoader('ARQUIVO.txt',encoding = 'UTF-8')
print(loader)
documents = loader.load()
#print(documents[0])


#text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
#texts = text_splitter.split_documents(documents)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap = 0,
)

import os 
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
print('---------')
#embeddings = OpenAIEmbeddings()


#embeddings = model.encode(texts)
texto = text_splitter.split_documents(documents)


from langchain.embeddings import HuggingFaceEmbeddings
from InstructorEmbedding import INSTRUCTOR

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(texto,embeddings)



query = 'Qual objetivo de Voldemort'
texto = db.similarity_search(query)
print(texto)

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm = HuggingFaceHub(repo_id='01-ai/Yi-34B',model_kwargs = {'temperature':0,'max_length':512})

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type='stuff',
                                 retriever = db.as_retriever())

print(qa.run('Qual obejtivo de Voldemort?'))


#llm = Ollama(model="llama2")
#chain = load_qa_chain(llm,chain_type='stuff')

#texto = db.similarity_search(query)
#print(chain.run(input_document=texto,question=query))




#vectordb = Chroma.from_documents(documents=texto, 
                                 #embedding=llama,
                                 #persist_directory=persist_directory)

#retriever = vectordb.as_retriever(search_kwargs={"k": 3})

#qa = RetrievalQA.from_chain_type(llm=llm,
 #                                chain_type='stuff',
  #                               retriever = retriever.as_retriever())





print('Quem foi Harry Potter')
'''
docsearch = FAISS.from_documents(texts,embeddings)

 

qa = RetrievalQA.from_chain_type(llm = OpenAI(),
chai_type ='stuff',retriever=docsearch.as_retriever())

print(qa.run('Quem foi Harry Potter'))'''