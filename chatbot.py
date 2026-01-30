from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def create_vectorstore(chunk):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunk, embedding=embeddings)
    
    return vectorstore


def create_conversation_chain(vectorstore):
    model_llm = ChatOpenAI(model='gpt-4o')
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm= model_llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )
    
    return conversation_chain
