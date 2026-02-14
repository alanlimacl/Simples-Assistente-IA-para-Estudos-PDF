from extractor import extract_pdf, extract_chunks
from chatbot import create_vectorstore, create_conversation_chain
from streamlit_chat import message
from visual import tittle_html
import streamlit as st

def main():
    st.set_page_config(page_title='Oráculo', page_icon='🔮')
    st.markdown(tittle_html, unsafe_allow_html=True)
    
    user_question = st.chat_input(placeholder='Como posso ajudar?')
    
    if('conversation' not in st.session_state):
        st.session_state.conversation = None
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if (user_question):
        if st.session_state.conversation:
            
            try:
                response = st.session_state.conversation.invoke({'question': user_question})
            
            except AttributeError:
                response = st.session_state.conversation({'question': user_question})

            ai_answer = response['answer']
            
            st.session_state.chat_history.append({'role': 'user', 'content': user_question})
            st.session_state.chat_history.append({'role': 'ai', 'content': ai_answer})
            
        else:
            st.warning("Por favor, envie um PDF primeiro para iniciar a conversa.")

    chat_container = st.container()
    
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            
            if msg['role'] == 'user':
                message(msg['content'], is_user=True, key=f"msg_{i}_user")
            
            else:
                message(msg['content'], is_user=False, key=f"msg_{i}_ai")
    
    with st.sidebar:
        
        st.title('Seus Arquivos')
        pdf_docs = st.file_uploader(
            "Selecione o arquivo (PDF) e clique em enviar.",
            type=['pdf'],
            accept_multiple_files=True)

        if st.button('Enviar'):
            if pdf_docs is not None:
                files_name = ', '.join([pdf.name for pdf in pdf_docs])
                with st.spinner(text=f'Processando "{files_name}"... Por favor aguarde.'):
                    all_text_pdf = extract_pdf(pdf_docs)
                    
                    chunk = extract_chunks(all_text_pdf)
                    
                    vectorstore = create_vectorstore(chunk)
                
                    st.session_state.conversation = create_conversation_chain(vectorstore)
                    
                    st.success('PDF Lido com sucesso!')


if __name__ == '__main__':
    import os
    caminho_deste_arquivo = os.path.abspath(__file__)
    print(caminho_deste_arquivo)
    
    PASTA_ATUAL = os.path.dirname(caminho_deste_arquivo)
    print(PASTA_ATUAL)
    
          