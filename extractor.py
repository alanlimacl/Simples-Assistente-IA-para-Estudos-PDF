from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader

def extract_pdf(lista_pdf):
    text = ''
    
    for pdf in lista_pdf:
        reader_pdf = PdfReader(pdf)
        
        for page in reader_pdf.pages:
            text += page.extract_text()
    
    return text


def extract_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 1200,
        chunk_overlap = 200,
        length_function = len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

