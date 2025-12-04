from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader

# chunk
from langchain_text_splitters import CharacterTextSplitter


def main():
    load_dotenv()
    st.set_page_config(page_title='Ask your PDF')
    st.header("Ask Your PDF")


    # upload file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.write(chunks)

if __name__ == '__main__':
    main()