from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader

# chunk
from langchain_text_splitters import CharacterTextSplitter

# Embedding,
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Semantic Search
from langchain_community.vectorstores import FAISS

# 
from langchain_community.callbacks import get_openai_callback

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

        # ✅ Chunking
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=200,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)


        # ✅ Embedding
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

        # Semantic Search
        knowledge_base = FAISS.from_texts(
            texts = chunks, 
            embedding=embeddings)

        # show user input
        user_question = st.text_input("Tanyakan apapun yang ada di PDF")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            context = "\n\n".join(d.page_content for d in docs)
            
            # ✅ LLM
            llm = ChatOpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
            )
            
            # Document Question Answering
            prompt = (
                "Kamu adalah asisten yang menjawab berdasarkan isi PDF. "
                "Jika informasi tidak ada di konteks, jawab dengan jujur bahwa informasi tersebut tidak ada.\n\n"
                f"Berikut konteks dari PDF:\n\n{context}\n\n"
                f"Pertanyaan: {user_question}\n\n"
                "Jawab dalam bahasa Indonesia yang jelas dan singkat."
            )

            with get_openai_callback() as cb:
                response = llm.invoke(prompt)
                print(cb)


            st.write(response.content)
         

if __name__ == '__main__':
    main()