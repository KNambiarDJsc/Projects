import streamlit as st
import pickle
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import dill


with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made by Karthik Nambiar')

load_dotenv()

def main():
    
    st.header("Chat with PDF 💬")
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        VectorStore = None
        if os.path.exists(f"{store_name}.pkl"):
            try:
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = dill.load(f)
            except EOFError:
                print("Error: The file is empty or corrupted.")
        else :
            embeddings = OpenAIEmbeddings(openai_api_key = "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")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            try:
                with open(f"{store_name}.pkl", "wb") as f:
                    dill.dump(VectorStore, f)
            except Exception as e:
                print(f"Error: {e}")


        query = st.text_input("Ask questions about your PDF file:")
        docs = []
        if VectorStore is not None:
            
            docs = VectorStore.similarity_search(query=query, k=3)
        else:
            # Handle the case when VectorStore is None
            print("VectorStore is not assigned.")

        if query:

            llm = OpenAI(openai_api_key = "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")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)

        





if __name__ == '__main__':
    main()