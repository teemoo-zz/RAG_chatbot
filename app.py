import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Document Chatter", layout="wide")

st.markdown("""
# Document Chatter
""")

# OpenAI API key input
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", key="openai_api_key_input")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4", "gpt-4-1106-preview"]
selected_model = st.sidebar.selectbox("Select OpenAI Model:", model_options)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(openai_api_key, model_name):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatOpenAI(model_name=model_name, temperature=0.3, openai_api_key=openai_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, openai_api_key, model_name):
    if st.session_state.vector_store is None:
        st.error("Please upload and process documents first.")
        return

    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain(openai_api_key, model_name)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("AI Document Chatbot 💬")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and openai_api_key:
        user_input(user_question, openai_api_key, selected_model)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and openai_api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = create_vector_store(text_chunks, openai_api_key)
                st.success("Done")

if __name__ == "__main__":
    main()
