import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import anthropic

# Set page config at the very beginning
st.set_page_config(page_title="Document Chatter", layout="wide")

# CSS styling (updated to include centered header)
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #333333;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #333333;
        }
        [data-testid="stSidebarNav"] {
            background-color: #333333;
        }
        .css-1d391kg {
            background-color: #333333;
        }
        [data-testid="stSidebar"] img {
            border-radius: 10px;
        }
        .centered-header {
            text-align: center;
            font-size: 5em;
            font-weight: bold;
            padding-top: 20px;
            padding-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Display the banner image
st.image("chatbot_banner.png", use_column_width=True)

# Centered header
st.markdown('<p class="centered-header">Document Chatter</p>', unsafe_allow_html=True)

# Custom Anthropic Chat Model
class CustomAnthropicChatModel:
    def __init__(self, model, temperature, anthropic_api_key):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.temperature = temperature

    def __call__(self, prompt):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content

# API provider selection
api_provider = st.radio("STEP 1: Select API Provider:", ("OpenAI", "Anthropic"))

# API key input
if api_provider == "OpenAI":
    api_key = st.text_input("STEP 2: Enter your OpenAI API Key:", type="password", key="api_key_input")
else:
    api_key = st.text_input("STEP 2: Enter your Anthropic API Key:", type="password", key="api_key_input")

# Model selection
if api_provider == "OpenAI":
    model_options = ["gpt-3.5-turbo", "gpt-4"]
else:
    model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]

selected_model = st.selectbox("STEP 3: Select Model:", model_options)

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

def create_vector_store(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(api_key, model_name, api_provider):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    if api_provider == "OpenAI":
        model = ChatOpenAI(model_name=model_name, temperature=0.3, openai_api_key=api_key)
    else:
        model = CustomAnthropicChatModel(model=model_name, temperature=0.3, anthropic_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key, model_name, api_provider):
    if st.session_state.vector_store is None:
        st.error("Please upload and process documents first.")
        return

    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain(api_key, model_name, api_provider)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("Your personal PDF chatbot")

    # File upload
    pdf_docs = st.file_uploader("STEP 4: Upload your PDF Files here", accept_multiple_files=True, key="pdf_uploader")

    # Submit and Process Button
    st.markdown("STEP 5: Click on the button below")
    if st.button("Submit & Process", key="process_button") and api_key:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            st.session_state.vector_store = create_vector_store(text_chunks, api_key)
            st.success("Done")

    # Ask a Question section
    user_question = st.text_input("STEP 6: Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:
        user_input(user_question, api_key, selected_model, api_provider)

    with st.sidebar:
        st.image("document_chatter.jpg", use_column_width=True)

        st.markdown("""
        <div style="background-color: white; padding: 10px; border-radius: 10px;">
        <p style="color: black;">In order to use this document reader application, follow these steps:</p>
        <ol style="color: black;">
            <li>Select API provider</li>
            <li>Insert your API key</li>
            <li>Select your model type</li>
            <li>Upload your PDF document</li>
            <li>Click on "Submit and Process"</li>
            <li>Ask your question</li>
        </ol>
        <p style="color: black;"><strong>Disclaimer:</strong>Please note that this application is in beta. The developer of this application does not take any responsibility for any user wrongdoings.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()