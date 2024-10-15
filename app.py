import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.docstore.document import Document
import anthropic
import re
import json

# Set page config at the very beginning
st.set_page_config(page_title="Essay Evaluator", layout="wide")

# CSS styling
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
        .rounded-image {
            border-radius: 15px;
            overflow: hidden;
        }
        .rounded-image img {
            border-radius: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Centered header
st.markdown('<p class="centered-header">Essay Evaluator</p>', unsafe_allow_html=True)

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

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None
if 'evaluation_displayed' not in st.session_state:
    st.session_state.evaluation_displayed = False
if 'combined_knowledge_base' not in st.session_state:
    st.session_state.combined_knowledge_base = None

# Add these new session state variables
if 'button_outputs' not in st.session_state:
    st.session_state.button_outputs = {
        'essay_info': None,
        'exec_summary': None,
        'strengths': None,
        'weaknesses': None,
        'grammar': None,
        'feedback': None,
        'detailed': None
    }
if 'current_output' not in st.session_state:
    st.session_state.current_output = None   

# Helper functions
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
    Use the following pieces of context to answer the question at the end. 
    The context includes both the original document content and an evaluation of the document.
    Provide a detailed answer based on all available information.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

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
    if st.session_state.combined_knowledge_base is None:
        st.error("Please generate an evaluation first before asking questions.")
        return

    docs = st.session_state.combined_knowledge_base.similarity_search(user_question, k=4)
    chain = get_conversational_chain(api_key, model_name, api_provider)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def format_prompt_for_model(prompt, api_provider):
    if api_provider == "OpenAI":
        return [HumanMessage(content=prompt)]
    else:  # Anthropic
        return prompt











def generate_essay_info(api_key, model_name, api_provider):
    if st.session_state.vector_store is None:
        st.error("Please upload and process documents first.")
        return

    all_docs = st.session_state.vector_store.similarity_search("", k=1000)
    full_text = " ".join([doc.page_content for doc in all_docs])

    prompt = f"""
    Based on the essay content provided below, please extract the following information:
    - Author name
    - Date
    - Title
    - Class/Course
    - Word Count
    - Student ID [The Student ID has mostly 8-9 digits]

    If you cannot find any of the information, leave it blank.

    Essay content:
    {full_text}

    Please provide the information in a structured format.
    """

    formatted_prompt = format_prompt_for_model(prompt, api_provider)
    model = get_model(api_key, model_name, api_provider)
    response = model(formatted_prompt)
    return get_response_content(response, api_provider)

def generate_executive_summary(api_key, model_name, api_provider):
    if st.session_state.vector_store is None:
        st.error("Please upload and process documents first.")
        return

    all_docs = st.session_state.vector_store.similarity_search("", k=1000)
    full_text = " ".join([doc.page_content for doc in all_docs])

    prompt = f"""
    Based on the essay content provided below, please provide a 7 to 10 sentence long summary of the uploaded essay.

    Essay content:
    {full_text}

    Please ensure the summary captures the main points and arguments of the essay concisely.
    """

    formatted_prompt = format_prompt_for_model(prompt, api_provider)
    model = get_model(api_key, model_name, api_provider)
    response = model(formatted_prompt)
    return get_response_content(response, api_provider)

def generate_specific_evaluation(api_key, model_name, api_provider, evaluation_type):
    if st.session_state.vector_store is None:
        st.error("Please upload and process documents first.")
        return

    all_docs = st.session_state.vector_store.similarity_search("", k=1000)
    full_text = " ".join([doc.page_content for doc in all_docs])

    prompt_templates = {
        "strengths": "List 5 to 7 key points detailing the positive aspects of the essay. Be precise and give the professor a good overview of what the student did well in writing the essay. There is no need for a summary at the end, only list the key points.",
        "weaknesses": "List 5 to 7 key points detailing the negative aspects of the essay. Be precise and give the professor a good overview of what the student did not do well in writing the essay. There is no need for a summary at the end, only list the key points.",
        "grammar": "Provide 5 to 7 key points focusing on grammar and essay structure. Include both positive and negative aspects, highlighting what could have been done better and what was done well. There is no need for a summary at the end, only list the key points."
    }

    prompt = f"""
    You are an experienced professor in the university English language department evaluating a student essay. Your feedback should be direct, specific, and constructive.

    Based on the essay content provided below, please {prompt_templates[evaluation_type]}

    Essay content:
    {full_text}

    Please base your evaluation solely on the provided essay content. Be direct and specific in your feedback, focusing on how the student can improve their writing. Use present tense when discussing what the author is doing in the essay.
    """

    formatted_prompt = format_prompt_for_model(prompt, api_provider)
    model = get_model(api_key, model_name, api_provider)
    response = model(formatted_prompt)
    return get_response_content(response, api_provider)

def generate_detailed_feedback(api_key, model_name, api_provider):
    if st.session_state.vector_store is None:
        st.error("Please upload and process documents first.")
        return

    all_docs = st.session_state.vector_store.similarity_search("", k=1000)
    full_text = " ".join([doc.page_content for doc in all_docs])

    prompt = f"""
    You are an experienced professor in the university English language department evaluating a student essay. Your feedback should be direct, specific, and constructive, mimicking the style of the following examples.
    Remember that the comment examples below are just a guidance for you to mimic the professor's style and tone of writing. Come up with your own comments and improvement points.
    
    - "Your header should have your last name and page number."
    - "Improper heading. Your heading should be on the left. You also need a header in the upper right-hand of your paper."
    - "You need a header in the upper right-hand of your paper. Your MLA-style header should have your last name and page number."
    - "Your title should give your audience an idea of your paper’s topic as well as your stance."
    - "Where is your introduction? Both this paragraph and the next have information that should be in the summary. Also, if you know the date of publication, then give it, as it helps your readers place the issue in context of time."
    - "Where is your informative thesis statement for your summary? You still need it although you might leave out a few elements."
    - "Remember that in a summary, you have to be true to the original. Be sure you cover the entire contents of the original article."
    - "If your thesis is not clearly stated, then it sets you up for failure in the body of your paper because your body paragraphs are supposed to develop and support your thesis."
    - "Paragraphs could be improved by focusing on clearer topic sentences and transitions. More support and development, as well as commentary, would make for stronger body paragraphs. Many body paragraphs seemed more like summaries."
    - "You need a more effective transitional sentence here that perhaps reminds readers what you just discussed in the paragraph above as well as introduces your second point"
    - "You are being repetitive. Try to end on a more memorable note."
    - "Good transition plus topic sentence here!"
    - "Think about the genre of this article. It’s not an academic essay."
    - "This sounds like your opinion. Remember that in a summary, you need to ensure that you are only providing objective information as relayed by the original author."
    - "Remember that your audience tends to remember the point you make last. Whatever you want to emphasize, you should present last in your thesis statement, and you should follow that order when you write your body paragraphs."
    - "Don't separate a paragraph just because you think it's long. You need to ask yourself, 'Are all of the sentences discussing one idea -- the idea that was outlined in the topic sentence? Are all of the sentences necessary?' Remember, 1 paragraph, 1 idea."

    Based on the essay content provided below, please provide a detailed evaluation covering all the following aspects:

    - HEADING / HEADER
    - TITLE
    - INTRODUCTION
    - SUMMARY
    - THESIS STATEMENT
    - ESSAY BODY
    - CONCLUSION
    - WORKS CITED
    - STYLE, MECHANICS, CONVENTIONS

    Essay content:
    {full_text}

    Please base your evaluation solely on the provided essay content. Be direct and specific in your feedback, focusing on how the student can improve their writing. Use present tense when discussing what the author is doing in the essay. Provide unique comments and improvement ideas while maintaining the directness and specificity seen in the sample feedback.
    """

    formatted_prompt = format_prompt_for_model(prompt, api_provider)
    model = get_model(api_key, model_name, api_provider)
    response = model(formatted_prompt)
    return get_response_content(response, api_provider)

def get_model(api_key, model_name, api_provider):
    if api_provider == "OpenAI":
        return ChatOpenAI(model_name=model_name, temperature=0.3, openai_api_key=api_key)
    else:
        return CustomAnthropicChatModel(model=model_name, temperature=0.3, anthropic_api_key=api_key)

def get_response_content(response, api_provider):
    if api_provider == "OpenAI":
        return response.content
    else:
        return response













def create_combined_knowledge_base(api_key):
    original_docs = st.session_state.vector_store.similarity_search("", k=1000)
    evaluation_doc = [Document(page_content=st.session_state.evaluation_result)]
    combined_docs = original_docs + evaluation_doc
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    combined_texts = [doc.page_content for doc in combined_docs]
    combined_knowledge_base = FAISS.from_texts(combined_texts, embedding=embeddings)
    
    return combined_knowledge_base

def generate_and_save_output(key, generate_func, *args):
    if st.session_state.button_outputs[key] is None:
        result = generate_func(*args)
        st.session_state.button_outputs[key] = result
    st.session_state.current_output = key


def create_combined_knowledge_base(api_key):
    if st.session_state.vector_store is None:
        return None

    original_docs = st.session_state.vector_store.similarity_search("", k=1000)
    evaluation_docs = []
    for key, value in st.session_state.button_outputs.items():
        if value is not None:
            evaluation_docs.append(Document(page_content=f"{key}: {value}"))
    
    combined_docs = original_docs + evaluation_docs
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    combined_texts = [doc.page_content for doc in combined_docs]
    combined_knowledge_base = FAISS.from_texts(combined_texts, embedding=embeddings)
    
    return combined_knowledge_base

def user_input(user_question, api_key, model_name, api_provider):
    combined_knowledge_base = create_combined_knowledge_base(api_key)
    
    if combined_knowledge_base is None:
        st.error("Please upload a document and generate at least one evaluation before asking questions.")
        return

    docs = combined_knowledge_base.similarity_search(user_question, k=4)
    chain = get_conversational_chain(api_key, model_name, api_provider)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


def main():
    # API provider selection
    api_provider = st.radio("STEP 1: Select API Provider:", ("OpenAI", "Anthropic"))

    # API key input
    if api_provider == "OpenAI":
        api_key = st.text_input("STEP 2: Enter your OpenAI API Key:", type="password", key="api_key_input")
    else:
        api_key = st.text_input("STEP 2: Enter your Anthropic API Key:", type="password", key="api_key_input")

    # Model selection
    if api_provider == "OpenAI":
        model_options = ["gpt-4-0125-preview", "gpt-4"]
    else:
        model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]

    selected_model = st.selectbox("STEP 3: Select Model:", model_options)

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

    # Buttons for generating specific parts of the evaluation
    st.markdown("STEP 6: Generate Evaluation Components")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if st.button("Essay Info", key="essay_info"):
            generate_and_save_output('essay_info', generate_essay_info, api_key, selected_model, api_provider)

    with col2:
        if st.button("Executive Summary", key="exec_summary"):
            generate_and_save_output('exec_summary', generate_executive_summary, api_key, selected_model, api_provider)

    with col3:
        if st.button("Essay Strengths", key="strengths"):
            generate_and_save_output('strengths', generate_specific_evaluation, api_key, selected_model, api_provider, "strengths")

    with col4:
        if st.button("Essay Weaknesses", key="weaknesses"):
            generate_and_save_output('weaknesses', generate_specific_evaluation, api_key, selected_model, api_provider, "weaknesses")

    with col5:
        if st.button("Grammar & Structure", key="grammar"):
            generate_and_save_output('grammar', generate_specific_evaluation, api_key, selected_model, api_provider, "grammar")

    with col6:
        if st.button("Detailed (indiv.) Feedback", key="detailed"):
            generate_and_save_output('detailed', generate_detailed_feedback, api_key, selected_model, api_provider)


    # Display current output
    if st.session_state.current_output:
        output_key = st.session_state.current_output
        st.markdown(f"## {output_key.replace('_', ' ').title()}")
        st.write(st.session_state.button_outputs[output_key])
    
    # Ask a Question section
    st.markdown("STEP 7: Ask a Question")
    user_question = st.text_input("Enter your question here (you can ask about the essay or the evaluations):", key="user_question")

    if user_question and api_key:
        st.markdown("## Question and Answer")
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
            <li>Generate Evaluation Components</li>
            <li>Ask your question</li>
        </ol>
        <p style="color: black;"><strong>Disclaimer:</strong>Please note that this application is in beta. The developer of this application does not take any responsibility for any user wrongdoings.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()