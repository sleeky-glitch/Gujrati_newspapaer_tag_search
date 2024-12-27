import streamlit as st
from deep_translator import GoogleTranslator
import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import torch
from datetime import datetime
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import tiktoken

# Initialize OpenAI API key from Streamlit secrets
if 'OPENAI_API_KEY' not in st.secrets:
    st.error('OPENAI_API_KEY not found in Streamlit secrets. Please add it to your app settings.')
    st.stop()

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

@st.cache_data(ttl=3600)  # Cache for 1 hour
def translate_text(text, source='en', target='gu'):
    try:
        translator = GoogleTranslator(source=source, target=target)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

def parse_date_from_filename(filename):
    match = re.search(r'(\d{2}-\d{2}-\d{4})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%d-%m-%Y')
        except ValueError:
            return None
    return None

@st.cache_data(ttl=3600)
def get_text_files(start_date=None, end_date=None):
    try:
        text_files = glob.glob("data/*.txt")
        if start_date and end_date:
            filtered_files = []
            for file_path in text_files:
                file_date = parse_date_from_filename(os.path.basename(file_path))
                if file_date and start_date <= file_date <= end_date:
                    filtered_files.append(file_path)
            return filtered_files
        return text_files
    except Exception as e:
        st.error(f"Error reading directory: {str(e)}")
        return []

@st.cache_resource
def create_index(_text_files):
    # Combine all text files
    combined_text = ""
    for file_path in _text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                combined_text += file.read() + "\n\n"
        except Exception as e:
            st.error(f"Error reading {file_path}: {str(e)}")
            continue

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    texts = text_splitter.split_text(combined_text)

    # Create embeddings and index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)

    return vectorstore

@st.cache_resource
def setup_qa_chain(vectorstore):
    # Create prompt template
    template = """You are a helpful bilingual assistant that provides answers in both English and Gujarati.
    Use the following context to answer the question. If you don't know the answer, just say you don't know.

    Context: {context}

    Question: {question}

    Please provide your answer in both English and Gujarati, clearly labeled as:
    English:
    [Your English answer here]

    ગુજરાતી:
    [Your Gujarati answer here]
    """

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Setup QA chain
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.2,
        max_tokens=512
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain

@st.cache_resource
def initialize_qa_system(_text_files):
    vectorstore = create_index(_text_files)
    qa_chain = setup_qa_chain(vectorstore)
    return qa_chain

def main():
    st.title("AI-Powered Bilingual News Q&A System")
    st.write("Ask questions about news articles in English or Gujarati")

    # Initialize session state for storing conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().replace(day=1),
            format="DD/MM/YYYY"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            format="DD/MM/YYYY"
        )

    if start_date > end_date:
        st.error("Error: Start date must be before end date")
        return

    # Convert dates to datetime
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Get text files and initialize QA system
    text_files = get_text_files(start_datetime, end_datetime)
    if not text_files:
        st.warning(f"No articles found between {start_date.strftime('%d-%m-%Y')} and {end_date.strftime('%d-%m-%Y')}")
        return

    with st.spinner("Initializing QA System..."):
        qa_chain = initialize_qa_system(tuple(text_files))

    # Display available files
    with st.expander("Available Text Files"):
        for file in text_files:
            file_date = parse_date_from_filename(os.path.basename(file))
            date_str = file_date.strftime('%d-%m-%Y') if file_date else 'Unknown date'
            st.write(f"📄 {os.path.basename(file)} ({date_str})")

    # Language selection
    input_language = st.radio(
        "Choose input language:",
        ["English", "Gujarati"],
        horizontal=True
    )

    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="Type your question here..."
    )

    if question and st.button("Get Answer"):
        with st.spinner("Processing your question..."):
            # Translate question to English if it's in Gujarati
            if input_language == "Gujarati":
                question_en = translate_text(question, source='gu', target='en')
                if not question_en:
                    st.error("Failed to translate question")
                    return
            else:
                question_en = question

            # Get answer from QA chain
            response = qa_chain({"query": question_en})
            answer = response['result']

            # Add to conversation history
            st.session_state.conversation_history.append({
                "question": question,
                "answer": answer
            })

            # Display results
            st.subheader("Answer:")
            st.write(answer)

    # Display conversation history
    if st.session_state.conversation_history:
        with st.expander("Conversation History", expanded=False):
            for i, exchange in enumerate(st.session_state.conversation_history, 1):
                st.markdown(f"**Q{i}: {exchange['question']}**")
                st.markdown(f"A{i}: {exchange['answer']}")
                st.markdown("---")

    # Add instructions
    with st.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        **How to use this Q&A system:**
        1. Select the date range for your search
        2. Choose your preferred input language
        3. Type your question about the news articles
        4. Click 'Get Answer' to receive a bilingual response
        5. View your conversation history below the answer

        **Tips for better results:**
        - Be specific in your questions
        - Include relevant dates or topics
        - Questions can be about any content in the selected date range
        """)

    # Gujarati typing help
    with st.expander("🔤 How to type in Gujarati", expanded=False):
        st.markdown("""
        **Options for typing in Gujarati:**
        1. **Google Input Tools:**
           - Visit [Google Input Tools](https://www.google.com/inputtools/try/)
           - Select Gujarati
           - Type and copy the text

        2. **Windows Gujarati Keyboard:**
           - Go to Windows Settings > Time & Language > Language
           - Add Gujarati as a language
           - Use the language bar to switch to Gujarati keyboard

        3. **Mobile Device:**
           - Install Gujarati keyboard from your app store
           - Switch to Gujarati input method
           - Type and share/copy the text
        """)

if __name__ == "__main__":
    main()
