import streamlit as st
from deep_translator import GoogleTranslator
import os
import glob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import torch
from datetime import datetime
import re

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    # Using multilingual model to handle both English and Gujarati
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)

def translate_to_gujarati(word):
    try:
        translator = GoogleTranslator(source='en', target='gu')
        translation = translator.translate(word)
        return translation
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

def parse_date_from_filename(filename):
    # Extract date from filename format: Gujarat_Samachar_05-12-2024_page14_part2_text.txt
    match = re.search(r'(\d{2}-\d{2}-\d{4})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%d-%m-%Y')
        except ValueError:
            return None
    return None

def get_all_text_files(start_date=None, end_date=None):
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

def create_embeddings(model, articles_data):
    embeddings = []
    for article in articles_data:
        embedding = model.encode(article['content'], convert_to_tensor=True)
        embeddings.append(embedding)
    return torch.stack(embeddings)

def load_or_create_embeddings(model, start_date=None, end_date=None):
    # Create a cache key based on the date range
    date_range_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}" if start_date and end_date else "all"
    cache_file = f"data/embeddings_cache_{date_range_str}.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            return cached_data['embeddings'], cached_data['articles_data']

    articles_data = []
    text_files = get_all_text_files(start_date, end_date)

    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                articles = content.split('//')

                for article in articles:
                    if article.strip():
                        articles_data.append({
                            'file': os.path.basename(file_path),
                            'content': article.strip()
                        })

        except Exception as e:
            st.error(f"Error reading {file_path}: {str(e)}")
            continue

    embeddings = create_embeddings(model, articles_data)

    # Cache the embeddings
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'articles_data': articles_data
        }, f)

    return embeddings, articles_data

def semantic_search(query, model, embeddings, articles_data, threshold=0.3):
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding.cpu().numpy().reshape(1, -1),
        embeddings.cpu().numpy()
    )[0]

    # Get indices of articles above threshold, sorted by similarity
    relevant_indices = np.where(similarities >= threshold)[0]
    sorted_indices = relevant_indices[np.argsort(-similarities[relevant_indices])]

    results = []
    for idx in sorted_indices:
        results.append({
            'file': articles_data[idx]['file'],
            'content': articles_data[idx]['content'],
            'similarity': similarities[idx]
        })

    return results

def display_results(results):
    if results:
        st.subheader(f"Found {len(results)} Relevant Articles:")

        # Group results by file
        files_dict = {}
        for result in results:
            if result['file'] not in files_dict:
                files_dict[result['file']] = []
            files_dict[result['file']].append((result['content'], result['similarity']))

        # Display results grouped by file
        for file_name, articles in files_dict.items():
            with st.expander(f"📄 {file_name} ({len(articles)} articles)"):
                for idx, (article, similarity) in enumerate(articles, 1):
                    st.markdown(f"**Article {idx}** (Relevance: {similarity:.2%})")
                    st.write(article)
                    if idx < len(articles):
                        st.markdown("---")
    else:
        st.warning("No relevant articles found.")

def main():
    st.title("AI-Powered Gujarati News Search")
    st.write("Semantic search for news articles in Gujarati")

    # Add date range filters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", 
                                  value=datetime.now().replace(day=1),
                                  format="DD/MM/YYYY")
    with col2:
        end_date = st.date_input("End Date", 
                                value=datetime.now(),
                                format="DD/MM/YYYY")

    if start_date > end_date:
        st.error("Error: Start date must be before end date")
        return

    # Convert date inputs to datetime objects for comparison
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Load the model and embeddings with date filter
    with st.spinner("Loading AI model..."):
        model = load_model()
        embeddings, articles_data = load_or_create_embeddings(model, start_datetime, end_datetime)

    # Display available text files
    text_files = get_all_text_files(start_datetime, end_datetime)
    if text_files:
        with st.expander("Available Text Files"):
            for file in text_files:
                file_date = parse_date_from_filename(os.path.basename(file))
                date_str = file_date.strftime('%d-%m-%Y') if file_date else 'Unknown date'
                st.write(f"📄 {os.path.basename(file)} ({date_str})")
    else:
        st.warning(f"No articles found between {start_date.strftime('%d-%m-%Y')} and {end_date.strftime('%d-%m-%Y')}")
        return

    # Add radio button for search type
    search_type = st.radio(
        "Choose search method:",
        ["English to Gujarati", "Direct Gujarati Input"],
        horizontal=True
    )

    # Add similarity threshold slider
    threshold = st.slider(
        "Relevance Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Adjust the minimum similarity score for matching articles"
    )

    if search_type == "English to Gujarati":
        search_word = st.text_input("Enter text to search (in English):")

        if search_word:
            with st.spinner("Translating..."):
                gujarati_word = translate_to_gujarati(search_word)

            if gujarati_word:
                st.success(f"Translated text: {gujarati_word}")

                if st.button("Search", key="english_search"):
                    with st.spinner("Performing semantic search..."):
                        results = semantic_search(gujarati_word, model, embeddings, articles_data, threshold)
                        display_results(results)

    else:  # Direct Gujarati Input
        st.markdown("""
        💡 **Tip:** To type in Gujarati:
        - Use Google Translate keyboard
        - Use Windows Gujarati keyboard
        - Copy and paste Gujarati text
        """)

        gujarati_word = st.text_input("Enter text in Gujarati:", key="gujarati_input")

        if gujarati_word:
            if st.button("Search", key="gujarati_search"):
                with st.spinner("Performing semantic search..."):
                    results = semantic_search(gujarati_word, model, embeddings, articles_data, threshold)
                    display_results(results)

    # Add footer with instructions
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    1. Select the date range for your search
    2. Choose your search method
    3. Adjust the relevance threshold as needed
    4. Enter your search text
    5. Click 'Search' to find semantically related articles
    6. Results are sorted by relevance score
    """)

    # Add information about keyboard support
    with st.expander("ℹ️ How to type in Gujarati"):
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

        4. **Copy & Paste:**
           - Use any online Gujarati typing tool
           - Copy the text and paste it here
        """)

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# - data/embeddings_cache_YYYYMMDD_YYYYMMDD.pkl (created for each unique date range when running for the first time)
