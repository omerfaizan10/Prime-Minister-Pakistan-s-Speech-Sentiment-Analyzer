import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# NLP tools
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
from textblob import TextBlob

# NLTK Downloads
nltk.download('wordnet')
nltk.download('stopwords')

# Streamlit Page Config
st.set_page_config(page_title="Shehbaz Sharif Speech Sentiment", layout="wide")


# Page session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Load NRC Emotion Lexicon
@st.cache_data
def load_emotion_lexicon():
    lexicon = {}
    with open("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", "r") as f:
        for line in f:
            word, emotion, association = line.strip().split("\t")
            if int(association) == 1:
                lexicon.setdefault(word, []).append(emotion)
    return lexicon

# Load CSV Data
@st.cache_data
def load_data():
    df = pd.read_csv('sentiment_speeches.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'VADER_Label' not in df.columns:
        def label_sentiment(score):
            if score >= 0.05:
                return 'positive'
            elif score <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        df['VADER_Label'] = df['VADER_Compound'].apply(label_sentiment)
    return df

# Emotion Detection Function (Safe)
def detect_emotions(text):
    if not isinstance(text, str):
        return {}
    tokens = re.findall(r'\b\w+\b', text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha()]
    emotions = Counter()
    for word in tokens:
        if word in emotion_lexicon:
            emotions.update(emotion_lexicon[word])
    return dict(emotions)

# Load Data and Lexicon
df = load_data()
emotion_lexicon = load_emotion_lexicon()

# Only run once
if 'Emotions' not in df.columns:
    df['Emotions'] = df['Content'].apply(detect_emotions)
    df.to_csv("sentiment_speeches.csv", index=False)

# -----------------------
# 🏠 Welcome Page
# -----------------------
if st.session_state.page == 'welcome':
    st.title("🇵🇰 Welcome to Shehbaz Sharif's Speech Sentiment Analyzer created by EngD. Omer Faizan")
    st.markdown("""
    This app uses NLP to analyze **sentiments and emotions** in Shehbaz Sharif’s speeches:

    ### 🔍 Capabilities:
    - Detect emotions: love, anger, patriotism, joy, etc.
    - Show sentiment trends (positive/negative/neutral)
    - Visualize emotional tone across all speeches

    ---
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/77/Flag_of_Pakistan.svg", width=120)

    if st.button("🚀 Start Analysis"):
        st.session_state.page = 'dashboard'
        st.rerun()

# -----------------------
# 📊 Dashboard Page
# -----------------------
elif st.session_state.page == 'dashboard':
    st.title("📊 Speech Sentiment & Emotion Dashboard")

    sentiment_option = st.radio(
        "Filter by Sentiment:",
        options=['All', 'positive', 'neutral', 'negative'],
        horizontal=True
    )

    if sentiment_option != 'All':
        filtered_df = df[df['VADER_Label'] == sentiment_option]
    else:
        filtered_df = df

    st.subheader("🗂 Filtered Speeches")
    st.dataframe(filtered_df[['Date', 'Title', 'VADER_Compound', 'VADER_Label']])

    # Emotion Frequency Chart
    st.subheader("🎭 Dominant Emotions Across Speeches")
    all_emotions = Counter()
    for emo_dict in filtered_df['Emotions']:
        all_emotions.update(emo_dict)

    top_emotions = dict(all_emotions.most_common(10))

    fig, ax = plt.subplots()
    sns.barplot(x=list(top_emotions.keys()), y=list(top_emotions.values()), palette='coolwarm', ax=ax)
    ax.set_title("Top Emotions in Selected Speeches")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Emotion")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 🔍 Search by Keyword
    st.subheader("🔍 Search and Read Speeches by Keyword")
    search_query = st.text_input("Enter keyword (e.g., economy, flood, investment)")

    if search_query:
        keyword_df = filtered_df[
            filtered_df['Title'].str.contains(search_query, case=False, na=False) |
            filtered_df['Content'].str.contains(search_query, case=False, na=False)
        ]

        if keyword_df.empty:
            st.warning(f"No speeches found containing '{search_query}'.")
        else:
            st.success(f"Found {len(keyword_df)} speeches containing '{search_query}'")
            selected_title = st.selectbox("Select a speech to read:", keyword_df['Title'].tolist())
            selected_content = keyword_df[keyword_df['Title'] == selected_title]['Content'].values[0]

            st.markdown(f"### 🗣️ {selected_title}")
            st.write(selected_content)

    # 📈 Sentiment Trendline
    st.subheader("📈 Average Sentiment Over Time")
    trend_df = filtered_df.copy()
    trend_df = trend_df.dropna(subset=['Date'])
    trend_df['Month'] = trend_df['Date'].dt.to_period('M').dt.to_timestamp()
    monthly_avg = trend_df.groupby('Month')['VADER_Compound'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=monthly_avg, x='Month', y='VADER_Compound', marker='o', ax=ax)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title("Monthly Average VADER Sentiment Score")
    ax.set_ylabel("Average Sentiment")
    ax.set_xlabel("Month")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Back button
    if st.button("⬅️ Back to Welcome Page"):
        st.session_state.page = 'welcome'
        st.rerun()
    # 📊 Individual Speech Emotion Radar
    st.subheader("🕵️‍♂️ Emotion Profile of a Single Speech")
    speech_titles = filtered_df['Title'].tolist()

    selected_speech = st.selectbox("Select a speech for emotion analysis:", speech_titles)

    if selected_speech:
        speech_row = filtered_df[filtered_df['Title'] == selected_speech]
        emotion_dict = speech_row['Emotions'].values[0]

        if isinstance(emotion_dict, str):
            import ast

            emotion_dict = ast.literal_eval(emotion_dict)

        if emotion_dict:
            # Radar chart
            labels = list(emotion_dict.keys())
            values = list(emotion_dict.values())
            values += values[:1]  # Loop closure

            angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

            fig, ax = plt.subplots(subplot_kw=dict(polar=True))
            ax.plot(angles, values, marker='o')
            ax.fill(angles, values, alpha=0.3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title(f"Emotion Radar for: {selected_speech}")
            st.pyplot(fig)
        else:
            st.info("No emotion data found for this speech.")
    # 📊 Emotion Radar per Speech
    st.subheader("🎯 Emotion Radar for Individual Speech")

    speech_options = filtered_df['Title'].tolist()
    selected_radar_title = st.selectbox("Choose a speech to visualize its emotional profile:", speech_options)

    if selected_radar_title:
        speech_row = filtered_df[filtered_df['Title'] == selected_radar_title]
        emotions_raw = speech_row['Emotions'].values[0]

        # Convert stringified dict back to actual dictionary (in case it's stored as a string)
        if isinstance(emotions_raw, str):
            import ast

            emotions = ast.literal_eval(emotions_raw)
        else:
            emotions = emotions_raw

        if emotions:
            labels = list(emotions.keys())
            values = list(emotions.values())
            values += values[:1]  # close the loop for radar

            angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.plot(angles, values, marker='o', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title(f"Emotional Intensity for:\n{selected_radar_title}", fontsize=13)
            st.pyplot(fig)
        else:
            st.warning("No emotions detected in this speech.")
if st.button("⬅️ Back to Welcome Page"):
        st.session_state.page = 'welcome'
        st.rerun()

     
