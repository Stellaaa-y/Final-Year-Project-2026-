import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - NO pop-up windows
import matplotlib.pyplot as plt
from transformers import pipeline
from collections import defaultdict

# ==================== NLTK DOWNLOAD (run once) ====================
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ==================== MODULE-LEVEL SETUP (required for multiprocessing on Windows) ====================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words and len(word) > 2]
    return tokens


def compute_coherence_values(dictionary, corpus, texts, start=3, limit=12, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                         passes=15, iterations=100, random_state=42, alpha='auto', eta='auto')
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherencemodel.get_coherence()
        coherence_values.append(coherence_score)
        model_list.append(model)
        print(f"  Topics: {num_topics} | Coherence: {coherence_score:.4f}")
    optimal_idx = np.argmax(coherence_values)
    optimal_num = range(start, limit, step)[optimal_idx]
    print(f"Optimal number of topics: {optimal_num} (highest coherence)")
    return model_list[optimal_idx], optimal_num


if __name__ == '__main__':
    # ==================== 1. LOAD DATA ====================
    print("Loading dataset...")
    df = pd.read_csv('1429_1.csv', encoding='utf-8')

    # Keep only the columns we need
    df = df[['reviews.text', 'reviews.date']].dropna(subset=['reviews.text'])

    # Clean date column
    df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')
    df = df.dropna(subset=['reviews.date'])
    df['year'] = df['reviews.date'].dt.year

    print(f"Total reviews loaded: {len(df)}")
    print(f"Years covered: {sorted(df['year'].unique())}")

    # ==================== 2. TEXT PREPROCESSING ====================
    print("Preprocessing text for LDA...")
    df['processed'] = df['reviews.text'].apply(preprocess_text)

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(df['processed'])
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in df['processed']]

    # ==================== 3. FIND OPTIMAL NUMBER OF TOPICS (LDA) ====================
    print("Finding optimal number of topics using coherence score...")
    lda_model, num_topics = compute_coherence_values(dictionary, corpus, df['processed'], start=3, limit=12)

    # ==================== 4. AUTO-GENERATE TOPIC NAMES ====================
    print("Auto-naming topics...")
    topic_names = {}
    for i in range(num_topics):
        top_words = lda_model.show_topic(i, topn=8)
        # Take top 3 meaningful words and make a readable name
        words = [word for word, _ in top_words][:4]
        name = " ".join(words).title().replace("Kindle", "Kindle").replace("Fire", "Fire")
        topic_names[i] = f"Topic {i}: {name[:60]}..." if len(name) > 60 else f"Topic {i}: {name}"

    print("Topic names:")
    for i, name in topic_names.items():
        print(f"   {name}")

    # ==================== 5. ASSIGN TOPIC TO EACH REVIEW ====================
    print("Assigning dominant topic to each review...")
    df['topic'] = [max(lda_model[bow], key=lambda x: x[1])[0] for bow in corpus]

    # ==================== 6. SENTIMENT ANALYSIS WITH BERT (3-class) ====================
    print("Running BERT sentiment analysis (positive/neutral/negative)...")
    # Using a model that natively supports 3 classes
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1  # CPU (change to 0 if you have GPU)
    )

    def get_sentiment(text):
        try:
            result = sentiment_pipeline(text[:512])[0]  # truncate to model limit
            label = result['label'].lower()
            return label if label in ['positive', 'neutral', 'negative'] else 'neutral'
        except:
            return 'neutral'

    df['sentiment'] = df['reviews.text'].apply(get_sentiment)
    print("Sentiment distribution overall:")
    print(df['sentiment'].value_counts())

    # ==================== 7. TOPIC + SENTIMENT SUMMARY ====================
    print("\nTopic-wise sentiment counts:")
    topic_sentiment = df.groupby(['topic', 'sentiment']).size().unstack(fill_value=0)
    print(topic_sentiment)

    # Save to CSV for easy viewing
    topic_sentiment.to_csv('topic_sentiment_summary.csv')
    print("Saved summary to topic_sentiment_summary.csv")

    # ==================== 8. WORD CLOUDS (saved to folder - NO pop-up) ====================
    print("Generating and saving word clouds...")
    os.makedirs('wordclouds', exist_ok=True)

    for topic_id in range(num_topics):
        topic_name_clean = topic_names[topic_id].replace(":", "").replace(" ", "_").replace("/", "_")
        for sent in ['positive', 'neutral', 'negative']:
            subset = df[(df['topic'] == topic_id) & (df['sentiment'] == sent)]['reviews.text']
            if len(subset) == 0:
                continue
            text_combined = " ".join(subset.astype(str))

            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                max_words=150,
                contour_width=3,
                contour_color='steelblue'
            ).generate(text_combined)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"{topic_names[topic_id]} - {sent.capitalize()} ({len(subset)} reviews)")

            filename = f"wordclouds/{topic_name_clean}_{sent}.png"
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"   Saved: {filename}")

    print("All word clouds saved to 'wordclouds/' folder (no windows popped up).")

    # ==================== 9. DYNAMIC ANALYSIS (Yearly Sentiment per Topic) ====================
    print("\nDynamic analysis: Yearly sentiment changes per topic")
    yearly = df.groupby(['year', 'topic', 'sentiment']).size().unstack(fill_value=0).reset_index()

    # Pivot for easy reading
    yearly_pivot = yearly.pivot_table(
        index=['year', 'topic'],
        values=['positive', 'neutral', 'negative'],
        aggfunc='sum',
        fill_value=0
    )

    print(yearly_pivot.head(20))
    yearly_pivot.to_csv('dynamic_topic_sentiment_by_year.csv')
    print("Saved full yearly breakdown to dynamic_topic_sentiment_by_year.csv")

    # Simple trend example: print percentage positive per topic per year
    print("\nExample: Positive sentiment % per topic over years")
    for topic_id in range(num_topics):
        topic_data = df[df['topic'] == topic_id]
        yearly_pos = topic_data.groupby('year')['sentiment'].apply(
            lambda x: (x == 'positive').mean() * 100
        )
        print(f"  {topic_names[topic_id]} → Positive %: {yearly_pos.to_dict()}")

    print("\n=== ANALYSIS COMPLETE ===")
    print("Files created:")
    print("   • topic_sentiment_summary.csv")
    print("   • dynamic_topic_sentiment_by_year.csv")
    print("   • wordclouds/ folder with  topic_sentiment.png images")
    print("Ready for your FYP report/presentation!")