from flask import Flask, render_template, request, send_file
import pickle
import pandas as pd
import re
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/analyze', methods=['POST'])
def analyze():
    reviews_text = request.form.get("reviews_text")
    file = request.files.get("file")

    if reviews_text and reviews_text.strip() != "":
        reviews_list = reviews_text.split("\n")
        df = pd.DataFrame({"Review": reviews_list})
    elif file:
        df = pd.read_csv(file)
        if 'Review' not in df.columns: return "CSV must contain a column named 'Review'"
    else:
        return "Please paste reviews or upload a CSV file."

    df['clean'] = df['Review'].apply(clean_text)
    vectors = vectorizer.transform(df['clean'])
    df['prediction'] = model.predict(vectors)

    counts = df['prediction'].value_counts()
    positive = counts.get('positive', 0)
    neutral = counts.get('neutral', 0)
    negative = counts.get('negative', 0)
    total = len(df)

    pos_percent = round((positive / total) * 100, 2)
    neu_percent = round((neutral / total) * 100, 2)
    neg_percent = round((negative / total) * 100, 2)

    df.to_csv("analysis_report.csv", index=False)

    # --- THEMED MATPLOTLIB SETTINGS ---
    plt.rcParams.update({'text.color': "white", 'axes.labelcolor': "white"})

    # --- PIE CHART ---
    plt.figure(figsize=(5, 5), facecolor='none')  # Use facecolor='none' here for transparency
    plt.pie(
        [positive, neutral, negative],
        labels=["Pos", "Neu", "Neg"],
        autopct='%1.1f%%',
        colors=['#00ff88', '#00d4ff', '#ff0055']
    )
    # The 'transparent' keyword goes HERE
    plt.savefig("static/pie.png", transparent=True)
    plt.close()

    # --- BAR CHART ---
    plt.figure(figsize=(5, 5), facecolor='none')
    plt.bar(
        ["Positive", "Neutral", "Negative"],
        [positive, neutral, negative],
        color=['#00ff88', '#00d4ff', '#ff0055']
    )
    plt.tick_params(colors='white')
    # And HERE
    plt.savefig("static/bar.png", transparent=True)
    plt.close()

    # BAR CHART
    plt.figure(figsize=(5, 5))
    plt.bar(["Positive", "Neutral", "Negative"], [positive, neutral, negative],
            color=['#00ff88', '#00d4ff', '#ff0055'])
    plt.tick_params(colors='white')
    plt.savefig("static/bar.png", transparent=True)
    plt.close()

    # WORDCLOUDS (Matching the Aqua/Dark theme)
    pos_text = " ".join(df[df['prediction'] == 'positive']['clean'])
    neg_text = " ".join(df[df['prediction'] == 'negative']['clean'])

    if pos_text.strip():
        WordCloud(width=400, height=250, background_color=None, mode="RGBA",
                  colormap="summer").generate(pos_text).to_file("static/positive_wc.png")
    if neg_text.strip():
        WordCloud(width=400, height=250, background_color=None, mode="RGBA",
                  colormap="autumn").generate(neg_text).to_file("static/negative_wc.png")

    return render_template("analysis.html", positive=pos_percent, neutral=neu_percent,
                           negative=neg_percent, overall=df['prediction'].mode()[0], message="Analysis Complete")


@app.route('/download')
def download():
    return send_file("analysis_report.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)