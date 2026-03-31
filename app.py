from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 加载轻量前沿AI模型（可随时换模型调参优化）
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route('/')
def index():
    return render_template("index.html")

# 摘要接口（支持调参：长度、严谨度）
@app.route('/get_summary', methods=['POST'])
def get_summary():
    data = request.get_json()
    text = data.get("text", "")
    min_len = int(data.get("min_len", 30))
    max_len = int(data.get("max_len", 80))
    
    res = summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)
    return jsonify({"summary": res[0]['summary_text']})

# 情感分析接口
@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    data = request.get_json()
    text = data.get("text", "")
    res = sentiment(text)[0]
    return jsonify({"label": res['label'], "score": round(res['score'],4)})

if __name__ == '__main__':
    app.run(debug=True)
