# Interactive sentiment analysis demo

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline
import torch
import time
import warnings
warnings.filterwarnings('ignore')

# Load pipeline on GPU (suppressing verbose output)
print("=" * 60)
print("🎭 Interactive Sentiment Analysis")
print("=" * 60)

device_id = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device_id
)

print(f"\n✅ Model loaded on: {'GPU' if device_id == 0 else 'CPU'}\n")


def analyze_sentiment(test_texts):
    """Analyze sentiment of a list of texts."""
    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: \"{text}\"")
        
        start = time.time()
        result = classifier(text)[0]
        inference_time = time.time() - start
        
        # Visual output
        sentiment = result['label']
        confidence = result['score'] * 100
        emoji = "😊" if sentiment == "POSITIVE" else "😞"
        
        print(f"  {emoji} {sentiment} ({confidence:.1f}% confident)")
        print(f"  ⚡ {inference_time*1000:.0f}ms\n")
    
    print("=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)

