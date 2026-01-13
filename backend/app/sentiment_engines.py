

class VADERSentimentEngine:

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError(
                "vaderSentiment not installed. "
                "Run: pip install vaderSentiment"
            )

    def analyze(self, text):
        scores = self.analyzer.polarity_scores(text)

        # Convert compound score to 3-class sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment, scores

    def batch_analyze(self, texts):
        return [self.analyze(text) for text in texts]


class RoBERTaSentimentEngine:

    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment-latest',
                 device=-1):
        try:
            from transformers import pipeline
            self.model = pipeline(
                'sentiment-analysis',
                model=model_name,
                device=device
            )
            self.max_length = 512  # RoBERTa token limit
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Run: pip install transformers torch"
            )

    def analyze(self, text):
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]

        result = self.model(text)[0]
        label = result['label']
        score = result['score']

        label_map = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive',
            'negative': 'Negative',
            'neutral': 'Neutral',
            'positive': 'Positive',
            'NEGATIVE': 'Negative',
            'NEUTRAL': 'Neutral',
            'POSITIVE': 'Positive'
        }

        sentiment = label_map.get(label, 'Neutral')

        return sentiment, score

    def batch_analyze(self, texts, batch_size=32):
        # Truncate all texts
        truncated_texts = [
            text[:self.max_length] if len(text) > self.max_length else text
            for text in texts
        ]

        results = []
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            batch_results = self.model(batch)

            for result in batch_results:
                label = result['label']
                score = result['score']

                label_map = {
                    'LABEL_0': 'Negative',
                    'LABEL_1': 'Neutral',
                    'LABEL_2': 'Positive',
                    'negative': 'Negative',
                    'neutral': 'Neutral',
                    'positive': 'Positive',
                    'NEGATIVE': 'Negative',
                    'NEUTRAL': 'Neutral',
                    'POSITIVE': 'Positive'
                }

                sentiment = label_map.get(label, 'Neutral')
                results.append((sentiment, score))

        return results


class TFIDFSentimentEngine:

    def __init__(self, model_path='./model/model.sav',
                 vectorizer_path='./model/tfidfVectorizer.pickle'):
        import pickle

        try:
            self.model = pickle.load(open(model_path, 'rb'))
            self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model files not found. Expected:\n"
                f"  - {model_path}\n"
                f"  - {vectorizer_path}"
            )

    def analyze(self, text):
        import pandas as pd

        df = pd.DataFrame([{'tweet': text}])
        tweet_vec = self.vectorizer.transform(df['tweet'])
        prediction = self.model.predict(tweet_vec)[0]

        return prediction, 0.0

    def batch_analyze(self, texts):
        import pandas as pd

        df = pd.DataFrame([{'tweet': text} for text in texts])
        tweet_vec = self.vectorizer.transform(df['tweet'])
        predictions = self.model.predict(tweet_vec)

        return [(pred, 0.0) for pred in predictions]


def get_sentiment_engine(engine_type='vader', **kwargs):
    engines = {
        'vader': VADERSentimentEngine,
        'roberta': RoBERTaSentimentEngine,
        'tfidf': TFIDFSentimentEngine
    }

    if engine_type not in engines:
        raise ValueError(
            f"Invalid engine type: {engine_type}. "
            f"Choose from: {list(engines.keys())}"
        )

    return engines[engine_type](**kwargs)
