# YouTube Sentiment Analysis

A Django-based sentiment analysis platform that analyzes YouTube video comments to surface audience sentiment and engagement insights.

Note: This repository is named `Reddit-Sentiment-Analysis` for legacy reasons; the current project scope is YouTube comment analysis.

## Features
- Dual data collection: YouTube Data API v3 or scraper (no API key needed)
- VADER sentiment engine tuned for social media text
- Spam detection, language filtering, and emoji handling
- Like-weighted analysis to highlight influential comments
- Word frequency outputs for positive and negative comments
- Top comments by engagement
- Analytics API with sentiment ratios and filtering stats

## Quick Start

### Prerequisites
- Python 3.8+
- Django 4.0+
- PostgreSQL or SQLite

### Installation
```bash
git clone https://github.com/<your-username>/Reddit-Sentiment-Analysis.git
cd Reddit-Sentiment-Analysis

# Backend dependencies
pipenv install
pipenv shell

cd backend
cp .env.example .env
# Edit .env and set SECRET_KEY and YOUTUBE_API_KEY (optional)

python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

### Frontend
See `frontend/README.md` for local dev and build steps.

### Get a YouTube API Key (Optional)
The scraper works without an API key, but the official API is faster and more reliable:

1. Go to https://console.cloud.google.com/
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create credentials -> API Key
5. Add it to `.env` as `YOUTUBE_API_KEY`

Free tier: 10,000 units/day (about 100 video analyses)

## Usage

### Analyze a YouTube Video

With API (recommended):
```bash
POST /api/youtube/analyze/
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "max_comments": 200,
  "use_api": true,
  "sentiment_model": "vader"
}
```

Without API (scraper mode):
```bash
POST /api/youtube/analyze/
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "max_comments": 100,
  "use_api": false,
  "sentiment_model": "vader"
}
```

### Example Response
```json
{
  "msg": "Analysis complete",
  "video": {
    "id": "dQw4w9WgXcQ",
    "title": "Rick Astley - Never Gonna Give You Up",
    "channel": "Rick Astley",
    "view_count": 1234567890,
    "like_count": 12000000
  },
  "sentiment_data": {
    "Positive": 150,
    "Negative": 30,
    "Neutral": 20
  },
  "sentiment_ratio": {
    "positive_percent": 75.0,
    "negative_percent": 15.0,
    "neutral_percent": 10.0
  },
  "like_weighted_sentiment": [],
  "top_words_positive": [
    {"word": "love", "count": 45},
    {"word": "amazing", "count": 32}
  ]
}
```

## Testing
```bash
cd backend
python test_youtube.py
```

Expected output:
```
Video ID Extraction
Comment Preprocessing
VADER Sentiment
YouTube API (or Scraper)
All tests passed
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/youtube/analyze/` | Analyze YouTube video comments |
| GET | `/api/youtube/analysis/<video_id>/` | Get saved analysis for a video |
| GET | `/api/youtube/analyses/` | Get all user's analyses |

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_url` | string | required | YouTube video URL or ID |
| `max_comments` | integer | 200 | Number of comments to analyze (1-1000) |
| `use_api` | boolean | true | Use official API (true) or scraper (false) |
| `emoji_mode` | string | "convert" | How to handle emojis: "remove", "convert", "keep" |
| `filter_spam` | boolean | true | Filter spam comments |
| `filter_language` | boolean | true | Filter non-English comments |
| `sentiment_model` | string | "vader" | Model to use: "vader", "tfidf", "roberta" |

## Architecture

```
YouTube Video URL
  -> Fetcher/Scraper
     - API: fast, 10k quota/day
     - Scraper: unlimited, slower
  -> Preprocessor
     - Emoji handling
     - Spam detection
     - Language filtering
     - Text cleaning
  -> Sentiment Engine
     - VADER (recommended)
     - RoBERTa (higher accuracy)
     - TF-IDF (legacy)
  -> Database
     - YouTubeVideo
     - YouTubeComment
     - YouTubeAnalysis
  -> Analytics API
     - Sentiment ratios
     - Like-weighted data
     - Word frequencies
     - Top comments
```

## Technology Stack

### Backend
- Django 4.0.2
- Django REST Framework
- PostgreSQL or SQLite
- NLTK
- scikit-learn

### YouTube Integration
- google-api-python-client (official YouTube API)
- youtube-comment-downloader (scraper)
- VADER Sentiment
- emoji
- langdetect

## Sentiment Models

| Model | Speed | Accuracy | GPU | Best For |
|-------|-------|----------|-----|----------|
| VADER | Fast | Medium-High | No | Social media comments |
| TF-IDF | Fast | Medium | No | Baseline/legacy |
| RoBERTa | Slow (CPU) | High | Optional | Accuracy-focused |

RoBERTa requires `transformers` and `torch` in the backend environment. With pipenv, these are included from `Pipfile`.

Recommendation: Use VADER for YouTube sentiment analysis.

## Use Cases

### Product Launch Analysis
Track sentiment on product announcement videos:
- Tech product launches (Apple, Samsung, Tesla)
- Game trailers and announcements
- Movie/TV show previews

### Brand Monitoring
Monitor brand reputation across YouTube:
- Real-time sentiment tracking
- Influencer impact analysis
- Competitor comparison

### Content Creator Insights
Help creators understand their audience:
- Video performance analysis
- Comment sentiment trends
- Identify controversial topics
- Most appreciated content

### Political and News Analysis
Analyze public opinion on current events:
- Debate videos
- News coverage sentiment
- Policy announcement reactions

### Educational Content
Help educators improve courses:
- Identify confusing topics
- Track student satisfaction
- Common questions from comments

## Security
- API keys stored in `.env` (never commit real keys)
- Django security middleware enabled
- SQL injection protection via ORM
- Input validation on all endpoints
- CORS configuration for frontend

## Performance

### YouTube API Mode
- Speed: 2-5 seconds for 100 comments
- Quota: about 100 videos/day (free tier)
- Reliability: 99.9%
- Metadata: full (title, views, likes)

### Scraper Mode
- Speed: 5-15 seconds for 100 comments
- Quota: unlimited
- Reliability: 85% (can break)
- Metadata: minimal

### VADER Sentiment
- Speed: about 0.5 seconds for 100 comments
- Accuracy: 85-90% on social media text
- Memory: <100MB
- GPU: not required

## Contributing
Contributions welcome. Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Credits
- VADER Sentiment: https://github.com/cjhutto/vaderSentiment
- YouTube Data API: https://developers.google.com/youtube/v3
- emoji: https://github.com/carpedm20/emoji
- langdetect: https://github.com/Mimino666/langdetect

## Support
- Run `python test_youtube.py` for diagnostics
- Open an issue for bugs or feature requests

## Roadmap
- [x] YouTube sentiment analysis
- [x] VADER sentiment engine
- [x] Spam detection and language filtering
- [x] Like-weighted sentiment
- [ ] Word cloud visualization endpoint
- [ ] Video transcript analysis
- [ ] Multi-language support
- [ ] Real-time monitoring dashboard
- [ ] Export to CSV/Excel
- [ ] Channel-level analytics
- [ ] Sentiment timeline visualization

Built with Django and VADER Sentiment.

Pure YouTube sentiment analysis; no Reddit or Twitter ingestion.
