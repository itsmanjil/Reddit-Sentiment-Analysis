
from __future__ import print_function

import json
import os
from functools import lru_cache
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from googleapiclient.errors import HttpError
import nltk

from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from nltk.tokenize import ToktokTokenizer, word_tokenize
from nltk.corpus import wordnet

from .models import *
from .youtube_fetcher import YouTubeFetcher
from .youtube_scraper import YouTubeScraper
from .youtube_preprocessor import YouTubePreprocessor
from .sentiment_engines import get_sentiment_engine

tokenizer = ToktokTokenizer()


@lru_cache(maxsize=1)
def get_stopword_list():
    try:
        return set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        return set()


with open("./files/contractions.json", "r") as f:
    contractions_dict = json.load(f)
contractions = contractions_dict["contractions"]

with open("./files/negations.json", "r") as f:
    neg_dict = json.load(f)
negations = neg_dict["negations"]


def replace_contractions(text):
    for word in text.split():
        w = word.lower()
        if w in contractions:
            text = text.replace(word, contractions[w])
    return text


class AntonymReplacer(object):
    def replace(self, word):
        antonyms = set()
        try:
            synsets = wordnet.synsets(word)
        except LookupError:
            return None

        for syn in synsets:
            if syn.pos() in ["a", "s"]:
                for lemma in syn.lemmas():
                    for antonym in lemma.antonyms():
                        antonyms.add(antonym.name())

        if len(antonyms) == 1:
            return antonyms.pop()
        else:
            if word in negations:
                word = word.replace(word, negations[word])
                return word
        return None

    def negReplacer(self, string):
        i = 0
        finalSent = ""
        try:
            sent = word_tokenize(string)
        except LookupError:
            # Fall back when NLTK punkt data isn't available.
            sent = tokenizer.tokenize(string)
        length_sent = len(sent)

        while i < length_sent:
            word = sent[i]
            if word == "not" and i + 1 < length_sent:
                antonymWord = self.replace(sent[i + 1])
                if antonymWord:
                    finalSent += antonymWord + " "
                    i += 2
                    continue

            finalSent += word + " "
            i += 1

        return finalSent


def replace_negation(text):
    replacer = AntonymReplacer()
    return replacer.negReplacer(text)


def remove_stopwords(text):
    stopword_list = get_stopword_list()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip().lower() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return " ".join(filtered_tokens)



@api_view(["POST"])
@permission_classes([IsAuthenticated])
def analyze_youtube_video(request):
    user = request.user

    # Extract parameters
    video_url = request.data.get("video_url")
    max_comments = request.data.get("max_comments", 200)
    use_api = request.data.get("use_api", True)
    emoji_mode = request.data.get("emoji_mode", "convert")
    filter_spam = request.data.get("filter_spam", True)
    filter_language = request.data.get("filter_language", True)
    sentiment_model = request.data.get("sentiment_model", "vader")

    if not video_url:
        return Response({"msg": "video_url is required"}, status=400)

    try:
        # Step 1: Fetch comments
        print(f"Fetching comments from: {video_url}")
        if use_api:
            try:
                fetcher = YouTubeFetcher()
                video_id = fetcher.extract_video_id(video_url)
                video_metadata = fetcher.fetch_video_metadata(video_id)
                comments_raw = fetcher.fetch_comments(video_url, max_results=max_comments)
            except HttpError as e:
                try:
                    error_details = json.loads(e.content).get('error', {})
                    reason = error_details.get('errors', [{}])[0].get('reason')
                    message = error_details.get('message', str(e))

                    if reason == 'quotaExceeded':
                        return Response({"msg": "YouTube API daily quota exceeded. Please try again tomorrow or use scraper mode (use_api: false)."}, status=429)
                    elif reason == 'developerKeyInvalid':
                        return Response({"msg": "The provided YOUTUBE_API_KEY is invalid. Please check your .env file and ensure it is correct."}, status=401)
                    elif reason == 'commentsDisabled':
                        return Response({"msg": "Comments are disabled for this video."}, status=403)
                    elif e.resp.status == 404:
                        return Response({"msg": "Video not found. Please check the URL."}, status=404)
                    else:
                        return Response({"msg": f"A YouTube API error occurred: {message}"}, status=502)
                except (json.JSONDecodeError, KeyError, IndexError):
                     return Response({"msg": f"An unhandled YouTube API error occurred: {str(e)}"}, status=502)
            except Exception as e:
                return Response(
                    {"msg": f"An unexpected error occurred with the YouTube API client: {str(e)}. Please ensure your YOUTUBE_API_KEY is correctly set in the .env file."},
                    status=502
                )
        else:
            try:
                scraper = YouTubeScraper()
                video_id = scraper.extract_video_id(video_url)
                video_metadata = scraper.fetch_video_metadata(video_id)
                comments_raw = scraper.fetch_comments(video_url, max_results=max_comments)
            except Exception as e:
                return Response(
                    {"msg": f"Scraper error: {str(e)}"},
                    status=502
                )

        if not comments_raw:
            return Response(
                {"msg": "No comments found for this video"},
                status=404
            )

        print(f"Fetched {len(comments_raw)} comments")

        # Step 2: Save or update video metadata
        video, created = YouTubeVideo.objects.get_or_create(
            video_id=video_id,
            defaults={
                'title': video_metadata['title'],
                'description': video_metadata.get('description', ''),
                'channel_name': video_metadata['channel'],
                'channel_id': video_metadata.get('channel_id', ''),
                'published_at': video_metadata['published_at'],
                'view_count': video_metadata.get('view_count', 0),
                'like_count': video_metadata.get('like_count', 0),
                'comment_count': video_metadata.get('comment_count', 0),
                'thumbnail_url': video_metadata.get('thumbnail_url', '')
            }
        )

        if not created:
            # Update metadata if video already exists
            video.view_count = video_metadata.get('view_count', video.view_count)
            video.like_count = video_metadata.get('like_count', video.like_count)
            video.comment_count = video_metadata.get('comment_count', video.comment_count)
            video.save()

        # Step 3: Preprocess comments
        print("Preprocessing comments...")
        preprocessor = YouTubePreprocessor()
        processed_comments, filter_stats = preprocessor.batch_preprocess(
            comments_raw,
            emoji_mode=emoji_mode,
            check_spam=filter_spam,
            check_lang=filter_language
        )

        if not processed_comments:
            return Response(
                {"msg": "All comments were filtered out. Try different filter settings."},
                status=400
            )

        print(f"Processed {len(processed_comments)} comments")

        # Step 4: Apply additional preprocessing (contractions, negations, stopwords)
        for item in processed_comments:
            processed_text = item['processed_text']
            processed_text = replace_contractions(processed_text)
            processed_text = replace_negation(processed_text)
            processed_text = remove_stopwords(processed_text)
            item['processed_text'] = processed_text

        # Step 5: Sentiment Analysis
        print(f"Running sentiment analysis using {sentiment_model}...")
        engine = get_sentiment_engine(sentiment_model)

        for item in processed_comments:
            sentiment, score = engine.analyze(item['processed_text'])
            item['sentiment'] = sentiment
            if isinstance(score, dict):  # VADER returns dict
                item['sentiment_score'] = score.get('compound', 0.0)
            else:  # RoBERTa returns float
                item['sentiment_score'] = score

        # Step 6: Save comments to database
        print("Saving comments to database...")
        for item in processed_comments:
            try:
                YouTubeComment.objects.update_or_create(
                    comment_id=item.get('comment_id', ''),
                    defaults={
                        'video': video,
                        'text': item['text'],
                        'author': item['author'],
                        'likes': item['likes'],
                        'published_at': item['published_at'],
                        'is_reply': item['is_reply'],
                        'sentiment': item['sentiment'],
                        'sentiment_score': item['sentiment_score'],
                        'is_spam': False,
                        'language': item['metadata']['language']
                    }
                )
            except Exception as e:
                print(f"Error saving comment: {e}")
                continue

        # Step 7: Generate analytics
        print("Generating analytics...")
        sentiments = [item['sentiment'] for item in processed_comments]
        sentiment_counts = Counter(sentiments)

        sentiment_data = {
            'Positive': sentiment_counts.get('Positive', 0),
            'Negative': sentiment_counts.get('Negative', 0),
            'Neutral': sentiment_counts.get('Neutral', 0)
        }

        # Like-weighted sentiment
        like_weighted = []
        for item in processed_comments:
            likes = item['likes']
            if likes > 0:
                like_weighted.append({
                    'likes': likes,
                    'sentiment': item['sentiment'],
                    'text': item['text'][:100],
                    'author': item['author']
                })

        like_weighted.sort(key=lambda x: x['likes'], reverse=True)

        # Top words for word clouds
        positive_words = []
        negative_words = []

        for item in processed_comments:
            words = item['processed_text'].split()
            if item['sentiment'] == 'Positive':
                positive_words.extend(words)
            elif item['sentiment'] == 'Negative':
                negative_words.extend(words)

        top_positive = Counter(positive_words).most_common(50)
        top_negative = Counter(negative_words).most_common(50)

        # Step 8: Save analysis
        analysis = YouTubeAnalysis.objects.create(
            user=user,
            video=video,
            sentiment_data=sentiment_data,
            like_weighted_sentiment=like_weighted[:20],
            top_words_positive=[{'word': w, 'count': c} for w, c in top_positive],
            top_words_negative=[{'word': w, 'count': c} for w, c in top_negative],
            total_comments_analyzed=len(processed_comments),
            filtered_spam_count=filter_stats['filtered_spam'],
            filtered_language_count=filter_stats['filtered_language'],
            filtered_short_count=filter_stats['filtered_short'],
            analysis_model=sentiment_model.upper()
        )

        # Step 9: Calculate percentages
        total = len(processed_comments)
        sentiment_ratio = {
            'positive_percent': round(sentiment_data['Positive'] / total * 100, 2) if total > 0 else 0,
            'negative_percent': round(sentiment_data['Negative'] / total * 100, 2) if total > 0 else 0,
            'neutral_percent': round(sentiment_data['Neutral'] / total * 100, 2) if total > 0 else 0
        }

        print("Analysis complete!")

        # Step 10: Return response
        return Response({
            'msg': 'Analysis complete',
            'video': {
                'id': video.video_id,
                'title': video.title,
                'channel': video.channel_name,
                'view_count': video.view_count,
                'like_count': video.like_count,
                'comment_count': video.comment_count,
                'thumbnail_url': video.thumbnail_url
            },
            'sentiment_data': sentiment_data,
            'sentiment_ratio': sentiment_ratio,
            'total_analyzed': len(processed_comments),
            'filtered': {
                'spam': filter_stats['filtered_spam'],
                'language': filter_stats['filtered_language'],
                'short': filter_stats['filtered_short'],
                'total': filter_stats['total']
            },
            'like_weighted_sentiment': like_weighted[:10],
            'top_words_positive': [{'word': w, 'count': c} for w, c in top_positive[:20]],
            'top_words_negative': [{'word': w, 'count': c} for w, c in top_negative[:20]],
            'analysis_id': analysis.id,
            'model_used': sentiment_model.upper()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(
            {"msg": f"Analysis failed: {str(e)}"},
            status=500
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_youtube_analysis(request, video_id):
    try:
        analysis = YouTubeAnalysis.objects.filter(
            video__video_id=video_id
        ).order_by('-fetched_date').first()

        if not analysis:
            return Response(
                {"msg": "No analysis found for this video"},
                status=404
            )

        total = analysis.total_comments_analyzed
        sentiment_ratio = {
            'positive_percent': round(analysis.sentiment_data['Positive'] / total * 100, 2) if total > 0 else 0,
            'negative_percent': round(analysis.sentiment_data['Negative'] / total * 100, 2) if total > 0 else 0,
            'neutral_percent': round(analysis.sentiment_data['Neutral'] / total * 100, 2) if total > 0 else 0
        }

        return Response({
            'data': {
                'video': {
                    'id': analysis.video.video_id,
                    'title': analysis.video.title,
                    'channel': analysis.video.channel_name,
                    'view_count': analysis.video.view_count,
                    'like_count': analysis.video.like_count,
                    'thumbnail_url': analysis.video.thumbnail_url
                },
                'sentiment_data': analysis.sentiment_data,
                'sentiment_ratio': sentiment_ratio,
                'like_weighted_sentiment': analysis.like_weighted_sentiment,
                'top_words_positive': analysis.top_words_positive,
                'top_words_negative': analysis.top_words_negative,
                'total_comments': analysis.total_comments_analyzed,
                'filtered': {
                    'spam': analysis.filtered_spam_count,
                    'language': analysis.filtered_language_count,
                    'short': analysis.filtered_short_count
                },
                'model_used': analysis.analysis_model,
                'fetched_date': analysis.fetched_date
            }
        })

    except Exception as e:
        return Response({"msg": str(e)}, status=500)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_user_youtube_analyses(request):
    try:
        analyses = YouTubeAnalysis.objects.filter(
            user=request.user
        ).select_related('video').order_by('-fetched_date')[:20]

        data = []
        for analysis in analyses:
            total = analysis.total_comments_analyzed
            data.append({
                'id': analysis.id,
                'video': {
                    'id': analysis.video.video_id,
                    'title': analysis.video.title,
                    'channel': analysis.video.channel_name,
                    'channel_id': analysis.video.channel_id,
                    'view_count': analysis.video.view_count,
                    'like_count': analysis.video.like_count,
                    'thumbnail_url': analysis.video.thumbnail_url
                },
                'sentiment_data': analysis.sentiment_data,
                'total_comments_analyzed': total,
                'positive_percent': round(analysis.sentiment_data['Positive'] / total * 100, 2) if total > 0 else 0,
                'negative_percent': round(analysis.sentiment_data['Negative'] / total * 100, 2) if total > 0 else 0,
                'like_weighted_sentiment': analysis.like_weighted_sentiment,
                'top_words_positive': analysis.top_words_positive,
                'top_words_negative': analysis.top_words_negative,
                'filtered': {
                    'spam': analysis.filtered_spam_count,
                    'language': analysis.filtered_language_count,
                    'short': analysis.filtered_short_count,
                    'total': analysis.filtered_spam_count + analysis.filtered_language_count + analysis.filtered_short_count
                },
                'analysis_model': analysis.analysis_model,
                'fetched_date': analysis.fetched_date
            })

        return Response({'data': data})

    except Exception as e:
        return Response({"msg": str(e)}, status=500)


# Health check endpoint
@api_view(["GET"])
@permission_classes((IsAuthenticated,))
def index(request):
    return JsonResponse({"data": "YouTube Sentiment Analysis API - v2.0"})


# Test endpoint
@api_view(["GET"])
def test_endpoint(request):
    return JsonResponse({"status": "Server is working", "message": "Test successful"})
