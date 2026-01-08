from __future__ import print_function

import ast
import datetime
from datetime import timedelta
import json
import os
import re
import time
import pickle
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import nltk
import praw
import requests

from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from nltk.tokenize import ToktokTokenizer, word_tokenize
from nltk.corpus import wordnet

from .models import *

tokenizer = ToktokTokenizer()


# -----------------------------
# Safe, lazy-loaded NLTK resources
# -----------------------------
@lru_cache(maxsize=1)
def get_stopword_list():
    try:
        return set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        # Do not crash server if stopwords not downloaded
        return set()


# -----------------------------
# Load contractions + negations files
# -----------------------------
with open("./files/contractions.json", "r") as f:
    contractions_dict = json.load(f)
contractions = contractions_dict["contractions"]

with open("./files/negations.json", "r") as f:
    neg_dict = json.load(f)
negations = neg_dict["negations"]


# -----------------------------
# API Health/Index
# -----------------------------
@api_view(["GET"])
@permission_classes((IsAuthenticated,))
def index(request):
    return JsonResponse({"data": "Hello from app"})


# -----------------------------
# Text cleaning helpers
# -----------------------------
def lower_case(text):
    return text.lower()

def remove_square_brackets(text):
    return re.sub(r"\[[^]]*\]", "", text)

def remove_username(text):
    return re.sub(r"@[^\s]+", "", text)

def remove_urls(text):
    return re.sub(r"((http\S+)|(www\.))", "", text)

def remove_special_characters(text):
    pattern = r"[^a-zA-Z\s]"
    return re.sub(pattern, "", text)

def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def remove_multiple(text):
    return re.sub(r"(.)\1{2,}", r"\1", text)

def replace_contractions(text):
    for word in text.split():
        w = word.lower()
        if w in contractions:
            text = text.replace(word, contractions[w])
    return text


class AntonymReplacer(object):
    def replace(self, word):
        antonyms = set()
        for syn in wordnet.synsets(word):
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
        sent = word_tokenize(string)
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


# -----------------------------
# Reddit fetch (real-time)
# -----------------------------
def fetch_posts(product, company, keywords):
    """
    Fetch up to 100 posts/comments from Reddit using PRAW.
    Free tier with generous limits.
    """
    try:
        # Get credentials from environment
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "SentimentAnalysis/1.0")

        missing = [k for k, v in {
            "REDDIT_CLIENT_ID": client_id,
            "REDDIT_CLIENT_SECRET": client_secret,
        }.items() if not v]

        if missing:
            raise RuntimeError(f"Missing Reddit credentials in backend/.env: {missing}")

        # Initialize Reddit instance
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

        # Build search query
        query = f"{product} {company} {keywords}"
        print(f"Reddit search query: {query}")

        # Fetch posts from Reddit (search across all subreddits)
        all_posts = []

        # Search for submissions
        for submission in reddit.subreddit("all").search(query, limit=50, time_filter="week"):
            created_at = datetime.datetime.fromtimestamp(submission.created_utc)
            # Combine title and selftext for better sentiment analysis
            text = f"{submission.title} {submission.selftext}"
            all_posts.append([
                text,
                created_at.year,
                created_at.month,
                created_at.day,
                created_at.hour,
                created_at.minute,
                created_at.second,
            ])

            # Also get top comments from each submission
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:5]:  # Top 5 comments per post
                if hasattr(comment, 'body') and len(comment.body) > 10:
                    comment_time = datetime.datetime.fromtimestamp(comment.created_utc)
                    all_posts.append([
                        comment.body,
                        comment_time.year,
                        comment_time.month,
                        comment_time.day,
                        comment_time.hour,
                        comment_time.minute,
                        comment_time.second,
                    ])

                    if len(all_posts) >= 100:
                        break

            if len(all_posts) >= 100:
                break

        if not all_posts:
            return []

        pd.set_option("display.max_colwidth", None)
        post_text = pd.DataFrame(
            data=all_posts,
            columns=["tweet", "year", "month", "day", "hour", "minute", "second"],
        )

        # Pre-processing pipeline
        post_text["tweet"] = post_text["tweet"].apply(lower_case)
        post_text["tweet"] = post_text["tweet"].apply(remove_multiple)
        post_text["tweet"] = post_text["tweet"].apply(remove_single_char)
        post_text["tweet"] = post_text["tweet"].apply(remove_special_characters)
        post_text["tweet"] = post_text["tweet"].apply(remove_square_brackets)
        post_text["tweet"] = post_text["tweet"].apply(remove_urls)
        post_text["tweet"] = post_text["tweet"].apply(remove_username)
        post_text["tweet"] = post_text["tweet"].apply(replace_contractions)
        post_text["tweet"] = post_text["tweet"].apply(replace_negation)
        post_text["tweet"] = post_text["tweet"].apply(remove_stopwords)

        return post_text

    except praw.exceptions.APIException as e:
        print(f"Reddit API error: {e}")
        raise RuntimeError(f"Reddit API error: {str(e)}")

    except Exception as e:
        print(f"Error fetching Reddit posts: {e}")
        raise


# -----------------------------
# Sentiment model
# -----------------------------
def find_sentiment(cleaned_tweets):
    loaded_model = pickle.load(open("./model/model.sav", "rb"))
    vectorizer = pickle.load(open("./model/tfidfVectorizer.pickle", "rb"))

    tweet_vec = vectorizer.transform(cleaned_tweets["tweet"])
    prediction = loaded_model.predict(tweet_vec)
    return prediction


def get_counts(values):
    content, count = np.unique(values, return_counts=True)
    return dict(zip(content, count))


def hour_counts(hourCount):
    receivedKeys = list(hourCount.keys())
    missingKeys = list(set(range(24)) - set(receivedKeys))

    for k in missingKeys:
        hourCount[k] = 0

    return hourCount


# -----------------------------
# Search endpoint
# -----------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def search_keywords(request):
    print("starting search...")
    user = request.user

    data = {
        "product_name": request.data["product_name"],
        "company_name": request.data["company_name"],
        "keywords": request.data["keywords"],
    }
    print("data.....", data)

    product_name = data["product_name"]

    # Fetch real-time posts from Reddit
    try:
        cleaned_tweets = fetch_posts(data["product_name"], data["company_name"], data["keywords"])
    except Exception as e:
        # Reddit/auth/config errors should be surfaced clearly
        return Response(
            {"msg": f"Reddit fetch failed: {str(e)}"},
            status=502
        )

    if len(cleaned_tweets) == 0:
        return Response(
            {"msg": "No posts found for these keywords. Try different keywords."},
            status=404
        )

    graphDataAvailable = False

    # Predict sentiments
    prediction = find_sentiment(cleaned_tweets)

    # Build dataframe
    lists = list(
        zip(
            prediction,
            cleaned_tweets["year"],
            cleaned_tweets["month"],
            cleaned_tweets["day"],
            cleaned_tweets["hour"],
            cleaned_tweets["minute"],
            cleaned_tweets["second"],
        )
    )
    finalDf = pd.DataFrame(lists, columns=["sentiment", "year", "month", "day", "hour", "minute", "second"])

    enddate = datetime.datetime(
        int(finalDf["year"].iloc[0]),
        int(finalDf["month"].iloc[0]),
        int(finalDf["day"].iloc[0]),
        int(finalDf["hour"].iloc[0]),
        int(finalDf["minute"].iloc[0]),
        int(finalDf["second"].iloc[0]),
    )
    startdate = datetime.datetime(
        int(finalDf["year"].iloc[-1]),
        int(finalDf["month"].iloc[-1]),
        int(finalDf["day"].iloc[-1]),
        int(finalDf["hour"].iloc[-1]),
        int(finalDf["minute"].iloc[-1]),
        int(finalDf["second"].iloc[-1]),
    )

    positiveDf = finalDf[finalDf["sentiment"] == "Positive"]
    negativeDf = finalDf[finalDf["sentiment"] == "Negative"]
    neutralDf = finalDf[finalDf["sentiment"] == "Neutral"]

    # Yesterday then fallback to today
    dayYesterdayDate = datetime.date.today() - datetime.timedelta(days=1)
    date = dayYesterdayDate
    dayYesterday = dayYesterdayDate.day

    positiveDfmonth = positiveDf[positiveDf["month"] == datetime.date.today().month]
    negativeDfmonth = negativeDf[negativeDf["month"] == datetime.date.today().month]
    neutralDfmonth = neutralDf[neutralDf["month"] == datetime.date.today().month]

    positiveDfday = positiveDfmonth[positiveDfmonth["day"] == dayYesterday]
    negativeDfday = negativeDfmonth[negativeDfmonth["day"] == dayYesterday]
    neutralDfday = neutralDfmonth[neutralDfmonth["day"] == dayYesterday]

    # FIXED BUG: len(...) == [] -> should be == 0
    if len(positiveDfday) == 0 or len(negativeDfday) == 0 or len(neutralDfday) == 0:
        dayTodayDate = datetime.date.today()
        date = dayTodayDate
        dayToday = dayTodayDate.day

        positiveDfday = positiveDfmonth[positiveDfmonth["day"] == dayToday]
        negativeDfday = negativeDfmonth[negativeDfmonth["day"] == dayToday]
        neutralDfday = neutralDfmonth[neutralDfmonth["day"] == dayToday]

    if len(positiveDfday) == 0 and len(negativeDfday) == 0 and len(neutralDfday) == 0:
        graphDataAvailable = False
    else:
        graphDataAvailable = True

    sentimentData = get_counts(prediction)

    hourCountPositive = hour_counts(get_counts(positiveDfday["hour"]))
    hourCountNegative = hour_counts(get_counts(negativeDfday["hour"]))
    hourCountNeutral = hour_counts(get_counts(neutralDfday["hour"]))

    hourCountPosUpdated = {}
    hourCountNegUpdated = {}
    hourCountNeutralUpdated = {}

    for i in range(0, 23, 2):
        hourCountPosUpdated[i] = hourCountPositive[i] + hourCountPositive[i + 1]
        hourCountNegUpdated[i] = hourCountNegative[i] + hourCountNegative[i + 1]
        hourCountNeutralUpdated[i] = hourCountNeutral[i] + hourCountNeutral[i + 1]

    hour_key = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    hourData = []

    for key in hour_key:
        if key <= 10:
            keyHour = f"{key}:00 A.M."
        elif key == 12:
            keyHour = f"{key}:00 P.M."
        else:
            keyHour = f"{key - 12}:00 P.M."

        hourData.append({
            "time": keyHour,
            "positive": hourCountPosUpdated[key],
            "negative": hourCountNegUpdated[key],
            "neutral": hourCountNeutralUpdated[key],
        })

    # Ensure all sentiments exist
    requiredKeys = ["Positive", "Negative", "Neutral"]
    for k in requiredKeys:
        if k not in sentimentData:
            sentimentData[k] = 0

    sentimentData = dict(sorted(sentimentData.items()))

    tweetData = TweetAnalysis(
        user=request.user,
        sentiment_data=sentimentData,
        hour_data=hourData,
        product_name=data["product_name"],
        fetched_date=date,
        graph_data_available=graphDataAvailable,
        start_date=startdate,
        end_date=enddate,
    )
    tweetData.save()

    return Response({
        "msg": "From search",
        "is_registered": user.is_registered,
        "data": data,
        "predicted_data": prediction,
        "sentiment_data": sentimentData,
        "hour_data": hourData,
        "product_name": product_name,
        "graphDataAvailable": graphDataAvailable,
    })


# -----------------------------
# Get last sentiment data
# -----------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getSentimentData(request):
    user = request.user

    # FIXED: avoid IndexError by using .first()
    tweetData = TweetAnalysis.objects.filter(user=user).order_by("-id").first()
    if tweetData is None:
        return Response({"message": "Has not searched yet"}, status=404)

    # SAFER: only literal_eval if stored as string
    json_data = tweetData.sentiment_data
    if isinstance(json_data, str):
        json_data = ast.literal_eval(json_data)

    outputSentiment = [{"sentiment": k, "value": json_data[k]} for k in json_data]

    hour_data_json = tweetData.hour_data
    if isinstance(hour_data_json, str):
        hour_data_json = ast.literal_eval(hour_data_json)

    data = {
        "user": tweetData.user.id,
        "sentiment_data": json_data,
        "output_sentiment": outputSentiment,
        "hour_data": hour_data_json,
        "product_name": tweetData.product_name,
        "fetched_date": tweetData.fetched_date,
        "graph_data_available": tweetData.graph_data_available,
        "start_date": tweetData.start_date,
        "end_date": tweetData.end_date,
    }

    return Response({"data": data, "msg": "Sentiment analysis Data"})


@api_view(["POST", "GET"])
def model_operation(request):
    return Response({"data": "done"})
