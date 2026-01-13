
import re
from datetime import datetime


class YouTubeScraper:

    def __init__(self):
        try:
            from youtube_comment_downloader import YoutubeCommentDownloader
            self.downloader = YoutubeCommentDownloader()
        except ImportError:
            raise ImportError(
                "youtube-comment-downloader not installed. "
                "Run: pip install youtube-comment-downloader"
            )

    def extract_video_id(self, url):
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([\w-]{11})',
            r'(?:youtu\.be\/)([\w-]{11})',
            r'(?:youtube\.com\/embed\/)([\w-]{11})',
            r'(?:youtube\.com\/v\/)([\w-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        if len(url) == 11 and re.match(r'^[\w-]{11}$', url):
            return url

        return None

    def fetch_comments(self, video_url, max_results=100, sort_by='top'):
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")

        comments = []
        sort_mode = 0 if sort_by == 'top' else 1

        try:
            # Note: youtube-comment-downloader returns a generator
            for comment in self.downloader.get_comments_from_url(
                f'https://www.youtube.com/watch?v={video_id}',
                sort=sort_mode
            ):
                if len(comments) >= max_results:
                    break

                # Parse the time string if available
                published_at = comment.get('time', '')
                if published_at:
                    published_at = datetime.now().isoformat()
                else:
                    published_at = datetime.now().isoformat()

                comment_data = {
                    'text': comment.get('text', ''),
                    'author': comment.get('author', 'Unknown'),
                    'likes': comment.get('votes', 0),
                    'published_at': published_at,
                    'reply_count': 0,  # Not available in scraper
                    'is_reply': comment.get('photo', '').startswith('https://yt3') == False,
                    'video_id': video_id,
                    'comment_id': comment.get('cid', '')
                }
                comments.append(comment_data)

            if not comments:
                raise RuntimeError("No comments found. Video may have comments disabled.")

            return comments

        except Exception as e:
            if "No comments found" in str(e) or len(comments) == 0:
                raise RuntimeError(
                    "No comments found. The video may have comments disabled, "
                    "be age-restricted, or the scraper may have been blocked."
                )
            raise RuntimeError(f"Scraper error: {str(e)}")

    def fetch_video_metadata(self, video_id):
        return {
            'title': f'Video {video_id}',
            'description': '',
            'channel': 'Unknown',
            'channel_id': '',
            'published_at': datetime.now().isoformat(),
            'view_count': 0,
            'like_count': 0,
            'comment_count': 0,
            'thumbnail_url': f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg'
        }
