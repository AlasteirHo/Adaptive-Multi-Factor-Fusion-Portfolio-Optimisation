"""
Reddit r/wallstreetbets Scraper
Extracts titles and text content from posts
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime

class RedditScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://www.reddit.com"
    
    def scrape_json(self, subreddit="wallstreetbets", limit=25, sort="hot"):
        """
        Scrape using Reddit's JSON endpoint (no API key needed)
        
        Args:
            subreddit: subreddit name (default: wallstreetbets)
            limit: number of posts to fetch (default: 25, max: 100)
            sort: sorting method - 'hot', 'new', 'top', 'rising' (default: hot)
        """
        url = f"{self.base_url}/r/{subreddit}/{sort}.json?limit={limit}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            posts = []
            for post in data['data']['children']:
                post_data = post['data']
                
                post_info = {
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'author': post_data.get('author', ''),
                    'score': post_data.get('score', 0),
                    'upvote_ratio': post_data.get('upvote_ratio', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                    'url': f"{self.base_url}{post_data.get('permalink', '')}",
                    'post_url': post_data.get('url', ''),
                    'flair': post_data.get('link_flair_text', ''),
                    'is_self': post_data.get('is_self', False)
                }
                posts.append(post_info)
            
            return posts
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return []
    
    def save_to_json(self, posts, filename="wallstreetbets_posts.json"):
        """Save scraped posts to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2, default=str)
        print(f"Saved {len(posts)} posts to {filename}")
    
    def save_to_csv(self, posts, filename="wallstreetbets_posts.csv"):
        """Save scraped posts to CSV file"""
        import csv
        
        if not posts:
            print("No posts to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=posts[0].keys())
            writer.writeheader()
            writer.writerows(posts)
        print(f"Saved {len(posts)} posts to {filename}")
    
    def print_posts(self, posts):
        """Print posts in a readable format"""
        for i, post in enumerate(posts, 1):
            print(f"\n{'='*80}")
            print(f"Post #{i}")
            print(f"{'='*80}")
            print(f"Title: {post['title']}")
            print(f"Author: u/{post['author']}")
            print(f"Score: {post['score']} | Comments: {post['num_comments']} | Upvote Ratio: {post['upvote_ratio']:.2%}")
            print(f"Flair: {post['flair']}")
            print(f"Created: {post['created_utc']}")
            print(f"URL: {post['url']}")
            if post['text']:
                print(f"\nText Preview: {post['text'][:200]}...")
            else:
                print(f"\nLink Post: {post['post_url']}")


def main():
    """Example usage"""
    scraper = RedditScraper()
    
    print("Scraping r/wallstreetbets...")
    print("="*80)
    
    # Scrape hot posts
    posts = scraper.scrape_json(subreddit="wallstreetbets", limit=25, sort="hot")
    
    if posts:
        # Display posts
        scraper.print_posts(posts)
        
        # Save to files
        scraper.save_to_json(posts)
        scraper.save_to_csv(posts)
        
        # Summary statistics
        print(f"\n{'='*80}")
        print(f"Summary: Scraped {len(posts)} posts")
        print(f"Total upvotes: {sum(p['score'] for p in posts)}")
        print(f"Total comments: {sum(p['num_comments'] for p in posts)}")
        print(f"Average score: {sum(p['score'] for p in posts) / len(posts):.1f}")
    else:
        print("No posts found or error occurred")


if __name__ == "__main__":
    main()