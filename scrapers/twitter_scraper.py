import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
from dotenv import load_dotenv
import random
import time
import csv
import os
import sys
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Suppress Windows handle errors on cleanup
if sys.platform == 'win32':
    import atexit
    atexit.register(lambda: None)

class TwitterScraper:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.driver = None

    def start_driver(self):
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--start-maximized")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        # Auto-detect Chrome version and download matching driver
        self.driver = uc.Chrome(
            options=options,
            driver_executable_path=None,  # Let it download fresh
            browser_executable_path=None  # Auto-detect Chrome location
        )
        print("Browser started successfully!")

    def login(self):
        self.driver.get("https://x.com/login")
        time.sleep(5)  # Wait longer for initial load

        # Try automated login, fall back to manual if it fails
        try:
            # Enter username - try multiple selectors
            username_input = None
            selectors = [
                'input[autocomplete="username"]',
                'input[name="text"]',
                'input[type="text"]'
            ]

            for selector in selectors:
                try:
                    username_input = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    print(f"Found username input with selector: {selector}")
                    break
                except:
                    continue

            if username_input is None:
                raise Exception("Could not find username input")

            username_input.send_keys(self.username)
            username_input.send_keys(Keys.ENTER)
            time.sleep(3)

            # Check if Twitter asks for phone/email verification
            try:
                verification_input = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]'))
                )
                verification_value = input("Twitter is asking for verification (phone/email). Enter the value: ")
                verification_input.send_keys(verification_value)
                verification_input.send_keys(Keys.ENTER)
                time.sleep(2)
            except:
                pass

            # Enter password
            password_input = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
            )
            password_input.send_keys(self.password)
            password_input.send_keys(Keys.ENTER)
            time.sleep(5)

            print("Login successful!")

        except Exception as e:
            print(f"Automated login failed: {e}")
            print("\n=== MANUAL LOGIN REQUIRED ===")
            print("Please log in manually in the browser window.")
            input("Press Enter after you have logged in successfully...")

    def check_for_error_message(self):
        """Check if Twitter is showing 'Something went wrong' error message"""
        try:
            page_source = self.driver.page_source.lower()
            error_indicators = [
                "something went wrong",
                "try reloading",
                "rate limit",
                "try again"
            ]
            for indicator in error_indicators:
                if indicator in page_source:
                    return True
            return False
        except:
            return False

    def search_tweets(self, ticker, date, max_refresh_attempts=3, error_wait_minutes=10):
        """Search for tweets with ticker symbol on a specific date"""
        # Format: $AMD until:2025-09-10 since:2025-09-09
        since_date = (date - timedelta(days=1)).strftime("%Y-%m-%d")
        until_date = date.strftime("%Y-%m-%d")

        query = f"${ticker} until:{until_date} since:{since_date}"
        search_url = f"https://x.com/search?q={query}&src=typed_query&f=top"

        self.driver.get(search_url)
        time.sleep(random.uniform(3, 6))  # Randomized delay to appear more human

        tweets_with_dollar = []
        
        # Try to scrape tweets, refresh page if none found
        for attempt in range(max_refresh_attempts):
            # Check for error message before scraping
            if self.check_for_error_message():
                print(f"    Detected 'Something went wrong' error. Waiting {error_wait_minutes} minutes before retrying...")
                # Wait for the specified time (default 10 minutes)
                for remaining in range(error_wait_minutes, 0, -1):
                    print(f"    Waiting... {remaining} minutes remaining", end='\r')
                    time.sleep(60)
                print()  # New line after countdown
                print(f"    Refreshing page after {error_wait_minutes} minute wait...")
                self.driver.refresh()
                time.sleep(5)

                # Check again after refresh
                if self.check_for_error_message():
                    print(f"    Error persists after waiting. Continuing to next attempt...")
                    continue

            tweets_with_dollar = self.scrape_tweets()

            if tweets_with_dollar:
                break

            # No tweets found, refresh and retry
            if attempt < max_refresh_attempts - 1:
                print(f"    No posts found, refreshing page (attempt {attempt + 2}/{max_refresh_attempts})...")
                self.driver.refresh()
                time.sleep(3)

        # If we found less than 10 tweets with $ sign, also try without it to get more
        if len(tweets_with_dollar) < 10:
            if len(tweets_with_dollar) == 0:
                print(f"    No posts found with ${ticker}, trying without $ sign: {ticker}...")
            else:
                print(f"    Found only {len(tweets_with_dollar)} tweets with ${ticker}")
                print(f"    Searching without $ sign to collect more tweets: {ticker}...")
            
            query_no_dollar = f"{ticker} until:{until_date} since:{since_date}"
            search_url_no_dollar = f"https://x.com/search?q={query_no_dollar}&src=typed_query&f=top"
            
            self.driver.get(search_url_no_dollar)
            time.sleep(random.uniform(3, 6))
            
            tweets_without_dollar = []
            
            # Try to scrape tweets without $ sign
            for attempt in range(max_refresh_attempts):
                if self.check_for_error_message():
                    print(f"    Detected error. Waiting {error_wait_minutes} minutes before retrying...")
                    for remaining in range(error_wait_minutes, 0, -1):
                        print(f"    Waiting... {remaining} minutes remaining", end='\r')
                        time.sleep(60)
                    print()
                    self.driver.refresh()
                    time.sleep(5)
                    if self.check_for_error_message():
                        continue
                
                tweets_without_dollar = self.scrape_tweets()
                
                if tweets_without_dollar:
                    break
                
                if attempt < max_refresh_attempts - 1:
                    print(f"    No posts found, refreshing page (attempt {attempt + 2}/{max_refresh_attempts})...")
                    self.driver.refresh()
                    time.sleep(3)
            
            # Combine tweets from both searches, avoiding duplicates
            if tweets_without_dollar:
                # Create a set of existing tweet identifiers (username + body)
                existing_tweets = {(t['username'], t['body']) for t in tweets_with_dollar}
                
                # Add new tweets that aren't duplicates
                added_count = 0
                for tweet in tweets_without_dollar:
                    tweet_id = (tweet['username'], tweet['body'])
                    if tweet_id not in existing_tweets:
                        tweets_with_dollar.append(tweet)
                        existing_tweets.add(tweet_id)
                        added_count += 1
                
                print(f"    Added {added_count} unique tweets from search without $ sign")
                print(f"    Total tweets collected: {len(tweets_with_dollar)}")
        
        return tweets_with_dollar

    def scrape_tweets(self):
        """Scrape tweets from the current page, scrolling to the very end"""
        tweets_data = []

        # Keep browser window focused to prevent Twitter from throttling content loading
        self.driver.execute_script("window.focus();")

        # Scroll to load ALL tweets until we reach the end
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        last_tweet_count = 0
        no_change_count = 0
        max_no_change = 5  # Increased attempts before stopping

        while True:
            # Ensure window stays active by simulating activity
            self.driver.execute_script("window.focus(); document.dispatchEvent(new Event('visibilitychange'));")

            # Find all tweet articles
            tweets = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

            for tweet in tweets:
                try:
                    tweet_data = self.extract_tweet_data(tweet)
                    if tweet_data and tweet_data not in tweets_data:
                        tweets_data.append(tweet_data)
                except Exception as e:
                    continue

            # Scroll down using multiple methods to ensure it works even when unfocused
            self.driver.execute_script("""
                window.scrollTo(0, document.body.scrollHeight);
                document.documentElement.scrollTop = document.documentElement.scrollHeight;
            """)
            time.sleep(random.uniform(2, 4))  # Randomized scroll delay

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            current_tweet_count = len(tweets_data)

            # Check both height AND tweet count to determine if we've reached the end
            if new_height == last_height and current_tweet_count == last_tweet_count:
                no_change_count += 1
                # Check if we've truly reached the end (multiple attempts with no new content)
                if no_change_count >= max_no_change:
                    print(f"    Reached end of page after {no_change_count} unchanged scrolls ({current_tweet_count} tweets)")
                    break
                # Wait a bit longer and try again in case content is still loading
                time.sleep(1.5)
            else:
                no_change_count = 0  # Reset counter when new content loads

            last_height = new_height
            last_tweet_count = current_tweet_count

        return tweets_data

    def extract_tweet_data(self, tweet_element):
        """Extract data from a single tweet element"""
        try:
            # Tweet body
            body_element = tweet_element.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
            body = body_element.text

            # Username
            try:
                user_element = tweet_element.find_element(By.CSS_SELECTOR, 'div[data-testid="User-Name"]')
                username = user_element.text.split('\n')[0]
            except:
                username = ""

            # Timestamp
            try:
                time_element = tweet_element.find_element(By.TAG_NAME, 'time')
                post_date = time_element.get_attribute('datetime')
            except:
                post_date = ""

            # Engagement metrics
            try:
                replies = self.get_metric(tweet_element, 'reply')
                retweets = self.get_metric(tweet_element, 'retweet')
                likes = self.get_metric(tweet_element, 'like')
            except:
                replies, retweets, likes = 0, 0, 0

            return {
                'username': username,
                'body': body,
                'post_date': post_date,
                'replies': replies,
                'retweets': retweets,
                'likes': likes
            }
        except:
            return None

    def get_metric(self, tweet_element, metric_type):
        """Extract engagement metric (replies, retweets, likes)"""
        try:
            element = tweet_element.find_element(By.CSS_SELECTOR, f'button[data-testid="{metric_type}"]')
            text = element.text.strip()
            if text == '':
                return 0
            # Handle K, M suffixes
            if 'K' in text:
                return int(float(text.replace('K', '')) * 1000)
            elif 'M' in text:
                return int(float(text.replace('M', '')) * 1000000)
            return int(text)
        except:
            return 0

    def scrape_date_range(self, ticker, start_date, end_date, output_file):
        """Scrape tweets for a date range, resuming from latest date in CSV if exists"""
        # Check for existing CSV and resume from next day after latest date
        latest_date = self.get_latest_date_from_csv(output_file)

        if latest_date:
            # Start from the day after the latest date in CSV
            resume_date = latest_date + timedelta(days=1)
            if resume_date > end_date:
                print(f"CSV file {output_file} already contains data up to {latest_date.strftime('%Y-%m-%d')}")
                print(f"No new dates to scrape (end_date is {end_date.strftime('%Y-%m-%d')})")
                return []

            print(f"Found existing CSV with data up to {latest_date.strftime('%Y-%m-%d')}")
            print(f"Resuming from {resume_date.strftime('%Y-%m-%d')}...")
            current_date = resume_date

            # Load existing tweets to preserve them
            all_tweets = self.load_existing_tweets(output_file)
            print(f"Loaded {len(all_tweets)} existing tweets from CSV")
        else:
            print(f"No existing CSV found, starting fresh from {start_date.strftime('%Y-%m-%d')}")
            current_date = start_date
            all_tweets = []

        while current_date <= end_date:
            print(f"Scraping ${ticker} for {current_date.strftime('%Y-%m-%d')}...")

            tweets = self.search_tweets(ticker, current_date)

            for tweet in tweets:
                tweet['ticker'] = ticker
                tweet['search_date'] = current_date.strftime('%Y-%m-%d')
                all_tweets.append(tweet)

            print(f"  Found {len(tweets)} tweets")

            # Save progress incrementally
            self.save_to_csv(all_tweets, output_file)

            current_date += timedelta(days=1)
            time.sleep(random.uniform(5, 10))  # Increased randomized delay between searches

        return all_tweets

    def save_to_csv(self, tweets, filename):
        """Save tweets to CSV file"""
        if not tweets:
            return

        fieldnames = ['ticker', 'search_date', 'username', 'body', 'post_date', 'replies', 'retweets', 'likes']

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tweets)

        print(f"Saved {len(tweets)} tweets to {filename}")

    def get_latest_date_from_csv(self, filename):
        """Get the latest search_date from an existing CSV file to resume scraping"""
        if not os.path.exists(filename):
            return None

        try:
            df = pd.read_csv(filename)
            if df.empty or 'search_date' not in df.columns:
                return None

            # Get the maximum search_date
            latest_date_str = df['search_date'].max()
            latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
            return latest_date
        except Exception as e:
            print(f"Error reading CSV file {filename}: {e}")
            return None

    def load_existing_tweets(self, filename):
        """Load existing tweets from CSV file"""
        if not os.path.exists(filename):
            return []

        try:
            df = pd.read_csv(filename)
            if df.empty:
                return []
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading existing tweets from {filename}: {e}")
            return []

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except OSError:
                pass  # Ignore Windows handle errors during cleanup


def main():
    # Login Details from environment variables
    USERNAME = os.getenv("TWITTER_USERNAME")
    PASSWORD = os.getenv("TWITTER_PASSWORD")  

    # All tickers to scrape
    TICKERS = [
        # Tech / Semiconductors
        "AMD",
        # Tech / Software
        "AAPL", "MSFT", "ORCL",
        # Tech / Consumer
        "AMZN", "TSLA",
        # Energy
        "XOM",
        # Consumer Discretionary
        "HD"
    ]

    START_DATE = datetime(2023, 10, 10)  # Starting date
    END_DATE = datetime(2025, 10, 10)    # End date

    scraper = TwitterScraper(USERNAME, PASSWORD)

    try:
        print("Starting browser...")
        scraper.start_driver()
        print("Logging in...")
        scraper.login()

        print(f"Starting to scrape tweets for {len(TICKERS)} tickers...")

        for ticker in TICKERS:
            output_file = f"tweets_{ticker}.csv"
            print(f"\n{'='*50}")
            print(f"Scraping ${ticker}...")
            print(f"{'='*50}")
            scraper.scrape_date_range(ticker, START_DATE, END_DATE, output_file)
            print(f"Completed ${ticker}, saved to {output_file}")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
