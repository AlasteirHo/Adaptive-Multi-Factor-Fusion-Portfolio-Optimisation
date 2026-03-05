import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
from dotenv import load_dotenv
import random
import time
import os
import sys
import subprocess
import json
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Suppress Windows handle errors on cleanup
if sys.platform == 'win32':
    import atexit
    atexit.register(lambda: None)

class TwitterScraper:
    COOKIE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twitter_cookies.json")

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
        options.add_argument("--disable-gpu") #Disable GPU acceleration, reduce computational usage

        # Additional stability flags to prevent crashes
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-sync")
        options.add_argument("--metrics-recording-only")
        options.add_argument("--disable-default-apps")
        options.add_argument("--mute-audio")
        options.add_argument("--no-first-run")
        options.add_argument("--disable-setuid-sandbox")
        options.add_argument("--disable-web-security")

        # Use a persistent Chrome profile so the browser has history/cookies
        # and doesn't look like a fresh bot instance each run (Bot detection evasion)
        # Store profile outside OneDrive to avoid sync-related file locking
        profile_dir = os.path.join(os.path.expanduser("~"), ".twitter_scraper_profile")
        os.makedirs(profile_dir, exist_ok=True)

        # Remove stale lock files from previous crashed sessions
        for lock_file in ["SingletonLock", "SingletonCookie", "SingletonSocket"]:
            lock_path = os.path.join(profile_dir, lock_file)
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
        default_lock = os.path.join(profile_dir, "Default", "Lock")
        if os.path.exists(default_lock):
            try:
                os.remove(default_lock)
            except OSError:
                pass

        # Auto-detect Chrome version to install right drivers
        chrome_version = None
        try:
            result = subprocess.run(
                ['reg', 'query', r'HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon', '/v', 'version'],
                capture_output=True, text=True
            )
            chrome_version = int(result.stdout.strip().split()[-1].split('.')[0])
        except Exception:
            pass

        print(f"undetected_chromedriver version: {uc.__version__}")
        print(f"Chrome version: {chrome_version}")
        print("Launching Chrome...")

        # Pass user_data_dir to uc.Chrome() directly (not as a Chrome argument)
        # so undetected_chromedriver can manage profile patching correctly
        try:
            self.driver = uc.Chrome(options=options, version_main=chrome_version, user_data_dir=profile_dir)
            print("Browser started successfully!")
        except Exception as e:
            print(f"Failed to start Chrome with profile. Error: {e}")
            print("Retrying without user_data_dir...")
            # Fallback: try without profile if it's corrupted
            self.driver = uc.Chrome(options=options, version_main=chrome_version)
            print("Browser started successfully (without profile)!")

    def save_cookies(self):
        cookies = self.driver.get_cookies()
        with open(self.COOKIE_FILE, 'w') as f:
            json.dump(cookies, f)
        print(f"Saved {len(cookies)} cookies to {self.COOKIE_FILE}")

    def load_cookies(self):
        if not os.path.exists(self.COOKIE_FILE):
            return False
        try:
            with open(self.COOKIE_FILE, 'r') as f:
                cookies = json.load(f)
            # Navigate to X first so the domain matches for cookie injection
            self.driver.get("https://x.com")
            time.sleep(3)
            for cookie in cookies:
                # Removes fields that may cause selenium errors
                cookie.pop('sameSite', None)
                cookie.pop('storeId', None)
                try:
                    self.driver.add_cookie(cookie)
                except Exception:
                    continue
            print(f"Loaded {len(cookies)} cookies from file")
            # Reload the page with cookies applied
            self.driver.get("https://x.com/home")
            time.sleep(5)
            # Check if we're actually logged in by looking for the home timeline
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-testid="AppTabBar_Home_Link"], a[href="/home"]'))
                )
                print("Cookie login successful")
                return True
            except:
                print("Cookies expired or invalid")
                return False
        except Exception as e:
            print(f"Failed to load cookies: {e}")
            return False

    def login(self):
        # With a persistent Chrome profile, we may already be logged in
        self.driver.get("https://x.com/home")
        time.sleep(5)

        # Check if we're already logged in (session persisted from profile)
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-testid="AppTabBar_Home_Link"], a[href="/home"]'))
            )
            print("Already logged in (session from Chrome profile)")
            self.click_search_icon()
            return
        except:
            pass

        # Try cookie-based login as fallback
        if self.load_cookies():
            self.click_search_icon()
            return

        # No session — manual login required (avoids bot detection)
        print("\nMANUAL LOGIN REQUIRED")
        print("A browser window will open to the X login page.")
        print("Please log in manually in the browser.")
        self.driver.get("https://x.com/login")
        time.sleep(3)
        input("Press Enter in this terminal after you have logged in...")
        time.sleep(2)

        # Save cookies as backup
        self.save_cookies()

        # Progress to next step
        self.click_search_icon()
    # Clicks on the search icon
    def click_search_icon(self):
        try:
            time.sleep(2)
            # Try multiple selectors for the search icon
            search_selectors = [
                'a[href="/explore"]',
                'a[data-testid="AppTabBar_Explore_Link"]',
                'a[aria-label="Search and explore"]',
                'svg path[d*="M10.25 3.75"]',  # Path to the search icon on twitter
            ]

            for selector in search_selectors:
                try:
                    search_icon = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    search_icon.click()
                    print("Search icon has been clicked")
                    time.sleep(2)
                    return
                except:
                    continue

            # Fallback method that navigates directly to explore page
            print("Could not find the search icon, resorting to explore page")
            self.driver.get("https://x.com/explore")
            time.sleep(3)

        except Exception as e:
            print(f"Error clicking search icon: {e}")

    def wait_for_tweets_to_load(self, timeout=10):
        # Wait for tweets to appear on the page, return True if loaded, False otherwise.
        for _ in range(timeout):
            try:
                tweets = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
                if tweets:
                    return True
            except:
                pass
            time.sleep(1)
        return False

    def check_for_error_message(self): # Rate limited
        # Error handling for 'Something went wrong' error message
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

    def type_search_query(self, query):
        # Type a search query into Twitter's search box character by character.
        # Navigate to Twitter home if not already there
        if "x.com" not in self.driver.current_url:
            self.driver.get("https://x.com/home")
            time.sleep(3)

        # Find and click the search box
        search_selectors = [
            'input[data-testid="SearchBox_Search_Input"]',
            'input[aria-label="Search query"]',
            'input[placeholder="Search"]'
        ]

        search_input = None
        for selector in search_selectors:
            try:
                search_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                break
            except:
                continue

        if search_input is None:
            # Fallback: click on search icon/link first
            try:
                search_link = self.driver.find_element(By.CSS_SELECTOR, 'a[href="/explore"]')
                search_link.click()
                time.sleep(2)
                search_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'input[data-testid="SearchBox_Search_Input"]'))
                )
            except:
                raise Exception("Could not find search input")

        # Clear any existing text by selecting all and deleting
        search_input.click()
        time.sleep(0.5)
        search_input.send_keys(Keys.CONTROL + "a")  # Select all
        time.sleep(0.2)
        search_input.send_keys(Keys.BACKSPACE)  # Delete selected text
        time.sleep(0.3)

        # Find the first space (after ticker) to add a pause
        first_space_idx = query.find(' ')

        for i, char in enumerate(query):
            search_input.send_keys(char)
            time.sleep(random.uniform(0.03, 0.10))  # Simulated human typing speed for bot evasion

            # Add extra pause after typing the ticker (at first space)
            if i == first_space_idx:
                time.sleep(random.uniform(0.5, 1.0))  # Pause after ticker + space

        time.sleep(1)

        # Click on the ticker suggestion from the dropdown (e.g. "$nvda")
        # Extract the ticker part (e.g. "$NVDA") from the query for matching
        ticker_part = query.split()[0] if query else query
        clicked_suggestion = False
        try:
            # Look for the suggestion div containing the ticker text
            suggestion = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH,
                    f'//span[translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz")="{ticker_part.lower()}"]/ancestor::div[contains(@class,"r-1mmae3n")]'
                ))
            )
            suggestion.click()
            print(f"Clicked ticker suggestion for {ticker_part}")
            clicked_suggestion = True
            time.sleep(random.uniform(3, 5))
        except:
            pass

        if not clicked_suggestion:
            # Fallback: try clicking "Search for" suggestion
            try:
                search_suggestion = WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "Search for")]'))
                )
                search_suggestion.click()
                print("Clicked 'Search for' suggestion")
                clicked_suggestion = True
                time.sleep(random.uniform(3, 5))
            except:
                pass

        if not clicked_suggestion:
            # Final fallback: press Enter
            print("No suggestion found, pressing Enter")
            search_input.send_keys(Keys.ENTER)
            time.sleep(random.uniform(3, 5))

        # Search lands on the "Top" tab by default

    def click_latest_tab(self):
        # Switch from Top tab to Latest tab
        latest_selectors = [
            'a[href*="f=live"]',
            '//span[text()="Latest"]/ancestor::a[@role="tab"]',
            '//div[@role="tablist"]//span[contains(text(), "Latest")]/ancestor::a',
            'a[aria-label*="Latest"]',
        ]
        for selector in latest_selectors:
            try:
                if selector.startswith('//'):
                    tab = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                else:
                    tab = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                tab.click()
                print("Switched to Latest tab")
                time.sleep(random.uniform(2, 4))
                return True
            except Exception:
                continue
        print("Could not click Latest tab")
        return False

    def _try_scrape(self, max_refresh_attempts=3, error_wait_minutes=10):
        # Attempt to scrape tweets from the current tab, with refresh retries
        for attempt in range(max_refresh_attempts):
            tweets_loaded = self.wait_for_tweets_to_load()
            if not tweets_loaded and attempt < max_refresh_attempts - 1:
                print(f"Tweets not loading, refreshing page (attempt {attempt + 1}/{max_refresh_attempts})...")
                self.driver.refresh()
                time.sleep(random.uniform(3, 5))
                continue

            if self.check_for_error_message():
                print(f"'Something went wrong' error found. Waiting {error_wait_minutes} minutes before retrying")
                for remaining in range(error_wait_minutes, 0, -1):
                    print(f"Waiting {remaining} minutes remaining", end='\r')
                    time.sleep(60)
                print()
                print(f"Refreshing page after {error_wait_minutes} minutes")
                self.driver.refresh()
                time.sleep(5)

                if self.check_for_error_message():
                    print(f"Error still occurs after waiting. Continuing")
                    continue

            tweets = self.scrape_tweets()
            if tweets:
                return tweets

            if attempt < max_refresh_attempts - 1:
                print(f"No posts found, refreshing page (attempt {attempt + 2}/{max_refresh_attempts})...")
                self.driver.refresh()
                time.sleep(3)

        return []

    def search_tweets(self, ticker, date, max_refresh_attempts=3, error_wait_minutes=10):
        # Searches for tweets with ticker symbol on a specific date chunk
        # Tries Top tab first, falls back to Latest tab if no tweets found
        since_date = date.strftime("%Y-%m-%d")
        until_date = (date + timedelta(days=1)).strftime("%Y-%m-%d")

        query = f"${ticker} until:{until_date} since:{since_date}"

        # Type the search query (lands on Top tab by default)
        self.type_search_query(query)
        time.sleep(random.uniform(2, 4))

        # Try Top tab first
        print("Trying Top tab...")
        tweets = self._try_scrape(max_refresh_attempts, error_wait_minutes)
        if tweets:
            return tweets

        # No tweets in Top tab, fall back to Latest tab
        print("No tweets in Top tab, switching to Latest...")
        if self.click_latest_tab():
            tweets = self._try_scrape(max_refresh_attempts, error_wait_minutes)

        return tweets
    

# ============ HTML text extraction ==============
    def scrape_tweets(self):
    # Scrape tweets from the current page, scrolling to the very end
        tweets_data = []

        # Force page visibility to prevent Twitter from throttling when unfocused
        self.driver.execute_script("""
            Object.defineProperty(document, 'hidden', { value: false, writable: true });
            Object.defineProperty(document, 'visibilityState', { value: 'visible', writable: true });
        """)

        # Scroll to load ALL tweets until we reach the end
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        last_tweet_count = 0
        no_change_count = 0
        max_no_change = 5  # Increased attempts before stopping

        while True:
            # Keep simulating activity to prevent throttling when window is unfocused
            self.driver.execute_script("""
                window.focus();
                document.dispatchEvent(new Event('visibilitychange'));
                document.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: Math.random() * 100, clientY: Math.random() * 100 }));
            """)

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
                window.scrollBy(0, window.innerHeight);
                window.scrollTo(0, document.body.scrollHeight);
                document.documentElement.scrollTop = document.documentElement.scrollHeight;
            """)
            time.sleep(random.uniform(2, 4))  # Randomized scroll delay (Bot detection Evasion)

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            current_tweet_count = len(tweets_data)

            # Check both height AND tweet count to determine if we've reached the end of the webpage
            if new_height == last_height and current_tweet_count == last_tweet_count:
                no_change_count += 1
                # Check if we've truly reached the end (multiple attempts with no new content)
                if no_change_count >= max_no_change:
                    print(f"Reached end of page after {no_change_count} unchanged scrolls ({current_tweet_count} tweets)")
                    break
                # Wait a bit longer and try again in case content is still loading
                time.sleep(1.5)
            else:
                no_change_count = 0  # Reset counter when new content loads

            last_height = new_height
            last_tweet_count = current_tweet_count

        return tweets_data

    def extract_tweet_data(self, tweet_element):
        # Extract data from a single tweet element (HTML payload)
        try:
            # Tweet body - replace newlines with spaces to keep CSV single-line
            body_element = tweet_element.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
            body = ' '.join(body_element.text.split())

            # Timestamp - skip tweet if no valid date found
            try:
                time_element = tweet_element.find_element(By.TAG_NAME, 'time')
                raw_date = time_element.get_attribute('datetime')
                if not raw_date:
                    return None
                # Reformat to consistent ISO 8601 with timezone offset (e.g. 2024-10-13T17:23:08+00:00)
                post_date = datetime.fromisoformat(raw_date.replace('Z', '+00:00')).strftime('%Y-%m-%dT%H:%M:%S+00:00')
            except:
                return None

            # Engagement metrics
            try:
                replies = self.get_metric(tweet_element, 'reply')
                retweets = self.get_metric(tweet_element, 'retweet')
                likes = self.get_metric(tweet_element, 'like')
            except:
                replies, retweets, likes = 0, 0, 0

            return {
                'body': body,
                'post_date': post_date,
                'replies': replies,
                'retweets': retweets,
                'likes': likes
            }
        except:
            return None

    def get_metric(self, tweet_element, metric_type):
        # Extract engagement metric (replies, retweets, likes)"""
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

    def find_missing_dates(self, filename, start_date, end_date):
        # Find dates within the full start-end range that have no data in the CSV
        scraped_dates = set()

        if os.path.exists(filename):
            try:
                # Only read search_date column to avoid body-comma misparse issues
                df = pd.read_csv(filename, usecols=['search_date'])
                if not df.empty:
                    scraped_dates = set(pd.to_datetime(df['search_date'], errors='coerce').dt.date)
                    scraped_dates.discard(None)
            except Exception as e:
                print(f"Error reading CSV for missing dates: {e}")

        check_start = start_date.date() if isinstance(start_date, datetime) else start_date
        check_end = end_date.date() if isinstance(end_date, datetime) else end_date

        missing = []
        current = check_start
        while current <= check_end:
            if current not in scraped_dates:
                missing.append(datetime(current.year, current.month, current.day))
            current += timedelta(days=1)

        return missing

    def sort_csv(self, filename):
        # Re-sort the CSV by post_date so gap-filled rows are in the right position
        if not os.path.exists(filename):
            return
        try:
            df = pd.read_csv(filename)
            rows_before = len(df)
            df['_parsed'] = pd.to_datetime(df['post_date'], errors='coerce')
            bad_rows = df['_parsed'].isna().sum()
            if bad_rows > 0:
                print(f"WARNING: sort_csv function is dropping {bad_rows} rows with unparseable post_date")
            df = df.dropna(subset=['_parsed'])
            df = df.sort_values('_parsed').reset_index(drop=True)
            df = df.drop(columns=['_parsed'])
            df.to_csv(filename, index=False)
            print(f"sort_csv: {rows_before} rows -> {len(df)} rows")
        except Exception as e:
            print(f"Error sorting CSV: {e}")

    def scrape_date_range(self, ticker, start_date, end_date, output_file):
        # Scrape tweets for all missing dates in the range, then move to the latest date in the CSV
        # This is only meant implemented due to X's API limits which cause some dates to not load properly
        missing_dates = self.find_missing_dates(output_file, start_date, end_date)

        if not missing_dates:
            print(f"No missing dates for ${ticker}, skipping.")
            return []

        print(f"Found {len(missing_dates)} missing date(s) for ${ticker}")

        for i, gap_date in enumerate(missing_dates, 1):
            print(f"[{i}/{len(missing_dates)}] Scraping ${ticker} for {gap_date.strftime('%Y-%m-%d')}...")
            tweets = self.search_tweets(ticker, gap_date)
            for tweet in tweets:
                tweet['ticker'] = ticker
                tweet['search_date'] = gap_date.strftime('%Y-%m-%d')
            print(f"Found {len(tweets)} tweets")
            self.save_to_csv(tweets, output_file)
            time.sleep(random.uniform(5, 10))

        print(f"Completed all missing dates for ${ticker}")
        self.sort_csv(output_file)

        return []

    def save_to_csv(self, tweets, filename):
        # Append new tweets to CSV file (append-only, no re-reading of existing data)
        if not tweets:
            print(f"No tweets to save for {filename}")
            return

        import csv
        columns = ['ticker', 'search_date', 'body', 'post_date', 'replies', 'retweets', 'likes']
        file_exists = os.path.exists(filename)

        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            for tweet in tweets:
                writer.writerow(tweet)

        print(f"Appended {len(tweets)} tweets to {filename}")

    def get_latest_date_from_csv(self, filename):
        # Get the latest search_date from an existing CSV file to resume scraping
        abs_path = os.path.abspath(filename)
        print(f"Checking for an existing CSV at: {abs_path}")

        if not os.path.exists(filename):
            print(f"File does not exist: {abs_path}")
            return None

        try:
            df = pd.read_csv(filename, usecols=['search_date'])
            if df.empty:
                print(f"The CSV is empty")
                return None

            df['date_parsed'] = pd.to_datetime(df['search_date'], errors='coerce')
            latest_date = df['date_parsed'].max()

            if pd.isna(latest_date):
                print(f"Could not parse any dates from search_date column")
                return None

            # Return as datetime with just the date part
            result = datetime(latest_date.year, latest_date.month, latest_date.day)
            print(f"Latest date found in CSV: {result.strftime('%Y-%m-%d')}")
            return result
        except Exception as e:
            print(f"Error reading CSV file {filename}: {e}")
            return None
        
    # Used to check for existing tweets in CSV (prevent duplications)
    def load_existing_tweets(self, filename):
        # Load existing tweets from CSV file
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
                # More aggressive cleanup to prevent handle errors
                self.driver.close()  # Close current window first
                time.sleep(1)
                self.driver.quit()   # Then quit the driver
            except (OSError, Exception) as e:
                print(f"Error during cleanup (ignored): {e}")
                # Force kill any remaining Chrome processes if needed
                try:
                    import psutil
                    for proc in psutil.process_iter(['name']):
                        if 'chrome' in proc.info['name'].lower():
                            proc.kill()
                except:
                    pass

# Return absolute path to the project's tweets folder.
def get_project_tweets_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, "..", "Raw_Data/Tweets"))

# Check if a ticker's CSV already has data up to the end date.
def check_ticker_completion(ticker, start_date, end_date, tweets_dir=None):
    if tweets_dir is None:
        tweets_dir = get_project_tweets_dir()
    csv_path = os.path.join(tweets_dir, f"tweets_{ticker}.csv")

    if not os.path.exists(csv_path):
        return False, None, None

    try:
        df = pd.read_csv(csv_path, usecols=['search_date'])
        if df.empty:
            return False, None, None

        scraped_dates = set(pd.to_datetime(df['search_date'], errors='coerce').dt.date)
        scraped_dates.discard(None)

        if not scraped_dates:
            return False, None, None

        latest = max(scraped_dates)
        latest_dt = datetime(latest.year, latest.month, latest.day)

        # Count missing dates in the full range
        check_start = start_date.date() if isinstance(start_date, datetime) else start_date
        check_end = end_date.date() if isinstance(end_date, datetime) else end_date

        missing_count = 0
        current = check_start
        while current <= check_end:
            if current not in scraped_dates:
                missing_count += 1
            current += timedelta(days=1)

        # Complete only if every date in the range has data
        return missing_count == 0, latest_dt, missing_count
    except Exception as e:
        print(f"Error checking {csv_path}: {e}")
        return False, None, None


def main():
    # Login Details from environment variables (.gitignored)
    USERNAME = os.getenv("TWITTER_USERNAME")
    PASSWORD = os.getenv("TWITTER_PASSWORD")

    # All tickers to scrape (Half of the queries to avoid rate limits)
    TICKERS = [
      # Technology
        "NVDA", "AAPL", "MSFT", "AVGO", "ORCL",
        # Communication Services
        "GOOGL", "META",
        # Consumer Discretionary
        "AMZN", "TSLA", "HD",
        # Financial Services
        "BRK.B", "JPM", "V", "MA",
        # Healthcare
        "LLY", "JNJ", "UNH",
        # Consumer Staples
        "WMT", "PG",
        # Energy
        "XOM",
    ]

    START_DATE = datetime(2023, 9, 1)  # Starting date
    END_DATE = datetime(2023,10 ,1)    # End date

    # Ensure tweets directory exists
    tweets_dir = get_project_tweets_dir()
    os.makedirs(tweets_dir, exist_ok=True)

    # Pre-check: filter out already completed tickers
    print("Checking tweets folder for completed tickers")
    tickers_to_scrape = []
    for ticker in TICKERS:
        is_complete, latest_date, missing_count = check_ticker_completion(ticker, START_DATE, END_DATE, tweets_dir)
        if is_complete:
            print(f"{ticker}: COMPLETED (all dates from {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')} covered)")
        else:
            if latest_date and missing_count:
                print(f" {ticker}: {missing_count} missing date(s) between {START_DATE.strftime('%Y-%m-%d')} and {END_DATE.strftime('%Y-%m-%d')}")
            else:
                print(f"{ticker}: No existing data, starting from {START_DATE.strftime('%Y-%m-%d')}")
            tickers_to_scrape.append(ticker)

    if not tickers_to_scrape:
        print("\nAll tickers are completed")
        return

    print(f"\nTickers left to scrape: {tickers_to_scrape}")

    scraper = TwitterScraper(USERNAME, PASSWORD)

    try:
        print("Starting a Google Chrome instance")
        scraper.start_driver()
        print("Logging in")
        scraper.login()

        print(f"Starting to scrape tweets for {len(tickers_to_scrape)} tickers")

        for ticker in tickers_to_scrape:
            output_file = os.path.join(tweets_dir, f"tweets_{ticker}.csv")
            print(f"\n{'='*50}")
            print(f"Scraping ${ticker}")
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
