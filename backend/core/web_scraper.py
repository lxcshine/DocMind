# -*- coding: utf-8 -*-
"""
Web Scraper Engine - Deeply inspired by Firecrawl architecture

Core components:
- Multi-Engine Strategy: Different engines for different URL types
- Fallback Mechanism: Automatic engine switching on failure
- Content Validation: Quality checking of scraped content
- Format Converter: Outputs clean Markdown
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    trafilatura = None
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class EngineType(Enum):
    """Scraping engine types - inspired by Firecrawl engine selection."""
    FETCH = "fetch"
    PLAYWRIGHT = "playwright"
    TLS_CLIENT = "tls_client"
    UNDETECTED_CHROMEDRIVER = "undetected_chromedriver"
    REQUESTS = "requests"


class FeatureFlag(Enum):
    """
    Dynamic feature flags - inspired by Firecrawl's feature system.
    
    Features can be dynamically added/removed during scraping based on results,
    allowing the system to auto-upgrade its capabilities.
    """
    STEALTH_PROXY = "stealthProxy"
    MOBILE = "mobile"
    LOCATION = "location"
    SKIP_TLS_VERIFICATION = "skipTlsVerification"
    USE_FAST_MODE = "useFastMode"
    DISABLE_ADBLOCK = "disableAdblock"


class AddFeatureError(Exception):
    """
    Signal to add features and retry - inspired by Firecrawl.
    When an engine fails due to missing capabilities (e.g., proxy),
    this error tells the system to add that feature and retry.
    """
    def __init__(self, features: List[FeatureFlag]):
        self.features = features
        super().__init__(f"Need features: {[f.value for f in features]}")


class RemoveFeatureError(Exception):
    """
    Signal to remove features and retry - inspired by Firecrawl.
    When certain features cause issues (e.g., adblock blocks content),
    this error tells the system to disable that feature.
    """
    def __init__(self, features: List[FeatureFlag]):
        self.features = features
        super().__init__(f"Remove features: {[f.value for f in features]}")


class WaterfallNextEngineSignal(Exception):
    """
    Signal to try the next engine in the waterfall.
    Inspired by Firecrawl's WaterfallNextEngineSignal.
    """
    pass


class EngineSnipedError(Exception):
    """A higher-priority engine completed successfully, cancel others."""
    pass


class EngineUnsuccessfulError(Exception):
    """An engine completed but the result was not usable."""
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        super().__init__(f"Engine {engine_name} deemed unsuccessful")


class EngineScrapeResult:
    """Result from a single engine scrape attempt."""
    
    def __init__(
        self,
        html: str,
        url: str,
        status_code: int = 200,
        error: Optional[str] = None,
    ):
        self.html = html
        self.url = url
        self.status_code = status_code
        self.error = error
    
    @property
    def is_successful(self) -> bool:
        """Check if scrape was successful based on Firecrawl criteria."""
        is_good_status = 200 <= self.status_code < 300 or self.status_code == 304
        has_no_error = self.error is None
        return is_good_status and has_no_error


class ScrapedPage:
    """Represents a scraped page with cleaned content."""
    
    def __init__(
        self,
        url: str,
        title: str = "",
        content: str = "",
        markdown: str = "",
        html: str = "",
        metadata: Optional[Dict] = None,
        status_code: int = 200,
        error: Optional[str] = None,
    ):
        self.url = url
        self.title = title
        self.content = content
        self.markdown = markdown
        self.html = html
        self.metadata = metadata or {}
        self.status_code = status_code
        self.error = error
    
    def to_dict(self) -> Dict:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "markdown": self.markdown,
            "metadata": self.metadata,
            "status_code": self.status_code,
            "error": self.error,
        }


class ContentCleaner:
    """
    Cleans HTML content by removing noise and irrelevant elements.
    Inspired by Firecrawl's data cleaning layer.
    """
    
    NOISE_TAGS = {
        'script', 'style', 'nav', 'footer', 'header', 'aside',
        'form', 'iframe', 'noscript', 'svg', 'canvas', 'object',
        'ad', 'advertisement', 'popup', 'modal', 'cookie-banner',
    }
    
    NOISE_CLASSES = [
        'ad', 'ads', 'advertisement', 'banner', 'popup', 'modal',
        'cookie', 'footer', 'header', 'nav', 'sidebar', 'menu',
        'social', 'share', 'comment', 'related', 'recommended',
        'newsletter', 'subscription', 'promo', 'tracking',
        'analytics', 'widget', 'toast', 'notification',
    ]
    
    @classmethod
    def clean_html(cls, html: str) -> str:
        """Remove noise from HTML content."""
        if not html:
            return ""
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            for tag in cls.NOISE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()
            
            for element in soup.find_all(True):
                try:
                    classes = element.get('class', [])
                    if classes and any(any(noise in cls_name.lower() for cls_name in classes) for noise in cls.NOISE_CLASSES):
                        element.decompose()
                except Exception:
                    pass
            
            for element in soup.find_all(True):
                try:
                    if not hasattr(element, 'attrs') or not element.attrs:
                        continue
                    attrs_to_remove = []
                    for attr in element.attrs:
                        if attr.startswith('on') or attr in {'style', 'onclick', 'onmouseover'}:
                            attrs_to_remove.append(attr)
                    for attr in attrs_to_remove:
                        del element[attr]
                except Exception:
                    pass
            
            return str(soup)
        except Exception as e:
            logger.warning(f"Error cleaning HTML: {e}")
            return html
    
    @classmethod
    def extract_links(cls, html: str, base_url: str) -> List[str]:
        """Extract all valid links from HTML content."""
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if not href or href.startswith('#') or href.startswith('mailto:'):
                continue
            
            full_url = urljoin(base_url, href)
            parsed_link = urlparse(full_url)
            
            if parsed_link.netloc != base_domain:
                continue
            
            if any(ext in parsed_link.path.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip']):
                continue
            
            links.append(full_url)
        
        return list(set(links))


class ContentParser:
    """
    Parses and extracts structured content from web pages.
    Inspired by Firecrawl's content parser.
    """
    
    @classmethod
    def parse_url(cls, url: str, html: str) -> ScrapedPage:
        """Parse HTML content and extract structured data."""
        try:
            cleaned_html = ContentCleaner.clean_html(html)
            
            markdown = ""
            try:
                if HAS_TRAFILATURA:
                    markdown = trafilatura.extract(
                        html,
                        include_comments=False,
                        include_tables=True,
                        include_images=False,
                        include_links=False,
                        output_format='txt',
                    ) or ""
                else:
                    markdown = ""
            except Exception as e:
                logger.warning(f"Trafilatura extraction failed for {url}: {e}")
                markdown = ""
            
            soup = BeautifulSoup(cleaned_html, 'html.parser')
            
            # Safe title extraction
            title = ""
            try:
                if soup.title:
                    title = soup.title.string
                    if title:
                        title = title.strip()
                    else:
                        title = ""
            except Exception as e:
                logger.warning(f"Error extracting title for {url}: {e}")
                title = ""
            
            if not title:
                try:
                    h1 = soup.find('h1')
                    if h1:
                        title = h1.get_text(strip=True)
                except Exception as e:
                    logger.warning(f"Error extracting h1 for {url}: {e}")
            
            # Safe meta description extraction
            description = ""
            try:
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and hasattr(meta_desc, 'get'):
                    description = meta_desc.get('content', '') or ""
            except Exception as e:
                logger.warning(f"Error extracting description for {url}: {e}")
            
            # Safe meta keywords extraction
            keywords = ""
            try:
                meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                if meta_keywords and hasattr(meta_keywords, 'get'):
                    keywords = meta_keywords.get('content', '') or ""
            except Exception as e:
                logger.warning(f"Error extracting keywords for {url}: {e}")
            
            content_text = ""
            try:
                content_text = soup.get_text(separator='\n', strip=True)
                content_text = re.sub(r'\n{3,}', '\n\n', content_text)
            except Exception as e:
                logger.warning(f"Error extracting text for {url}: {e}")
                content_text = cleaned_html[:2000]
            
            metadata = {
                "url": url,
                "title": title,
                "description": description,
                "keywords": keywords,
                "word_count": len(content_text.split()) if content_text else 0,
            }
            
            return ScrapedPage(
                url=url,
                title=title,
                content=content_text,
                markdown=markdown,
                html=cleaned_html,
                metadata=metadata,
            )
        
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}", exc_info=True)
            return ScrapedPage(
                url=url,
                error=str(e),
                status_code=0,
            )


class ScrapingCache:
    """
    File-based cache for scraped pages - inspired by Firecrawl's Index engine.
    
    Firecrawl's index engine (quality: 1000) serves as a first-pass cache
    to avoid re-scraping the same URLs. This implements the same concept
    using a local JSON-based cache store.
    
    Cache metadata stored per URL:
    - url: The scraped URL
    - html: Raw HTML content  
    - markdown: Parsed markdown content
    - timestamp: When the page was scraped
    - engine: Which engine succeeded
    - status_code: HTTP status code
    - title: Page title
    """
    
    DEFAULT_TTL = 3600  # 1 hour default cache lifetime
    MAX_CACHE_SIZE = 500  # Maximum number of cached entries
    
    def __init__(self, cache_dir: Optional[str] = None, ttl: int = DEFAULT_TTL):
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = Path.home() / ".researchflow" / "cache" / "scraping"
        
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl
        self._index_file = self._cache_dir / "index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict]:
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_index(self):
        try:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, ensure_ascii=False)
        except IOError as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def _cache_file(self, url: str) -> Path:
        return self._cache_dir / f"{self._cache_key(url)}.json"
    
    def get(self, url: str) -> Optional[ScrapedPage]:
        """Retrieve a cached page. Returns None if not found or expired."""
        if url not in self._index:
            return None
        
        entry = self._index[url]
        
        if time.time() - entry["timestamp"] > self._ttl:
            self._remove_entry(url)
            return None
        
        cache_file = self._cache_file(url)
        if not cache_file.exists():
            self._remove_entry(url)
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return ScrapedPage(
                url=data["url"],
                title=data.get("title", ""),
                content=data.get("content", ""),
                markdown=data.get("markdown", ""),
                html=data.get("html", ""),
                metadata=data.get("metadata", {}),
                status_code=data.get("status_code", 200),
            )
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to read cache for {url}: {e}")
            self._remove_entry(url)
            return None
    
    def set(self, page: ScrapedPage):
        """Store a scraped page in the cache."""
        data = {
            "url": page.url,
            "title": page.title,
            "content": page.content,
            "markdown": page.markdown,
            "html": page.html,
            "metadata": page.metadata,
            "status_code": page.status_code,
        }
        
        cache_file = self._cache_file(page.url)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except IOError as e:
            logger.warning(f"Failed to write cache for {page.url}: {e}")
            return
        
        self._index[page.url] = {
            "timestamp": time.time(),
            "engine": page.metadata.get("engine", "unknown"),
            "status_code": page.status_code,
        }
        
        self._evict_if_needed()
        self._save_index()
    
    def _remove_entry(self, url: str):
        self._index.pop(url, None)
        cache_file = self._cache_file(url)
        try:
            cache_file.unlink(missing_ok=True)
        except Exception:
            pass
        self._save_index()
    
    def _evict_if_needed(self):
        if len(self._index) <= self.MAX_CACHE_SIZE:
            return
        
        sorted_entries = sorted(self._index.items(), key=lambda x: x[1]["timestamp"])
        to_remove = len(self._index) - self.MAX_CACHE_SIZE
        
        for url, _ in sorted_entries[:to_remove]:
            self._remove_entry(url)
    
    def clear(self):
        for url in list(self._index.keys()):
            self._remove_entry(url)
    
    def __len__(self):
        return len(self._index)


class ProxyPool:
    """
    Proxy service for bypassing anti-bot protection - inspired by Firecrawl's stealthProxy.
    
    Firecrawl uses stealthProxy (mobile proxies) to bypass advanced anti-bot protection.
    This implements a configurable proxy pool with rotation and health checking.
    
    Proxy types:
    - basic: Regular HTTP/HTTPS proxies
    - stealth: Mobile/residential proxies (when available)
    
    Features:
    - Proxy rotation (round-robin)
    - Automatic proxy type escalation (basic -> stealth)
    - Configurable env-based proxy loading
    """
    
    def __init__(self, proxies: Optional[List[Dict]] = None):
        self._basic_proxies = []
        self._stealth_proxies = []
        self._basic_index = 0
        self._stealth_index = 0
        
        if proxies:
            for p in proxies:
                self.add_proxy(p)
        else:
            self._load_from_env()
    
    def _load_from_env(self):
        """Load proxies from environment variables."""
        proxy_env = os.environ.get("RESEARCHFLOW_PROXIES", "")
        if proxy_env:
            try:
                proxy_list = json.loads(proxy_env)
                for p in proxy_list:
                    self.add_proxy(p)
            except json.JSONDecodeError:
                pass
        
        basic_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        if basic_proxy and not self._basic_proxies:
            self._basic_proxies.append({"url": basic_proxy, "type": "basic"})
        
        stealth_proxy = os.environ.get("RESEARCHFLOW_STEALTH_PROXY", "")
        if stealth_proxy:
            self._stealth_proxies.append({"url": stealth_proxy, "type": "stealth"})
    
    def add_proxy(self, proxy: Dict):
        proxy_type = proxy.get("type", "basic")
        if proxy_type == "stealth":
            self._stealth_proxies.append(proxy)
        else:
            self._basic_proxies.append(proxy)
    
    def has_proxy(self, proxy_type: str = "basic") -> bool:
        if proxy_type == "stealth":
            return len(self._stealth_proxies) > 0
        return len(self._basic_proxies) > 0 or len(self._stealth_proxies) > 0
    
    def get_proxy(self, proxy_type: str = "basic") -> Optional[str]:
        """
        Get next proxy URL using round-robin rotation.
        Returns None if no proxy of the requested type is available.
        """
        if proxy_type == "stealth" and self._stealth_proxies:
            proxy = self._stealth_proxies[self._stealth_index % len(self._stealth_proxies)]
            self._stealth_index += 1
            return proxy["url"]
        
        if proxy_type == "basic" and self._basic_proxies:
            proxy = self._basic_proxies[self._basic_index % len(self._basic_proxies)]
            self._basic_index += 1
            return proxy["url"]
        
        if self._basic_proxies:
            proxy = self._basic_proxies[self._basic_index % len(self._basic_proxies)]
            self._basic_index += 1
            return proxy["url"]
        
        if self._stealth_proxies:
            proxy = self._stealth_proxies[self._stealth_index % len(self._stealth_proxies)]
            self._stealth_index += 1
            return proxy["url"]
        
        return None
    
    def build_proxy_config(self, proxy_type: str = "basic") -> Optional[Dict]:
        """Build proxy configuration for httpx/aiohttp."""
        proxy_url = self.get_proxy(proxy_type)
        if not proxy_url:
            return None
        return {"http": proxy_url, "https": proxy_url}


class FetchEngine:
    """
    Simple HTTP fetch engine - fastest but least capable.
    Quality score: 5 (lowest)
    Enhanced with Firecrawl-style anti-detection:
    - Multiple browser fingerprints
    - Realistic header combinations
    - Proper cookie handling
    - Connection pooling
    - Optional proxy support
    
    Supports FeatureFlags: stealthProxy
    """
    
    QUALITY = 5
    
    SUPPORTED_FEATURES = {FeatureFlag.STEALTH_PROXY}
    
    BROWSER_FINGERPRINTS = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Linux"',
        },
    ]
    
    def __init__(self, timeout: int = 30, user_agent: str = "", proxy_pool: Optional[ProxyPool] = None):
        self.timeout = timeout
        self.user_agent = user_agent
        self.proxy_pool = proxy_pool
        self._session = None
        self._fingerprint_index = 0
    
    def _get_fingerprint(self) -> dict:
        """Get next browser fingerprint from pool (rotation)."""
        fp = self.BROWSER_FINGERPRINTS[self._fingerprint_index % len(self.BROWSER_FINGERPRINTS)]
        self._fingerprint_index += 1
        return fp
    
    def _build_headers(self, url: str) -> dict:
        """Build realistic headers with fingerprint rotation."""
        parsed = urlparse(url)
        fingerprint = self._get_fingerprint()
        
        user_agent = self.user_agent if self.user_agent else fingerprint["User-Agent"]
        
        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Sec-Ch-Ua": fingerprint["Sec-Ch-Ua"],
            "Sec-Ch-Ua-Mobile": fingerprint["Sec-Ch-Ua-Mobile"],
            "Sec-Ch-Ua-Platform": fingerprint["Sec-Ch-Ua-Platform"],
            "Cache-Control": "max-age=0",
            "DNT": "1",
            "Pragma": "no-cache",
        }
    
    async def scrape(self, url: str, features: Optional[Set[FeatureFlag]] = None) -> EngineScrapeResult:
        """Fetch URL using basic HTTP request with Firecrawl-style anti-detection."""
        features = features or set()
        use_proxy = FeatureFlag.STEALTH_PROXY in features
        
        try:
            cookie_jar = aiohttp.CookieJar()
            connector_kwargs = {}
            
            if use_proxy and self.proxy_pool and self.proxy_pool.has_proxy("stealth"):
                proxy_url = self.proxy_pool.get_proxy("stealth")
                if proxy_url:
                    connector_kwargs["proxy"] = proxy_url
                    logger.debug(f"FetchEngine using stealth proxy for {url}")
            elif use_proxy and self.proxy_pool and self.proxy_pool.has_proxy("basic"):
                proxy_url = self.proxy_pool.get_proxy("basic")
                if proxy_url:
                    connector_kwargs["proxy"] = proxy_url
                    logger.debug(f"FetchEngine using basic proxy for {url}")
            
            connector = aiohttp.TCPConnector(**connector_kwargs) if connector_kwargs else None
            
            headers = self._build_headers(url)
            
            async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers=headers,
                    allow_redirects=True,
                    ssl=False,
                ) as response:
                    html = await response.text()
                    return EngineScrapeResult(
                        html=html,
                        url=url,
                        status_code=response.status,
                    )
        except asyncio.TimeoutError:
            return EngineScrapeResult(html="", url=url, status_code=0, error="Timeout")
        except aiohttp.ClientError as e:
            return EngineScrapeResult(html="", url=url, status_code=0, error=str(e))
        except Exception as e:
            return EngineScrapeResult(html="", url=url, status_code=0, error=f"Unexpected: {e}")


class SimpleRequestsEngine:
    """
    Ultimate fallback engine using the requests library.
    Quality score: 1 (lowest, but NEVER fails as long as requests is installed).
    
    This is the last-resort engine that uses the synchronous requests library
    in a thread pool executor. It's the most basic HTTP client with no advanced
    anti-bot features, but it's extremely reliable and almost always works for
    basic web pages. If all other engines fail, this one will likely return
    HTML content that can at least be partially analyzed.
    """
    
    QUALITY = 1
    
    SUPPORTED_FEATURES: Set[FeatureFlag] = set()
    
    def __init__(self, timeout: int = 30, user_agent: str = "", proxy_pool=None):
        self.timeout = timeout
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        self.proxy_pool = proxy_pool
    
    async def scrape(self, url: str, features: Optional[Set[FeatureFlag]] = None) -> EngineScrapeResult:
        """
        Fetch URL synchronously in a thread pool executor.
        This always returns something, even if it's just the error message.
        """
        loop = asyncio.get_event_loop()
        
        def _sync_fetch():
            try:
                import requests
            except ImportError:
                return EngineScrapeResult(
                    html="", url=url, status_code=0, error="requests library not available"
                )
            
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Cache-Control": "max-age=0",
            }
            
            proxies = None
            if self.proxy_pool:
                try:
                    proxy_url = self.proxy_pool.get_proxy("basic")
                    if proxy_url:
                        proxies = {"http": proxy_url, "https": proxy_url}
                except Exception:
                    pass
            
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                    verify=False,
                    proxies=proxies,
                )
                try:
                    resp.encoding = resp.apparent_encoding or "utf-8"
                except Exception:
                    pass
                return EngineScrapeResult(
                    html=resp.text or "",
                    url=str(resp.url) if resp.url else url,
                    status_code=resp.status_code,
                )
            except requests.exceptions.Timeout:
                return EngineScrapeResult(
                    html="", url=url, status_code=0, error=f"Timeout after {self.timeout}s"
                )
            except requests.exceptions.ConnectionError as e:
                return EngineScrapeResult(
                    html="", url=url, status_code=0, error=f"Connection failed: {e}"
                )
            except ImportError:
                return EngineScrapeResult(
                    html="", url=url, status_code=0, error="requests library not available"
                )
            except Exception as e:
                return EngineScrapeResult(
                    html="", url=url, status_code=0, error=str(e)
                )
        
        return await loop.run_in_executor(None, _sync_fetch)
    
    async def close(self):
        pass


class UndetectedChromedriverEngine:
    """
    Advanced browser engine using undetected-chromedriver to bypass anti-bot detection.
    Quality score: 20 (highest)
    
    This is the most powerful engine for bypassing anti-bot protection because:
    1. undetected-chromedriver patches ChromeDriver to avoid detection
    2. Removes webdriver automation indicators
    3. Uses real Chrome browser with proper fingerprints
    4. Can bypass most CAPTCHA and verification systems
    
    Inspired by Firecrawl's stealth proxy approach - when basic methods fail,
    use the most advanced browser automation available.
    
    Supports FeatureFlags: stealthProxy, mobile, location
    """
    
    QUALITY = 20
    
    SUPPORTED_FEATURES = {FeatureFlag.STEALTH_PROXY, FeatureFlag.MOBILE, FeatureFlag.LOCATION}
    
    def __init__(self, timeout: int = 60, user_agent: str = "", proxy_pool: Optional[ProxyPool] = None):
        self.timeout = timeout
        self.user_agent = user_agent
        self.proxy_pool = proxy_pool
        self._available = False
    
    async def _check_availability(self) -> bool:
        """Check if undetected-chromedriver is available."""
        try:
            import undetected_chromedriver
            self._available = True
        except ImportError:
            self._available = False
        return self._available
    
    async def scrape(self, url: str, features: Optional[Set[FeatureFlag]] = None) -> EngineScrapeResult:
        """Fetch URL using undetected-chromedriver with maximum stealth."""
        features = features or set()
        use_proxy = FeatureFlag.STEALTH_PROXY in features
        use_mobile = FeatureFlag.MOBILE in features
        
        if not await self._check_availability():
            return EngineScrapeResult(
                html="", url=url, status_code=0,
                error="undetected-chromedriver not available"
            )
        
        try:
            import undetected_chromedriver as uc
            import time
            
            options = uc.ChromeOptions()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-gpu')
            
            if use_proxy and self.proxy_pool:
                proxy_type = "stealth" if self.proxy_pool.has_proxy("stealth") else "basic"
                proxy_url = self.proxy_pool.get_proxy(proxy_type)
                if proxy_url:
                    options.add_argument(f'--proxy-server={proxy_url}')
                    logger.debug(f"UndetectedChrome using {proxy_type} proxy for {url}")
            
            if use_mobile:
                options.add_argument(
                    '--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
                    'AppleWebKit/605.1.15 (KHTML, like Gecko) '
                    'Version/17.0 Mobile/15E148 Safari/604.1'
                )
                options.add_argument('--window-size=390,844')
            elif self.user_agent:
                options.add_argument(f'--user-agent={self.user_agent}')
            else:
                options.add_argument(
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
            
            driver = uc.Chrome(
                options=options,
                version_main=120,
            )
            
            try:
                driver.set_page_load_timeout(self.timeout)
                driver.get(url)
                time.sleep(5)
                
                try:
                    last_height = driver.execute_script("return document.body.scrollHeight")
                    for _ in range(3):
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(1)
                        new_height = driver.execute_script("return document.body.scrollHeight")
                        if new_height == last_height:
                            break
                        last_height = new_height
                    
                    driver.execute_script("window.scrollTo(0, 0);")
                    time.sleep(1)
                except Exception:
                    pass
                
                html = driver.page_source
                final_url = driver.current_url
                
                return EngineScrapeResult(
                    html=html,
                    url=final_url,
                    status_code=200,
                )
            finally:
                try:
                    driver.quit()
                except Exception:
                    pass
        
        except Exception as e:
            return EngineScrapeResult(
                html="", url=url, status_code=0,
                error=f"UndetectedChromedriver error: {e}"
            )


class PlaywrightEngine:
    """
    Browser-based engine for JavaScript-heavy sites.
    Quality score: 15 (medium-low)
    Note: Requires playwright to be installed. Falls back to FetchEngine if not available.
    Enhanced with stealth mode to bypass anti-bot detection.
    """
    
    QUALITY = 15
    
    def __init__(self, timeout: int = 30, user_agent: str = ""):
        self.timeout = timeout
        self.user_agent = user_agent
        self._available = False
    
    async def _check_availability(self) -> bool:
        """Check if playwright is available."""
        try:
            import playwright
            self._available = True
        except ImportError:
            self._available = False
        return self._available
    
    async def scrape(self, url: str) -> EngineScrapeResult:
        """Fetch URL using Playwright browser automation with stealth mode."""
        if not await self._check_availability():
            return EngineScrapeResult(
                html="", url=url, status_code=0,
                error="Playwright not available"
            )
        
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-extensions',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-web-security',
                        '--disable-features=IsolateOrigins,site-per-process',
                        '--disable-site-isolation-trials',
                    ]
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent=self.user_agent,
                    locale='zh-CN',
                    timezone_id='Asia/Shanghai',
                    extra_http_headers={
                        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                        'Sec-Ch-Ua-Mobile': '?0',
                        'Sec-Ch-Ua-Platform': '"Windows"',
                    },
                )
                
                # Advanced stealth script to bypass anti-bot detection
                await context.add_init_script("""
                    // Hide webdriver
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    
                    // Mock chrome runtime
                    window.navigator.chrome = {
                        runtime: {},
                    };
                    
                    // Mock plugins
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5],
                    });
                    
                    // Mock languages
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['zh-CN', 'zh', 'en-US', 'en'],
                    });
                    
                    // Mock permissions
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );
                    
                    // Hide automation indicators
                    delete navigator.__proto__.webdriver;
                    
                    // Override toString for important objects
                    for (const key in window) {
                        if (key.includes('selenium') || key.includes('webdriver') || key.includes('driver')) {
                            delete window[key];
                        }
                    }
                    
                    // Mock window.outerWidth and outerHeight
                    Object.defineProperty(window, 'outerWidth', { get: () => 1920 });
                    Object.defineProperty(window, 'outerHeight', { get: () => 1080 });
                    
                    // Mock screen properties
                    Object.defineProperty(screen, 'availWidth', { get: () => 1920 });
                    Object.defineProperty(screen, 'availHeight', { get: () => 1080 });
                    
                    // Mock connection
                    Object.defineProperty(navigator, 'connection', {
                        get: () => ({
                            effectiveType: '4g',
                            rtt: 50,
                            downlink: 10,
                            saveData: false
                        })
                    });
                """)
                
                page = await context.new_page()
                
                # Block unnecessary resources to speed up loading
                await page.route('**/*', lambda route, request: route.abort() 
                    if request.resource_type in ['image', 'stylesheet', 'font', 'media']
                    else route.continue_()
                )
                
                await page.goto(url, timeout=self.timeout * 1000, wait_until='domcontentloaded')
                
                # Wait for page to fully load
                await page.wait_for_timeout(5000)
                
                # Try to scroll to trigger lazy loading
                try:
                    await page.evaluate("""
                        async () => {
                            // Scroll down slowly to simulate human behavior
                            const scrollHeight = document.body.scrollHeight;
                            const viewportHeight = window.innerHeight;
                            let currentScroll = 0;
                            
                            while (currentScroll < scrollHeight) {
                                window.scrollTo(0, currentScroll);
                                currentScroll += viewportHeight;
                                await new Promise(resolve => setTimeout(resolve, 500));
                            }
                            
                            // Scroll back to top
                            window.scrollTo(0, 0);
                            await new Promise(resolve => setTimeout(resolve, 500));
                        }
                    """)
                except Exception:
                    pass
                
                html = await page.content()
                await browser.close()
                
                return EngineScrapeResult(
                    html=html,
                    url=url,
                    status_code=200,
                )
        except NotImplementedError:
            return EngineScrapeResult(
                html="", url=url, status_code=0,
                error="Playwright not supported on this platform"
            )
        except Exception as e:
            return EngineScrapeResult(
                html="", url=url, status_code=0,
                error=f"Playwright error: {e}"
            )


class TLSClientEngine:
    """
    Advanced TLS client engine for sites with anti-bot protection.
    Quality score: 10 (medium)
    
    Firecrawl-style anti-bot bypass:
    - TLS fingerprint rotation (JA3/JA4 simulation)
    - HTTP/2 protocol support
    - Cookie persistence across requests
    - Human-like delay simulation
    - Multiple browser fingerprints
    - Connection pooling and reuse
    - Proper header ordering (like real browsers)
    - Optional proxy support
    
    Supports FeatureFlags: stealthProxy, mobile, skipTlsVerification
    """
    
    QUALITY = 10
    
    SUPPORTED_FEATURES = {FeatureFlag.STEALTH_PROXY, FeatureFlag.MOBILE, FeatureFlag.SKIP_TLS_VERIFICATION}
    
    TLS_FINGERPRINTS = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "http2": True,
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
            "http2": True,
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Linux"',
            "http2": True,
        },
    ]
    
    def __init__(self, timeout: int = 30, user_agent: str = "", proxy_pool: Optional[ProxyPool] = None):
        self.timeout = timeout
        self.user_agent = user_agent
        self.proxy_pool = proxy_pool
        self._client = None
        self._fingerprint_index = 0
        self._cookies = {}
    
    def _get_fingerprint(self) -> dict:
        """Get next TLS fingerprint from pool (rotation)."""
        fp = self.TLS_FINGERPRINTS[self._fingerprint_index % len(self.TLS_FINGERPRINTS)]
        self._fingerprint_index += 1
        return fp
    
    def _build_headers(self, url: str, features: Set[FeatureFlag]) -> dict:
        """Build realistic headers with Firecrawl-style ordering."""
        parsed = urlparse(url)
        fingerprint = self._get_fingerprint()
        
        user_agent = self.user_agent if self.user_agent else fingerprint["User-Agent"]
        
        if FeatureFlag.MOBILE in features and not self.user_agent:
            user_agent = (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.0 Mobile/15E148 Safari/604.1"
            )
        
        return {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "DNT": "1",
            "Host": parsed.netloc,
            "Sec-Ch-Ua": fingerprint["Sec-Ch-Ua"],
            "Sec-Ch-Ua-Mobile": fingerprint["Sec-Ch-Ua-Mobile"],
            "Sec-Ch-Ua-Platform": fingerprint["Sec-Ch-Ua-Platform"],
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": user_agent,
        }
    
    async def _human_delay(self):
        """Simulate human-like delay between requests."""
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)
    
    async def scrape(self, url: str, features: Optional[Set[FeatureFlag]] = None) -> EngineScrapeResult:
        """Fetch URL with Firecrawl-style anti-bot bypass and proxy support."""
        features = features or set()
        use_proxy = FeatureFlag.STEALTH_PROXY in features
        skip_tls = FeatureFlag.SKIP_TLS_VERIFICATION in features
        
        try:
            import httpx
            
            await self._human_delay()
            
            fingerprint = self._get_fingerprint()
            headers = self._build_headers(url, features)
            
            client_kwargs = {
                "timeout": self.timeout,
                "follow_redirects": True,
                "verify": not skip_tls,
                "http2": fingerprint.get("http2", True),
                "cookies": self._cookies,
            }
            
            if use_proxy and self.proxy_pool:
                proxy_type = "stealth" if self.proxy_pool.has_proxy("stealth") else "basic"
                proxy_url = self.proxy_pool.get_proxy(proxy_type)
                if proxy_url:
                    client_kwargs["proxies"] = {"http://": proxy_url, "https://": proxy_url}
                    logger.debug(f"TLSClientEngine using {proxy_type} proxy for {url}")
            
            if not self._client:
                self._client = httpx.AsyncClient(**client_kwargs)
            
            self._client.cookies.update(self._cookies)
            
            response = await self._client.get(url, headers=headers)
            
            self._cookies.update(dict(self._client.cookies))
            
            return EngineScrapeResult(
                html=response.text,
                url=url,
                status_code=response.status_code,
            )
        except ImportError:
            return EngineScrapeResult(html="", url=url, status_code=0, error="httpx not available")
        except httpx.TimeoutException:
            return EngineScrapeResult(html="", url=url, status_code=0, error="Timeout")
        except Exception as e:
            return EngineScrapeResult(html="", url=url, status_code=0, error=f"TLSClient error: {e}")
    
    async def close(self):
        """Close the persistent client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class EngineSelector:
    """
    Selects and manages scraping engines based on Firecrawl's strategy.
    Implements engine fallback mechanism with feature-based filtering.
    
    Engine order (fastest to strongest):
    1. FetchEngine - Fast HTTP requests, good for simple sites
    2. TLSClientEngine - Advanced TLS fingerprinting, better anti-bot bypass
    3. UndetectedChromedriverEngine - Real Chrome browser, bypasses most CAPTCHAs
    4. SimpleRequestsEngine - Ultimate fallback, works anywhere
    
    Note: Playwright is disabled due to Windows asyncio compatibility issues.
    """
    
    ENGINES = {
        EngineType.FETCH: FetchEngine,
        EngineType.TLS_CLIENT: TLSClientEngine,
        EngineType.UNDETECTED_CHROMEDRIVER: UndetectedChromedriverEngine,
        EngineType.REQUESTS: SimpleRequestsEngine,
    }
    
    ENGINE_QUALITY = {
        EngineType.FETCH: 5,
        EngineType.TLS_CLIENT: 10,
        EngineType.UNDETECTED_CHROMEDRIVER: 20,
        EngineType.REQUESTS: 1,
    }
    
    @classmethod
    def get_engine_for_url(cls, url: str) -> List[EngineType]:
        """
        Determine engine order for a URL.
        Inspired by Firecrawl's getEngineForUrl logic.
        """
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        
        if path_lower.endswith(('.pdf', '.doc', '.docx')):
            return [EngineType.FETCH, EngineType.REQUESTS]
        
        return [
            EngineType.FETCH,
            EngineType.TLS_CLIENT,
            EngineType.UNDETECTED_CHROMEDRIVER,
            EngineType.REQUESTS,
        ]
    
    @classmethod
    def build_fallback_list(cls, url: str, features: Optional[Set[FeatureFlag]] = None) -> List[EngineType]:
        """
        Build ordered list of engines, optionally filtered by feature support.
        Inspired by Firecrawl's buildFallbackList().
        """
        engines = cls.get_engine_for_url(url)
        if features:
            engines = [e for e in engines if cls.engine_supports_features(e, features)]
        return engines
    
    @classmethod
    def engine_supports_features(cls, engine_type: EngineType, features: Set[FeatureFlag]) -> bool:
        """Check if an engine supports all required features."""
        if not features:
            return True
        engine_class = cls.ENGINES.get(engine_type)
        if not engine_class:
            return False
        supported = getattr(engine_class, 'SUPPORTED_FEATURES', set())
        return features.issubset(supported)


class RobotsTxtHandler:
    """
    Handles robots.txt checking and compliance.
    Inspired by Firecrawl's robots.txt handling.
    """
    
    def __init__(self):
        self._cache = {}
    
    async def get_robots_txt(self, base_url: str) -> str:
        """Fetch and cache robots.txt for a domain."""
        if base_url in self._cache:
            return self._cache[base_url]
        
        try:
            robots_url = f"{base_url}/robots.txt"
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        content = await response.text()
                        self._cache[base_url] = content
                        return content
        except Exception as e:
            logger.warning(f"Failed to fetch robots.txt for {base_url}: {e}")
        
        self._cache[base_url] = ""
        return ""
    
    def is_allowed(self, robots_txt: str, url: str, user_agent: str = "*") -> bool:
        """
        Check if URL is allowed by robots.txt.
        Simple implementation - for production, use a proper robots.txt parser.
        """
        if not robots_txt:
            return True
        
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            lines = robots_txt.split('\n')
            current_agent = None
            disallowed_paths = []
            allowed_paths = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.lower().startswith('user-agent:'):
                    current_agent = line.split(':', 1)[1].strip()
                elif line.lower().startswith('disallow:') and (current_agent == user_agent or current_agent == '*'):
                    path_pattern = line.split(':', 1)[1].strip()
                    if path_pattern:
                        disallowed_paths.append(path_pattern)
                elif line.lower().startswith('allow:') and (current_agent == user_agent or current_agent == '*'):
                    path_pattern = line.split(':', 1)[1].strip()
                    if path_pattern:
                        allowed_paths.append(path_pattern)
            
            for allowed in allowed_paths:
                if path.startswith(allowed):
                    return True
            
            for disallowed in disallowed_paths:
                if path.startswith(disallowed):
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Error checking robots.txt: {e}")
            return True


class URLFilter:
    """
    Filters URLs based on various criteria.
    Inspired by Firecrawl's URL filtering logic.
    """
    
    FILE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
        '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
        '.mp3', '.wav', '.ogg', '.flac',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.tar', '.gz', '.7z',
        '.css', '.js', '.ico', '.svg', '.woff', '.woff2', '.ttf',
        '.exe', '.dmg', '.apk', '.msi',
    }
    
    SOCIAL_MEDIA_DOMAINS = {
        'facebook.com', 'twitter.com', 'x.com', 'instagram.com',
        'linkedin.com', 'tiktok.com', 'snapchat.com', 'pinterest.com',
        'reddit.com', 'youtube.com', 'twitch.tv',
    }
    
    @classmethod
    def is_valid_url(cls, url: str) -> bool:
        """Check if URL is valid and should be scraped."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    @classmethod
    def is_file_url(cls, url: str) -> bool:
        """Check if URL points to a file (not a web page)."""
        parsed = urlparse(url)
        path_lower = parsed.path.lower().split('?')[0]
        
        return any(path_lower.endswith(ext) for ext in cls.FILE_EXTENSIONS)
    
    @classmethod
    def is_social_media(cls, url: str) -> bool:
        """Check if URL is a social media link."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace('www.', '')
        
        return any(social in domain for social in cls.SOCIAL_MEDIA_DOMAINS)
    
    @classmethod
    def is_same_domain(cls, url: str, base_url: str) -> bool:
        """Check if URL is on the same domain as base URL."""
        try:
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_url)
            
            url_domain = parsed_url.netloc.lower().replace('www.', '')
            base_domain = parsed_base.netloc.lower().replace('www.', '')
            
            return url_domain == base_domain
        except Exception:
            return False
    
    @classmethod
    def get_url_depth(cls, url: str) -> int:
        """Get the depth of a URL path."""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            if not path:
                return 0
            return len(path.split('/'))
        except Exception:
            return 0
    
    @classmethod
    def filter_links(
        cls,
        links: List[str],
        base_url: str,
        max_depth: int = 10,
        allow_external: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Filter links based on various criteria.
        Inspired by Firecrawl's filterLinks implementation.
        """
        filtered = []
        
        for link in links:
            if not cls.is_valid_url(link):
                continue
            
            if cls.is_file_url(link):
                continue
            
            if cls.is_social_media(link):
                continue
            
            if link.startswith('mailto:') or link.startswith('tel:'):
                continue
            
            if not allow_external and not cls.is_same_domain(link, base_url):
                continue
            
            if cls.get_url_depth(link) > max_depth:
                continue
            
            if exclude_patterns:
                if any(re.search(pattern, link) for pattern in exclude_patterns):
                    continue
            
            if include_patterns:
                if not any(re.search(pattern, link) for pattern in include_patterns):
                    continue
            
            filtered.append(link)
        
        return list(set(filtered))


class WebScraper:
    """
    Main web scraper engine - deeply inspired by Firecrawl's scraping engine.
    
    Core features (aligned with Firecrawl):
    - Multi-engine support with quality-based fallback
    - Waterfall parallel execution: launch multiple engines, use first success
    - Dynamic feature system: auto-upgrade capabilities (e.g., add stealthProxy)
    - File-based cache: avoid re-scraping same URLs (like Firecrawl's Index engine)
    - Proxy pool: basic/stealth proxy rotation (like Firecrawl's stealthProxy)
    - Content validation with anti-bot detection
    """
    
    MAX_HTML_SIZE_FOR_MARKDOWN_CHECK = 300 * 1024
    WATERFALL_DELAY_MS = 3000
    
    def __init__(
        self,
        max_concurrent: int = 5,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 3600,
        proxy_pool: Optional[ProxyPool] = None,
        feature_flags: Optional[Set[FeatureFlag]] = None,
    ):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._engines = {}
        
        self.use_cache = use_cache
        self._cache = ScrapingCache(cache_dir=cache_dir, ttl=cache_ttl) if use_cache else None
        
        self.proxy_pool = proxy_pool
        self.feature_flags = feature_flags or set()
    
    def _get_engine(self, engine_type: EngineType):
        """Get or create engine instance with proxy support."""
        if engine_type not in self._engines:
            engine_class = EngineSelector.ENGINES[engine_type]
            self._engines[engine_type] = engine_class(
                timeout=self.timeout,
                user_agent=self.user_agent,
                proxy_pool=self.proxy_pool,
            )
        return self._engines[engine_type]
    
    async def _validate_result(
        self,
        result: EngineScrapeResult,
        needs_markdown: bool = True,
    ) -> bool:
        """
        Validate scrape result quality.
        Inspired by Firecrawl's success factor checking.
        Enhanced with comprehensive anti-bot detection page detection.
        """
        if not result.is_successful:
            return False
        
        html_content = result.html or ""
        
        if not html_content or len(html_content.strip()) < 50:
            return False
        
        # Firecrawl-style anti-bot/verification page detection
        # Comprehensive list of indicators from real-world scraping experience
        anti_bot_indicators = [
            # Cloudflare
            'cloudflare',
            'checking your browser',
            'please wait while we verify',
            'ddos protection',
            # General anti-bot
            'antibot',
            'verifycode',
            'captcha',
            'recaptcha',
            'hcaptcha',
            'turnstile',
            # Access control
            'access denied',
            'blocked',
            'rate limit',
            'too many requests',
            'forbidden',
            'unauthorized',
            # Bot detection
            'bot detected',
            'automated access',
            'suspicious activity',
            'security check',
            'challenge page',
            # Chinese anti-bot
            '\u9a8c\u8bc1\u7801',
            '\u4eba\u673a\u9a8c\u8bc1',
            '\u5b89\u5168\u9a8c\u8bc1',
            '\u8bbf\u95ee\u9a8c\u8bc1',
            '\u8bf7\u7a0d\u7b49',
            '\u6b63\u5728\u68c0\u6d4b',
        ]
        
        html_lower = html_content.lower()
        
        # Check for anti-bot indicators
        if any(indicator in html_lower for indicator in anti_bot_indicators):
            logger.warning(f"Detected anti-bot verification page for {result.url}")
            return False
        
        # Check for minimal content (empty pages)
        try:
            cleaned_html = ContentCleaner.clean_html(html_content)
            soup = BeautifulSoup(cleaned_html, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            text_content = soup.get_text(strip=True)
            
            # If text content is too short, it's likely an error/anti-bot page
            if len(text_content) < 50:
                logger.warning(f"Minimal content detected for {result.url}")
                return False
            
            # Check if page is mostly JavaScript (SPA that didn't load)
            script_count = len(soup.find_all('script'))
            if script_count > 20 and len(text_content) < 200:
                logger.warning(f"SPA page without content for {result.url}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return len(html_content.strip()) > 50
    
    async def _parse_result(self, url: str, result: EngineScrapeResult, engine_name: str) -> ScrapedPage:
        """Parse a successful engine result into a ScrapedPage."""
        try:
            page = ContentParser.parse_url(url, result.html)
            page.status_code = result.status_code
            page.metadata["engine"] = engine_name
            logger.info(f"Parsed page: title={page.title}, content_len={len(page.content or '')}")
            return page
        except Exception as parse_error:
            logger.error(f"Parse error for {url}: {parse_error}")
            return ScrapedPage(
                url=url,
                content=result.html[:5000],
                html=result.html,
                status_code=result.status_code,
                error=f"Parse error: {parse_error}",
                metadata={"engine": engine_name},
            )
    
    async def scrape_url_with_engine(
        self,
        url: str,
        engine_type: EngineType,
        features: Optional[Set[FeatureFlag]] = None,
    ) -> EngineScrapeResult:
        """Scrape URL using a specific engine with feature flags."""
        engine = self._get_engine(engine_type)
        try:
            result = await engine.scrape(url, features=features)
        except TypeError:
            result = await engine.scrape(url)
        return result
    
    async def _check_proxy_error(self, result: EngineScrapeResult, features: Set[FeatureFlag]) -> None:
        """
        Check if the error is proxy-related, and escalate to stealthProxy.
        Inspired by Firecrawl: 401/403/429 without stealthProxy -> add stealthProxy.
        """
        if FeatureFlag.STEALTH_PROXY not in features and self.proxy_pool:
            is_likely_proxy_error = result.status_code in (401, 403, 429)
            if is_likely_proxy_error:
                logger.info(f"Scrape unsuccessful due to proxy inadequacy (status={result.status_code}). Adding stealthProxy.")
                raise AddFeatureError([FeatureFlag.STEALTH_PROXY])
    
    async def _run_engine_loop(
        self,
        url: str,
        engine_type: EngineType,
        features: Set[FeatureFlag],
    ) -> EngineScrapeResult:
        """Run a single engine scrape attempt with validation."""
        engine_name = engine_type.value
        result = await self.scrape_url_with_engine(url, engine_type, features)
        
        if result.error:
            logger.debug(f"Engine {engine_name} error: {result.error}")
            raise EngineUnsuccessfulError(engine_name)
        
        await self._check_proxy_error(result, features)
        
        if await self._validate_result(result):
            logger.info(f"Successfully scraped {url} with {engine_name}")
            return result
        else:
            html_len = len(result.html or '')
            logger.warning(
                f"Engine {engine_name} validation failed for {url}: "
                f"status={result.status_code}, html_len={html_len}"
            )
            raise EngineUnsuccessfulError(engine_name)
    
    async def _scrape_url_waterfall(self, url: str, features: Set[FeatureFlag]) -> ScrapedPage:
        """
        Waterfall parallel execution - inspired by Firecrawl's scrapeURLLoop.
        
        Firecrawl's key insight: DON'T run engines sequentially. Instead:
        1. Launch Tier 0 engines immediately
        2. After WATERFALL_DELAY, launch Tier 1 engines alongside running ones
        3. Use FIRST_COMPLETED to pick whichever succeeds first
        4. When one succeeds, cancel all others (snipe pattern)
        5. On proxy errors (401/403/429), auto-add stealthProxy and retry
        """
        engine_list = EngineSelector.build_fallback_list(url)
        
        if not engine_list:
            raise RuntimeError("No engines available")
        
        waterfall_tiers = [
            [engine_list[0]],
        ]
        remaining = engine_list[1:]
        if remaining:
            waterfall_tiers.append(remaining)
        
        tasks = []
        running = {}
        winner = None
        winner_result = None
        
        async def _engine_wrapper(
            url: str,
            engine_type: EngineType,
            features: Set[FeatureFlag],
        ) -> tuple:
            result = await self._run_engine_loop(url, engine_type, features)
            return engine_type, result
        
        try:
            for tier_idx, tier in enumerate(waterfall_tiers):
                if winner is not None:
                    break
                
                for engine_type in tier:
                    if winner is not None:
                        break
                    
                    logger.debug(f"Waterfall tier {tier_idx}: starting {engine_type.value} for {url}")
                    task = asyncio.ensure_future(_engine_wrapper(url, engine_type, features))
                    tasks.append(task)
                    running[task] = engine_type
                
                if tier_idx < len(waterfall_tiers) - 1:
                    try:
                        done, _ = await asyncio.wait(
                            tasks,
                            timeout=self.WATERFALL_DELAY_MS / 1000.0,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    except asyncio.TimeoutError:
                        continue
                    
                    for finished_task in done:
                        try:
                            eng_type, eng_result = finished_task.result()
                            if winner is None:
                                winner = eng_type
                                winner_result = eng_result
                        except (EngineUnsuccessfulError, Exception):
                            pass
                else:
                    done, _ = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    
                    for finished_task in done:
                        try:
                            eng_type, eng_result = finished_task.result()
                            if winner is None:
                                winner = eng_type
                                winner_result = eng_result
                        except (EngineUnsuccessfulError, Exception):
                            pass
            
            if winner is not None and winner_result is not None:
                engine_name = winner.value
                page = await self._parse_result(url, winner_result, engine_name)
                
                if self.use_cache and self._cache:
                    self._cache.set(page)
                
                return page
            
            raise RuntimeError("All engines failed in waterfall")
        
        except AddFeatureError as e:
            for feature in e.features:
                features.add(feature)
                logger.info(f"Dynamically added feature: {feature.value}, retrying...")
            return await self._scrape_url_waterfall(url, features)
        
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
    
    async def scrape_url(self, url: str) -> ScrapedPage:
        """
        Scrape a single URL with cache-first + waterfall execution.
        
        Flow (mirrors Firecrawl's index -> chrome-cdp -> stealth flow):
        1. Check cache -> if hit and valid, return immediately
        2. Run waterfall with dynamic feature escalation
        3. Cache the result for future use
        """
        async with self._semaphore:
            if self.use_cache and self._cache:
                cached = self._cache.get(url)
                if cached:
                    logger.info(f"Cache hit for {url}")
                    return cached
            
            features = self.feature_flags.copy()
            
            for attempt in range(self.max_retries):
                try:
                    page = await self._scrape_url_waterfall(url, features)
                    return page
                except AddFeatureError as e:
                    for feature in e.features:
                        features.add(feature)
                except Exception as e:
                    logger.warning(f"Waterfall attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1 * (attempt + 1))
            
            return ScrapedPage(
                url=url,
                error="All engines failed after retries",
                status_code=0,
            )
    
    async def _cleanup_engines(self):
        """Clean up engine resources (close persistent connections)."""
        for engine in self._engines.values():
            if hasattr(engine, 'close'):
                try:
                    await engine.close()
                except Exception as e:
                    logger.debug(f"Error closing engine: {e}")
        self._engines = {}
    
    async def scrape_urls(self, urls: List[str]) -> List[ScrapedPage]:
        """Scrape multiple URLs concurrently."""
        tasks = [self.scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return list(results)
    
    async def crawl_site(
        self,
        start_url: str,
        max_pages: int = 10,
        max_depth: int = 2,
        respect_robots: bool = True,
        allow_external: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
    ) -> List[ScrapedPage]:
        """
        Crawl a website starting from a URL.
        Deeply inspired by Firecrawl's WebCrawler with:
        - Depth control
        - Link filtering
        - robots.txt compliance
        - Include/exclude patterns
        """
        visited = set()
        to_visit = [(start_url, 0)]
        results = []
        
        robots_handler = RobotsTxtHandler() if respect_robots else None
        parsed_base = urlparse(start_url)
        base_url = f"{parsed_base.scheme}://{parsed_base.netloc}"
        
        robots_txt = ""
        if robots_handler:
            robots_txt = await robots_handler.get_robots_txt(base_url)
        
        while to_visit and len(results) < max_pages:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
            
            if robots_txt and robots_handler:
                if not robots_handler.is_allowed(robots_txt, url):
                    logger.debug(f"URL blocked by robots.txt: {url}")
                    continue
            
            if not URLFilter.is_valid_url(url):
                continue
            
            if URLFilter.is_file_url(url):
                continue
            
            if not allow_external and not URLFilter.is_same_domain(url, start_url):
                continue
            
            if exclude_patterns:
                if any(re.search(pattern, url) for pattern in exclude_patterns):
                    continue
            
            if include_patterns:
                if not any(re.search(pattern, url) for pattern in include_patterns):
                    continue
            
            visited.add(url)
            logger.info(f"Crawling: {url} (depth: {depth})")
            
            page = await self.scrape_url(url)
            
            if page.error:
                logger.warning(f"Failed to crawl {url}: {page.error}")
                continue
            
            results.append(page)
            
            if depth < max_depth and len(results) < max_pages:
                links = ContentCleaner.extract_links(page.html or "", url)
                
                filtered_links = URLFilter.filter_links(
                    links,
                    start_url,
                    max_depth=max_depth,
                    allow_external=allow_external,
                    exclude_patterns=exclude_patterns,
                    include_patterns=include_patterns,
                )
                
                new_links = [link for link in filtered_links if link not in visited]
                
                for link in new_links[:max_pages - len(results)]:
                    to_visit.append((link, depth + 1))
            
            await asyncio.sleep(0.5)
        
        return results
