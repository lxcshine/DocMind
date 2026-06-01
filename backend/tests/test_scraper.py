"""
Test script for the refactored web scraper.
Tests engine fallback, content validation, and URL filtering.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.web_scraper import (
    WebScraper,
    URLFilter,
    RobotsTxtHandler,
    EngineSelector,
    EngineType,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_single_url_scrape():
    """Test scraping a single URL with engine fallback."""
    logger.info("=" * 60)
    logger.info("Test 1: Single URL Scrape with Engine Fallback")
    logger.info("=" * 60)
    
    scraper = WebScraper(
        max_concurrent=3,
        timeout=30,
        max_retries=2,
    )
    
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]
    
    for url in test_urls:
        logger.info(f"\nScraping: {url}")
        page = await scraper.scrape_url(url)
        
        if page.error:
            logger.error(f"Error: {page.error}")
        else:
            logger.info(f"Success!")
            logger.info(f"  Title: {page.title}")
            logger.info(f"  Status: {page.status_code}")
            logger.info(f"  Content length: {len(page.content)} chars")
            logger.info(f"  Markdown length: {len(page.markdown)} chars")
            logger.info(f"  Word count: {page.metadata.get('word_count', 0)}")


async def test_multiple_urls():
    """Test scraping multiple URLs concurrently."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Multiple URLs Concurrent Scraping")
    logger.info("=" * 60)
    
    scraper = WebScraper(max_concurrent=5)
    
    urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://httpbin.org/json",
    ]
    
    logger.info(f"Scraping {len(urls)} URLs concurrently...")
    results = await scraper.scrape_urls(urls)
    
    success_count = sum(1 for r in results if not r.error)
    logger.info(f"Results: {success_count}/{len(results)} successful")
    
    for result in results:
        status = "?" if not result.error else "?"
        logger.info(f"  {status} {result.url}: {result.title or 'No title'}")


async def test_url_filtering():
    """Test URL filtering logic."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: URL Filtering")
    logger.info("=" * 60)
    
    test_links = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/image.jpg",
        "https://example.com/document.pdf",
        "https://facebook.com/some-page",
        "https://example.com/deep/nested/page",
        "mailto:test@example.com",
        "https://external.com/page",
    ]
    
    base_url = "https://example.com"
    
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Input links: {len(test_links)}")
    
    filtered = URLFilter.filter_links(
        test_links,
        base_url,
        max_depth=2,
        allow_external=False,
    )
    
    logger.info(f"Filtered links: {len(filtered)}")
    for link in filtered:
        logger.info(f"  ? {link}")


async def test_engine_selection():
    """Test engine selection for different URL types."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Engine Selection Strategy")
    logger.info("=" * 60)
    
    test_urls = [
        "https://example.com/page",
        "https://example.com/document.pdf",
        "https://twitter.com/user",
        "https://x.com/user",
    ]
    
    for url in test_urls:
        engines = EngineSelector.get_engine_for_url(url)
        engine_names = [e.value for e in engines]
        logger.info(f"URL: {url}")
        logger.info(f"  Engine order: {' -> '.join(engine_names)}")


async def test_robots_txt():
    """Test robots.txt handling."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: robots.txt Handling")
    logger.info("=" * 60)
    
    handler = RobotsTxtHandler()
    
    test_urls = [
        "https://example.com",
        "https://httpbin.org",
    ]
    
    for base_url in test_urls:
        logger.info(f"\nFetching robots.txt for: {base_url}")
        robots_txt = await handler.get_robots_txt(base_url)
        
        if robots_txt:
            logger.info(f"  Found ({len(robots_txt)} chars)")
            logger.info(f"  Preview: {robots_txt[:100]}...")
        else:
            logger.info("  No robots.txt found")


async def test_crawl_site():
    """Test crawling a small site."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Site Crawling (Limited)")
    logger.info("=" * 60)
    
    scraper = WebScraper(max_concurrent=2)
    
    logger.info("Crawling https://example.com (max 3 pages, depth 1)...")
    
    results = await scraper.crawl_site(
        start_url="https://example.com",
        max_pages=3,
        max_depth=1,
        respect_robots=True,
    )
    
    logger.info(f"Crawled {len(results)} pages:")
    for page in results:
        logger.info(f"  ? {page.url}: {page.title}")


async def main():
    """Run all tests."""
    logger.info("Starting Web Scraper Tests")
    logger.info("Based on Firecrawl architecture\n")
    
    try:
        await test_single_url_scrape()
        await test_multiple_urls()
        await test_url_filtering()
        await test_engine_selection()
        await test_robots_txt()
        await test_crawl_site()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
