"""
Search API - Deep web search with real content scraping
Workflow:
1. User submits search query
2. System uses Bing/Baidu/Sogou search engines to get real URLs
3. Deep scrape each URL to extract actual content
4. Process and synthesize all scraped content
5. LLM generates comprehensive answer based on real content
6. Results include answer, sources, and detailed scraped content
"""
# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import asyncio
import re
import os
import base64
from io import BytesIO
from urllib.parse import urlparse
from openai import OpenAI
from config.settings import settings
from core.web_scraper import WebScraper, ScrapedPage
from core.chat_history import ChatHistoryStore

logger = logging.getLogger(__name__)
router = APIRouter()
llm_client = OpenAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.GEMINI_BASE_URL,
)
history_store = ChatHistoryStore(
    host=settings.MYSQL_HOST,
    port=settings.MYSQL_PORT,
    user=settings.MYSQL_USER,
    password=settings.MYSQL_PASSWORD,
    database=settings.MYSQL_DATABASE,
)
def detect_user_intent(query: str) -> Dict[str, Any]:

    """
    Detect user intent from query.
    Identifies if user wants data analysis, visualization, or just search.
    """
    query_lower = query.lower()
    chart_keywords = [
        '折线图', '柱状图', '饼图', '散点图',
        '图表', '画图', '可视化', '统计图',
        'line chart', 'bar chart', 'pie chart', 'scatter plot', 'chart', 'graph', 'visualization',
    ]
    analysis_keywords = [
        '数据分析', '统计分析', '趋势分析', '对比分析',
        '提取数据', '数据提取', '数据处理',
        'data analysis', 'analyze data', 'extract data', 'statistics', 'trend', 'compare',
    ]
    has_chart = any(kw in query_lower for kw in chart_keywords)
    has_analysis = any(kw in query_lower for kw in analysis_keywords)
    intent = "search"
    if has_chart and has_analysis:
        intent = "analyze_and_chart"
    elif has_chart:
        intent = "chart"
    elif has_analysis:
        intent = "analyze"
    return {
        "intent": intent,
        "needs_chart": has_chart,
        "needs_analysis": has_analysis,
    }
CHINESE_FILLER_WORDS = {
    '帮我搜索一下', '帮我搜一下', '帮我查一下', '帮我找一下',
    '查一下', '查一查', '查查', '查看看', '查一个', '查个',
    '帮我查', '帮我找', '帮我搜', '帮我搜索',
    '查询一下', '搜索一下', '查找一下', '检索一下',
    '搜一下', '搜搜', '搜一个', '搜个',
    '找一下', '找一找', '找找', '找一个', '找个',
    '看一下', '看一看', '看看', '看一个', '看个',
    '请问', '想问', '想知道', '我想知道', '我想了解',
    '麻烦', '能不能', '可不可以', '是否', '有没有',
    '是什么', '什么是', '怎么样', '如何', '怎么',
    '请', '求', '告诉我', '来一个', '给一个',
    '查询', '搜索', '查找', '检索',
    '一下', '一个', '一次', '一些', '有关', '关于',
    '情况', '内容', '方面', '信息', '数据', '资料',
}
CHINESE_STOP_WORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就',
    '不', '人', '都', '一', '一个', '上', '也', '很',
    '到', '说', '要', '去', '你', '会', '着', '没',
    '看', '好', '自己', '这', '他', '她', '它', '们',
    '那', '哪', '什么', '怎么', '如何', '为什么',
    '可以', '吗', '呢', '吧', '啊', '哦', '嗯', '呀',
    '嘛', '呵', '哈', '哇', '哟', '噢',
    '请', '让', '把', '被', '给', '跟', '对', '从',
    '于', '向', '由', '用', '以', '而', '且', '或',
    '但', '如果', '因为', '所以', '虽然', '然而',
    '不过', '还是', '只是', '就是', '还有', '或者',
    '这个', '那个', '这些', '那些', '这里', '那里',
    '现在', '目前', '今天', '昨天', '明天',
    '非常', '真的', '比较', '更加', '越来越',
    '已经', '曾经', '将会', '可能', '应该',
    '需要', '必须', '能够', '可以', '会',
    '的', '得', '地', '之', '等', '等等',
    '多', '少', '多少', '几', '一些', '一点',
    '时', '时候', '年', '月', '日', '天',
    '前', '后', '里', '外', '中', '内', '间',
    '下面', '上面', '前面', '后面', '旁边',
}
def extract_chinese_keywords(query: str) -> str:

    """
    Extract meaningful Chinese keywords by removing filler words and stop words.
    Used as a fast fallback when LLM is unavailable.
    Example: '查一下上海2026年的房价' -> '上海 2026 房价'
    """
    result = query.strip()
    for filler in sorted(CHINESE_FILLER_WORDS, key=len, reverse=True):
        result = result.replace(filler, ' ')
    parts = result.split()
    clean_parts = []
    for part in parts:
        if part not in CHINESE_STOP_WORDS and len(part.strip()) >= 1:
            clean_parts.append(part)
    keywords = ' '.join(clean_parts).strip()
    if not keywords or len(keywords) < 2:
        return query.strip()
    return keywords
QUERY_PREPROCESS_SYSTEM_PROMPT = """You are a search query optimizer. Your task is to convert natural language questions into effective search engine keywords.
Rules:
1. Extract ONLY the core search terms - remove all filler words like "查一下", "帮我", "请问", "what is", "can you find"
2. Output ONLY the optimized keywords separated by spaces, nothing else
3. Keep important qualifiers like year, location, category
4. Preserve proper nouns and technical terms exactly as-is
Examples:
- "查一下上海2026年的房价" -> "上海 2026年 房价"
- "帮我搜索最近关于AI的新闻" -> "AI 新闻 最新"
- "What is the weather like in Tokyo next week?" -> "Tokyo weather forecast next week"
- "找一下北京到上海的火车票" -> "北京 上海 火车票"""

async def preprocess_query(query: str) -> str:

    """
    Convert natural language query to optimized search keywords.
    Uses LLM to extract core keywords. Falls back to Chinese keyword extraction
    if LLM is unavailable or fails.
    """
    query = query.strip()
    if len(query) <= 10:
        return query
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
    has_filler = any(filler in query for filler in CHINESE_FILLER_WORDS)
    if not has_chinese or not has_filler:
        return query
    try:
        response = llm_client.chat.completions.create(
            model=settings.GEMINI_MODEL,
            messages=[
                {"role": "system", "content": QUERY_PREPROCESS_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=100,
        )
        keywords = response.choices[0].message.content.strip()
        if keywords and len(keywords) >= 3:
            logger.info(f"Query preprocessed: '{query}' -> '{keywords}'")
            return keywords
    except Exception as e:
        logger.warning(f"LLM query preprocessing failed: {e}, using fallback")
    fallback = extract_chinese_keywords(query)
    logger.info(f"Query preprocessed (fallback): '{query}' -> '{fallback}'")
    return fallback
def extract_url_from_text(text: str) -> Optional[str]:

    """
    Extract URL from text input.
    Returns the first URL found in the text, or None if no URL found.
    """
    url_pattern = re.compile(
        r'(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)', re.IGNORECASE)
    match = url_pattern.search(text)
    if match:
        url = match.group(0)
        # Clean up URL (remove trailing punctuation that's not part of URL)
        url = url.rstrip('.,;:!?\'"')
        return url
    return None
def is_valid_url(text: str) -> bool:

    """
    Check if the input text is a valid URL.
    Inspired by Firecrawl's URL validation logic.
    """
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    if url_pattern.match(text):
        return True
    if text.startswith('www.') and '.' in text:
        return True
    return False
def generate_chart_from_data(

        data_points: List[Dict[str, Any]],
        chart_type: str = "line",
        title: str = "Data Visualization",
        x_label: str = "X",
        y_label: str = "Y",
) -> str:
    """
    Generate a chart from structured data and return as base64 encoded image.
    Args:
        data_points: List of dicts with 'x' and 'y' keys
        chart_type: 'line', 'bar', 'scatter', 'pie'
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
    Returns:
        Base64 encoded PNG image string
    """
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(10, 6))
        x_values = [dp.get('x', '') for dp in data_points]
        y_values = [dp.get('y', 0) for dp in data_points]
        if chart_type == "line":
            ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=8, color='#1677ff')
            ax.fill_between(range(len(y_values)), y_values, alpha=0.1, color='#1677ff')
        elif chart_type == "bar":
            ax.bar(x_values, y_values, color='#1677ff', edgecolor='white', linewidth=1.5)
        elif chart_type == "scatter":
            ax.scatter(x_values, y_values, color='#1677ff', s=100, alpha=0.7)
        elif chart_type == "pie":
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(y_values, labels=x_values, autopct='%1.1f%%', startangle=90,
                   colors=plt.cm.Set3.colors[:len(x_values)])
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return img_base64
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        if chart_type != "pie":
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        return ""
def normalize_url(text: str) -> str:

    """Normalize URL by adding https:// if missing."""
    if text.startswith('www.'):
        return f'https://{text}'
    return text
class SearchRequest(BaseModel):

    query: str
    session_id: Optional[str] = None
    max_pages: int = 5
    max_depth: int = 1
    include_urls: Optional[List[str]] = None
class SearchResult(BaseModel):

    answer: str
    sources: List[Dict[str, Any]]
    scraped_pages: List[Dict[str, Any]]
class SearchResultItem(BaseModel):

    url: str
    title: str
    description: str
    position: int = 0
class SearchResultsResponse(BaseModel):

    query: str
    results: List[SearchResultItem]
    total: int
class ScrapePageRequest(BaseModel):

    url: str
class ScrapePageResponse(BaseModel):

    url: str
    title: str
    content: str
    markdown: str
    analysis: str
    metadata: Dict[str, Any] = {}
    status_code: int = 0
    error: Optional[str] = None
SEARCH_SYSTEM_PROMPT = """You are a powerful research assistant. Your ONLY job is to extract and present useful information from web search results.
CRITICAL RULES:
1. Answer in the SAME LANGUAGE as the user's question
2. NEVER say "I cannot find information" or "unable to find" — ALWAYS present what IS available
3. NEVER write long explanations about why information is unavailable — just present what you have
4. NEVER list sources and explain why each is irrelevant — skip irrelevant content silently
5. If some sources are irrelevant, just don't mention them
6. Extract every useful fact, number, and detail from the content you DO have
7. Be direct and concise — no meta-commentary about the search process
Response structure:
- Start with the most useful information you found
- Use headings, bullet points, and tables where appropriate
- Cite sources with [1], [2] only for factual claims
- End with a brief summary if helpful
Content rules:
- Extract numbers, dates, statistics, names — be specific
- Use markdown tables for structured data
- Use LaTeX for formulas ($$ display, $ inline)
- No filler sentences, no apologies, no disclaimers about search quality"""

class SearchResult:

    """Represents a single search result from DuckDuckGo."""
    def __init__(self, title: str, url: str, snippet: str):

        self.title = title
        self.url = url
        self.snippet = snippet
    def to_dict(self) -> Dict[str, str]:

        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }
async def search_bing(query: str, max_results: int = 10) -> List[SearchResult]:

    """
    Search using Bing and return real URLs with snippets.
    Enhanced with better headers to avoid anti-bot detection.
    """
    try:
        import aiohttp
        from bs4 import BeautifulSoup
        from urllib.parse import quote_plus

        search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Cache-Control": "max-age=0",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers, timeout=aiohttp.ClientTimeout(total=15),
                                   allow_redirects=True) as response:
                if response.status != 200:
                    logger.warning(f"Bing search failed: HTTP {response.status}")
                    return []
                html = await response.text()
                # Check for anti-bot detection
                if 'antibot' in html.lower() or 'verifycode' in html.lower() or 'captcha' in html.lower():
                    logger.warning("Bing anti-bot detection triggered")
                    return []
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                for result in soup.select('#b_results .b_algo'):
                    title_elem = result.select_one('h2 a')
                    snippet_elem = result.select_one('.b_caption p')
                    if title_elem and snippet_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True)
                        if url and title:
                            results.append(SearchResult(title, url, snippet))
                logger.info(f"Bing found {len(results)} results for: {query}")
                return results[:max_results]
    except Exception as e:
        logger.error(f"Bing search error: {e}")
        return []
async def search_baidu(query: str, max_results: int = 10) -> List[SearchResult]:

    """
    Search using Baidu and return real URLs with snippets.
    Enhanced with better headers to avoid anti-bot detection.
    Special handling for Baidu's redirect URLs.
    """
    try:
        import aiohttp
        from bs4 import BeautifulSoup
        from urllib.parse import quote_plus, urljoin, unquote

        search_url = f"https://www.baidu.com/s?wd={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Cache-Control": "max-age=0",
            "Referer": "https://www.baidu.com/",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers, timeout=aiohttp.ClientTimeout(total=15),
                                   allow_redirects=True) as response:
                if response.status != 200:
                    logger.warning(f"Baidu search failed: HTTP {response.status}")
                    return []
                html = await response.text()
                # Check for anti-bot detection
                if 'antibot' in html.lower() or 'verifycode' in html.lower() or 'captcha' in html.lower():
                    logger.warning("Baidu anti-bot detection triggered")
                    return []
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                for result in soup.select('#content_left .result, #content_left .result-op'):
                    title_elem = result.select_one('.t a, .c-title a')
                    snippet_elem = result.select_one('.c-abstract, .c-span-last')
                    if title_elem and snippet_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True)
                        # Skip empty or invalid URLs
                        if not url or url.startswith('#') or url.startswith('javascript:'):
                            continue
                        # Handle Baidu redirect URLs
                        # Baidu uses /link?url=XXX format which redirects to actual URL
                        if url.startswith('/link?url='):
                            # Convert to full URL - scraper will follow redirect
                            url = urljoin('https://www.baidu.com', url)
                        elif url.startswith('/'):
                            # Other relative URLs
                            url = urljoin('https://www.baidu.com', url)
                        # Skip Baidu internal pages (cache, snapshot, etc.)
                        # But KEEP baidu.com/link?url= as scraper will follow redirect
                        if any(skip in url.lower() for skip in [
                            'baidu.com/search/',
                            'baidu.com/cache/',
                            'baidu.com/snap/',
                            'baidustatic.com',
                        ]):
                            continue
                        if url and title:
                            results.append(SearchResult(title, url, snippet))
                logger.info(f"Baidu found {len(results)} results for: {query}")
                return results[:max_results]
    except Exception as e:
        logger.error(f"Baidu search error: {e}")
        return []
async def search_sogou(query: str, max_results: int = 10) -> List[SearchResult]:

    """
    Search using Sogou and return real URLs with snippets.
    Enhanced with better headers to avoid anti-bot detection.
    """
    try:
        import aiohttp
        from bs4 import BeautifulSoup
        from urllib.parse import quote_plus

        search_url = f"https://www.sogou.com/web?query={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Cache-Control": "max-age=0",
            "Referer": "https://www.sogou.com/",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers, timeout=aiohttp.ClientTimeout(total=15),
                                   allow_redirects=True) as response:
                if response.status != 200:
                    logger.warning(f"Sogou search failed: HTTP {response.status}")
                    return []
                html = await response.text()
                # Check for anti-bot detection
                if 'antibot' in html.lower() or 'verifycode' in html.lower() or 'captcha' in html.lower():
                    logger.warning("Sogou anti-bot detection triggered")
                    return []
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                for result in soup.select('.results .vrwrap, .results .txt-box'):
                    title_elem = result.select_one('.vr-title a, h3 a')
                    snippet_elem = result.select_one('.star-wiki, .attribute-centent, .txt-info')
                    if title_elem and snippet_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True)
                        if url and title:
                            results.append(SearchResult(title, url, snippet))
                logger.info(f"Sogou found {len(results)} results for: {query}")
                return results[:max_results]
    except Exception as e:
        logger.error(f"Sogou search error: {e}")
        return []
QUERY_EXPANSION_SYSTEM_PROMPT = """Generate 2-3 alternative search queries from the user's question.
Rules:
1. Output ONLY the alternative queries, one per line, nothing else
2. Each query should focus on different aspects or phrasings
3. Use concise keyword-style queries (not natural sentences)
4. Include synonyms and related terms for better coverage
5. Keep original proper nouns and dates
Examples:
- "查询2025年各省高考录取分数线情况" ->
2025年 高考 录取 分数线 各省
2025高考 各省 批次线 汇总
2025年 高考 各省 录取分数线 统计
- "上海2026年房价走势预测" ->
上海 2026年 房价 预测
上海 房价 走势 2026
2026年 上海 楼市 趋势"""

IRRELEVANT_DOMAIN_PATTERNS = [
    'kuaidi100.com', 'ickd.cn', 'sto.cn', 'sf-express.com', 'yundaex.com',
    'zto.com', 'yto.net.cn', 'deppon.com', 'jiayu.cn',
    'dict.', 'zidian.', 'baike.baidu.com/item/',
    'tianyancha.com', 'qichacha.com', 'qixin.com',
    'douyin.com', 'kuaishou.com', 'bilibili.com/video',
    'porn', 'xxx', 'vip', 'bet', 'casino',
]
IRRELEVANT_TITLE_PATTERNS = [
    '快递', '物流', '查询快递', '单号查询', '查快递',
    '百度百科', '汉语词典', '字典', '词典',
    '天眼查', '企查查', '启信宝',
    '抖音', '快手',
]
def _deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:

    """Deduplicate by URL domain+path."""
    seen = set()
    unique = []
    for r in results:
        key = r.url.rstrip('/').split('?')[0]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique
def _is_relevant_result(result: SearchResult, query: str) -> bool:

    """Filter out clearly irrelevant results."""
    url_lower = result.url.lower()
    title_lower = result.title.lower()
    snippet_lower = result.snippet.lower()
    for pattern in IRRELEVANT_DOMAIN_PATTERNS:
        if pattern in url_lower:
            logger.debug(f"Filtered irrelevant domain: {result.url} ({pattern})")
            return False
    for pattern in IRRELEVANT_TITLE_PATTERNS:
        if pattern in title_lower:
            logger.debug(f"Filtered irrelevant title: {result.title} ({pattern})")
            return False
    return True
def _score_result(result: SearchResult, query: str) -> float:

    """Score result relevance based on keyword matching."""
    score = 0.0
    query_terms = query.lower().split()
    title_lower = result.title.lower()
    snippet_lower = result.snippet.lower()
    for term in query_terms:
        if term in title_lower:
            score += 3.0
        if term in snippet_lower:
            score += 1.0
    url_path = result.url.lower()
    for term in query_terms:
        if term in url_path:
            score += 2.0
    if any(kw in title_lower for kw in ['官方', '政府', '教育部', 'gov.cn', 'edu.cn', '新华社']):
        score += 5.0
    return score
async def _search_with_engine(engine_name: str, search_fn, query: str, max_results: int) -> List[SearchResult]:

    """Wrapper for safe search execution."""
    try:
        results = await search_fn(query, max_results)
        logger.info(f"[{engine_name}] {len(results)} results for '{query}'")
        return results
    except Exception as e:
        logger.warning(f"[{engine_name}] search error: {e}")
        return []
async def _expand_query(query: str) -> List[str]:

    """Generate keyword variations to improve search coverage."""
    variations = [query]
    no_space_query = query.replace(' ', '')
    if no_space_query != query:
        variations.append(no_space_query)
    try:
        response = llm_client.chat.completions.create(
            model=settings.GEMINI_MODEL,
            messages=[
                {"role": "system", "content": QUERY_EXPANSION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        expanded = response.choices[0].message.content.strip()
        for line in expanded.split('\n'):
            line = line.strip()
            if line and len(line) >= 4 and line not in variations:
                variations.append(line)
        logger.info(f"Query expanded: {len(variations)} variations")
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
    return variations[:4]
async def perform_web_search(query: str, max_results: int = 10) -> List[SearchResult]:

    """
    Enhanced web search with:
    - Parallel engine queries (Bing + Baidu + Sogou simultaneously)
    - Keyword expansion for better coverage
    - Smart result deduplication and relevance filtering
    - Relevance scoring and ranking
    """
    search_query = await preprocess_query(query)
    logger.info(f"Search query: '{query}' -> optimized: '{search_query}'")
    expanded_queries = await _expand_query(search_query)
    all_tasks = []
    for q in expanded_queries:
        all_tasks.extend([
            _search_with_engine("Bing", search_bing, q, max_results),
            _search_with_engine("Baidu", search_baidu, q, max_results),
            _search_with_engine("Sogou", search_sogou, q, max_results),
        ])
    all_results_lists = await asyncio.gather(*all_tasks, return_exceptions=True)
    raw_results = []
    engine_results_count = {"Bing": 0, "Baidu": 0, "Sogou": 0}
    engine_idx = 0
    for q_idx in range(len(expanded_queries)):
        for eng_name in ["Bing", "Baidu", "Sogou"]:
            result = all_results_lists[engine_idx]
            if isinstance(result, list):
                raw_results.extend(result)
                engine_results_count[eng_name] += len(result)
            engine_idx += 1
    logger.info(
        f"Raw results: Bing={engine_results_count['Bing']}, "
        f"Baidu={engine_results_count['Baidu']}, Sogou={engine_results_count['Sogou']}"
    )
    if not raw_results:
        logger.warning(f"All engines failed for '{search_query}', trying original query...")
        if search_query != query:
            return await perform_web_search(query, max_results)
        return []
    unique_results = _deduplicate_results(raw_results)
    logger.info(f"After dedup: {len(unique_results)} unique (from {len(raw_results)} raw)")
    relevant_results = [r for r in unique_results if _is_relevant_result(r, search_query)]
    logger.info(
        f"After relevance filter: {len(relevant_results)} (removed {len(unique_results) - len(relevant_results)} irrelevant)")
    if not relevant_results:
        logger.warning(f"No relevant results after filtering for '{search_query}'")
        if len(unique_results) > 0:
            relevant_results = unique_results[:max_results]
    scored = [(r, _score_result(r, search_query)) for r in relevant_results]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_results = [r for r, s in scored[:max_results * 2]]
    logger.info(f"Top {min(max_results, len(top_results))} results with highest relevance scores")
    return top_results[:max_results]
    return results
async def deep_scrape_urls(urls: List[str], scraper: WebScraper) -> List[ScrapedPage]:

    """
    Deep scrape URLs to get full content.
    Implements retry logic and content validation.
    """
    if not urls:
        return []
    logger.info(f"Deep scraping {len(urls)} URLs...")
    scraped_pages = await scraper.scrape_urls(urls)
    successful = [p for p in scraped_pages if not p.error and len(p.content) > 100]
    failed = [p for p in scraped_pages if p.error or len(p.content) <= 100]
    if failed:
        logger.warning(f"Failed to scrape {len(failed)} URLs:")
        for page in failed:
            logger.warning(f"  - {page.url}: {page.error or 'Content too short'}")
    logger.info(f"Successfully scraped {len(successful)}/{len(urls)} URLs")
    return successful
def build_context_from_pages(pages: List[ScrapedPage]) -> str:

    """
    Build comprehensive context from scraped pages.
    Optimizes content for LLM consumption.
    """
    context_parts = []
    for i, page in enumerate(pages):
        content = page.markdown or page.content
        # Increased from 5000 to 15000 chars per page for better context
        if len(content) > 30000:
            content = content[:30000] + "\n...(content truncated)"
        context_parts.append(
            f"[Source {i + 1}: {page.title}]({page.url})\n"
            f"{content}\n"
        )
    context = "\n\n" + "=" * 80 + "\n\n".join(context_parts)
    logger.info(f"Built context: {len(context)} chars from {len(pages)} sources")
    return context
@router.post("/search/results", response_model=SearchResultsResponse)
async def search_results(request: SearchRequest):

    """
    Firecrawl-style step 1: Search the web and return result list (URLs + titles + descriptions).
    Does NOT scrape pages yet — user selects which pages to scrape.
    """
    try:
        search_results_list = await perform_web_search(request.query, request.max_pages)
        if not search_results_list:
            return SearchResultsResponse(query=request.query, results=[], total=0)
        items = []
        for i, r in enumerate(search_results_list):
            items.append(SearchResultItem(
                url=r.url,
                title=r.title or "",
                description=r.snippet or "",
                position=i + 1,
            ))
        return SearchResultsResponse(
            query=request.query,
            results=items,
            total=len(items),
        )
    except Exception as e:
        logger.error(f"Search results API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
SCRAPE_PAGE_SYSTEM_PROMPT = """You are an expert page analyst. Analyze the scraped webpage content thoroughly.
RULES:
1. Start with a brief summary of what this page is about (2-3 sentences)
2. Extract key facts, data points, statistics, and structured information
3. Identify main topics and themes
4. Use markdown tables for any structured data
5. Cite specific sections of the content
6. Keep analysis focused and actionable
7. Respond in the SAME LANGUAGE as the page content"""

@router.post("/search/scrape-page", response_model=ScrapePageResponse)
async def scrape_page(request: ScrapePageRequest):

    """
    Firecrawl-style step 2: Scrape and deeply analyze a single URL.
    User clicks "Scrape Page" on a search result to trigger this.
    """
    try:
        scraper = WebScraper(
            max_concurrent=1,
            timeout=60,
            max_retries=2,
        )
        page = await scraper.scrape_url(request.url)
        if page.error:
            return ScrapePageResponse(
                url=request.url,
                title="",
                content="",
                markdown="",
                analysis="",
                metadata={},
                status_code=0,
                error=page.error,
            )
        content = page.markdown or page.content or ""
        analysis = ""
        if content.strip():
            try:
                context = content[:30000]
                response = llm_client.chat.completions.create(
                    model=settings.GEMINI_MODEL,
                    messages=[
                        {"role": "system", "content": SCRAPE_PAGE_SYSTEM_PROMPT},
                        {"role": "user",
                         "content": f"Page URL: {request.url}\nPage Title: {page.title}\n\nContent:\n{context}\n\nAnalyze this page thoroughly."},
                    ],
                    temperature=0.5,
                    max_tokens=4096,
                )
                analysis = response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"LLM analysis failed for {request.url}: {e}")
                analysis = f"*Page scraped but analysis unavailable: {e}*"
        return ScrapePageResponse(
            url=page.url or request.url,
            title=page.title or "",
            content=page.content or "",
            markdown=page.markdown or "",
            analysis=analysis,
            metadata=page.metadata or {},
            status_code=page.status_code or 200,
            error=None,
        )
    except Exception as e:
        logger.error(f"Scrape page API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/search")
async def search(request: SearchRequest):

    """
    Search the web and generate an AI answer based on real content.
    Workflow:
    1. Use search engines to find relevant URLs
    2. Deep scrape each URL to extract actual content
    3. Build comprehensive context from all scraped content
    4. Use LLM to generate detailed answer
    5. Return answer with sources
    """
    try:
        search_results = await perform_web_search(request.query, request.max_pages)
        if not search_results:
            return {
                "answer": "Unable to find relevant web content for this query. Please try rephrasing your question.",
                "sources": [],
                "scraped_pages": [],
            }
        urls = [r.url for r in search_results]
        scraper = WebScraper(
            max_concurrent=3,
            timeout=30,
            max_retries=2,
        )
        scraped_pages = await deep_scrape_urls(urls, scraper)
        if not scraped_pages:
            return {
                "answer": "Found search results but unable to scrape the web pages. This may be due to access restrictions. Please try a different query.",
                "sources": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in search_results],
                "scraped_pages": [],
            }
        context = build_context_from_pages(scraped_pages)
        user_prompt = (
            f"Query: {request.query}\n\n"
            f"Scraped content from web:\n{context}\n\n"
            f"Answer the query using the content above. Be direct and thorough."
        )
        response = llm_client.chat.completions.create(
            model=settings.GEMINI_MODEL,
            messages=[
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=16384,
        )
        answer = response.choices[0].message.content
        sources = [
            {
                "title": page.title,
                "url": page.url,
                "description": page.metadata.get("description", ""),
                "word_count": page.metadata.get("word_count", 0),
            }
            for page in scraped_pages
        ]
        # Don't save search history to chat - search is independent from chat
        return {
            "answer": answer,
            "sources": sources,
            "scraped_pages": [p.to_dict() for p in scraped_pages],
        }
    except Exception as e:
        logger.error(f"Search API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/search/stream")
async def search_stream(request: SearchRequest):

    """
    Stream search results with real-time progress.
    Automatically detects if input contains a URL and scrapes it directly.
    Supports data analysis and chart generation modes.
    """
    try:
        # Don't save search history to chat - search is independent from chat
        # Detect user intent
        intent = detect_user_intent(request.query.strip())
        logger.info(f"Detected intent: {intent}")
        # Check if input contains a URL (even if mixed with other text)
        extracted_url = extract_url_from_text(request.query.strip())
        has_url = extracted_url is not None
        async def generate():

            if has_url:
                # Direct URL scraping mode - inspired by Firecrawl's scrape_url
                url = normalize_url(extracted_url)
                logger.info(f"Detected URL in input, scraping directly: {url}")
                yield json.dumps({"type": "status", "data": f"Scraping {url}..."}, ensure_ascii=False) + "\n"
                scraper = WebScraper(
                    max_concurrent=1,
                    timeout=30,
                    max_retries=3,
                )
                scraped_page = await scraper.scrape_url(url)
                if scraped_page.error or len(scraped_page.content) < 100:
                    error_msg = scraped_page.error or 'Content too short'
                    # Check if it's an anti-bot protection issue
                    if any(keyword in error_msg.lower() for keyword in
                           ['anti-bot', 'verification', 'captcha', 'cloudflare']):
                        error_msg = (
                            "\u26a0\ufe0f \u8be5\u7f51\u7ad9\u542f\u7528\u4e86\u53cd\u722c\u866b\u4fdd\u62a4\uff0c\u65e0\u6cd5\u76f4\u63a5\u6293\u53d6\u5185\u5bb9\u3002\n\n"
                            f"\u7f51\u7ad9: {url}\n"
                            "\u539f\u56e0: \u7f51\u7ad9\u4f7f\u7528\u4e86\u9a8c\u8bc1\u7801/\u673a\u5668\u4eba\u9a8c\u8bc1\u673a\u5236\n\n"
                            "\u5efa\u8bae:\n"
                            "1. \u5c1d\u8bd5\u624b\u52a8\u8bbf\u95ee\u8be5\u7f51\u7ad9\u67e5\u770b\u5185\u5bb9\n"
                            "2. \u4f7f\u7528\u641c\u7d22\u5f15\u64ce\u641c\u7d22\u76f8\u5173\u4fe1\u606f\n"
                            "3. \u5c1d\u8bd5\u5176\u4ed6\u7c7b\u4f3c\u7f51\u7ad9"
                        )
                    yield json.dumps({
                        "type": "error",
                        "data": error_msg,
                    }, ensure_ascii=False) + "\n"
                    return
                # Validate that scraped content matches the target URL
                target_domain = urlparse(url).netloc
                scraped_domain = urlparse(scraped_page.url).netloc
                if target_domain != scraped_domain:
                    logger.warning(f"Content domain mismatch: requested {target_domain}, got {scraped_domain}")
                    error_msg = (
                        f"\u26a0\ufe0f \u7f51\u7ad9\u5185\u5bb9\u65e0\u6cd5\u8bbf\u95ee\u3002\n\n"
                        f"\u8bf7\u6c42\u7f51\u7ad9: {url}\n"
                        f"\u5b9e\u9645\u6293\u53d6: {scraped_page.url}\n\n"
                        f"\u53ef\u80fd\u539f\u56e0:\n"
                        "1. \u7f51\u7ad9\u4f7f\u7528\u4e86\u53cd\u722c\u866b\u4fdd\u62a4\n"
                        "2. \u7f51\u7ad9\u9700\u8981 JavaScript \u6e32\u67d3\u5185\u5bb9\n"
                        "3. \u7f51\u7ad9\u88ab\u91cd\u5b9a\u5411\u5230\u5176\u4ed6\u9875\u9762\n\n"
                        "\u5efa\u8bae:\n"
                        "1. \u624b\u52a8\u8bbf\u95ee\u8be5\u7f51\u7ad9\u786e\u8ba4\u5185\u5bb9\n"
                        "2. \u4f7f\u7528\u641c\u7d22\u5f15\u64ce\u641c\u7d22\u76f8\u5173\u4fe1\u606f"
                    )
                    yield json.dumps({
                        "type": "error",
                        "data": error_msg,
                    }, ensure_ascii=False) + "\n"
                    return
                # Check if content is relevant to the target URL
                content = scraped_page.markdown or scraped_page.content
                target_keywords = set(url.lower().split('/'))
                content_lower = content.lower()
                # Check if content contains relevant keywords from URL
                relevant_keywords = [kw for kw in target_keywords if len(kw) > 3 and kw.isalnum()]
                keyword_matches = sum(1 for kw in relevant_keywords if kw in content_lower)
                if len(relevant_keywords) > 0 and keyword_matches < len(relevant_keywords) * 0.3:
                    logger.warning(
                        f"Content relevance low: {keyword_matches}/{len(relevant_keywords)} keywords matched")
                    # Don't fail, but log warning - content might still be useful
                yield json.dumps({
                    "type": "scraped",
                    "data": {
                        "success_count": 1,
                        "pages": [scraped_page.to_dict()],
                    }
                }, ensure_ascii=False) + "\n"
                # Build context from scraped page
                content = scraped_page.markdown or scraped_page.content
                # Increased from 8000 to 20000 chars for direct URL scraping
                if len(content) > 20000:
                    content = content[:20000] + "\n...(content truncated)"
                context = f"[Source: {scraped_page.title}]({scraped_page.url})\n{content}\n"
                sources = [{
                    "title": scraped_page.title,
                    "url": scraped_page.url,
                }]
                yield json.dumps({
                    "type": "sources",
                    "data": sources,
                }, ensure_ascii=False) + "\n"
                # If user wants data analysis and chart generation
                if intent["needs_analysis"] or intent["needs_chart"]:
                    yield json.dumps({"type": "status", "data": "Extracting data from content..."},
                                     ensure_ascii=False) + "\n"
                    # Step 1: Use LLM to extract structured data
                    data_extraction_prompt = (
                        f"URL: {url}\n\n"
                        f"Scraped Content:\n"
                        f"{context}\n\n"
                        f"User Request: {request.query}\n\n"
                        f"Please extract structured data points from this content for analysis and visualization.\n"
                        f"Return ONLY a JSON array of objects with 'x' and 'y' keys, like:\n"
                        f'[{{"x": "2021", "y": 100}}, {{"x": "2022", "y": 150}}]\n'
                        f"The 'x' should be labels (dates, categories, etc.) and 'y' should be numeric values.\n"
                        f"Return ONLY the JSON array, no other text."
                    )
                    try:
                        extraction_response = llm_client.chat.completions.create(
                            model=settings.GEMINI_MODEL,
                            messages=[
                                {"role": "system",
                                 "content": "You are a data extraction expert. Return ONLY valid JSON."},
                                {"role": "user", "content": data_extraction_prompt},
                            ],
                            temperature=0.3,
                            max_tokens=4096,
                        )
                        extracted_text = extraction_response.choices[0].message.content.strip()
                        logger.info(f"LLM extraction response: {extracted_text[:200]}...")
                        # Parse JSON from response - robust extraction
                        try:
                            # Try to extract JSON from markdown code blocks
                            json_str = extracted_text
                            # Method 1: Extract from ```json ... ``` blocks
                            if '```' in extracted_text:
                                # Try to find json code block
                                json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', extracted_text)
                                if json_match:
                                    json_str = json_match.group(1)
                                else:
                                    # Fallback: just split on ```
                                    parts = extracted_text.split('```')
                                    if len(parts) >= 2:
                                        json_str = parts[1]
                                        if json_str.startswith('json'):
                                            json_str = json_str[4:]
                            # Method 2: Try to find JSON array in text
                            if not json_str.strip().startswith('['):
                                array_match = re.search(r'\[[\s\S]*\]', json_str)
                                if array_match:
                                    json_str = array_match.group(0)
                            # Clean up JSON string
                            json_str = json_str.strip()
                            # Try to parse JSON
                            try:
                                data_points = json.loads(json_str)
                            except json.JSONDecodeError as json_err:
                                logger.warning(f"Standard JSON parse failed, trying relaxed parsing: {json_err}")
                                # Try to fix common JSON issues
                                # Remove trailing commas
                                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                                # Replace single quotes with double quotes
                                json_str = json_str.replace("'", '"')
                                # Remove comments
                                json_str = re.sub(r'//.*?\n', '', json_str)
                                data_points = json.loads(json_str)
                            # Validate data structure
                            if not isinstance(data_points, list) or len(data_points) == 0:
                                raise ValueError("Invalid data format: expected non-empty array")
                            # Validate each data point has x and y
                            for i, point in enumerate(data_points):
                                if not isinstance(point, dict):
                                    raise ValueError(f"Data point {i} is not an object")
                                if 'x' not in point or 'y' not in point:
                                    raise ValueError(f"Data point {i} missing 'x' or 'y' key")
                                # Ensure y is numeric
                                try:
                                    point['y'] = float(point['y'])
                                except (ValueError, TypeError):
                                    raise ValueError(f"Data point {i} 'y' value is not numeric: {point['y']}")
                            logger.info(f"Successfully extracted {len(data_points)} data points")
                            yield json.dumps({
                                "type": "data_extracted",
                                "data": {
                                    "count": len(data_points),
                                    "points": data_points,
                                }
                            }, ensure_ascii=False) + "\n"
                        except (json.JSONDecodeError, ValueError) as parse_error:
                            logger.error(f"Failed to parse extracted data: {parse_error}")
                            logger.error(f"Raw LLM response: {extracted_text[:500]}")
                            yield json.dumps({
                                "type": "error",
                                "data": f"Failed to extract structured data from content: {str(parse_error)}",
                            }, ensure_ascii=False) + "\n"
                            return
                    except Exception as extraction_error:
                        logger.error(f"Data extraction failed: {extraction_error}")
                        yield json.dumps({
                            "type": "error",
                            "data": f"Failed to extract data: {str(extraction_error)}",
                        }, ensure_ascii=False) + "\n"
                        return
                    # Step 2: Generate chart if requested
                    if intent["needs_chart"]:
                        yield json.dumps({"type": "status", "data": "Generating chart..."}, ensure_ascii=False) + "\n"
                        # Determine chart type from query
                        chart_type = "line"  # default
                        query_lower = request.query.lower()
                        if '\u67f1\u72b6\u56fe' in query_lower or 'bar chart' in query_lower:
                            chart_type = "bar"
                        elif '\u997c\u56fe' in query_lower or 'pie chart' in query_lower:
                            chart_type = "pie"
                        elif '\u6563\u70b9\u56fe' in query_lower or 'scatter' in query_lower:
                            chart_type = "scatter"
                        # Generate chart
                        chart_title = f"{scraped_page.title} - Data Analysis"
                        chart_image = generate_chart_from_data(
                            data_points=data_points,
                            chart_type=chart_type,
                            title=chart_title,
                            x_label="Category",
                            y_label="Value",
                        )
                        if chart_image:
                            yield json.dumps({
                                "type": "chart",
                                "data": {
                                    "image": chart_image,
                                    "chart_type": chart_type,
                                    "title": chart_title,
                                }
                            }, ensure_ascii=False) + "\n"
                        else:
                            yield json.dumps({
                                "type": "warning",
                                "data": "Failed to generate chart",
                            }, ensure_ascii=False) + "\n"
                        # Step 3: Generate analysis text
                        yield json.dumps({"type": "status", "data": "Analyzing data..."}, ensure_ascii=False) + "\n"
                        analysis_prompt = (
                            f"URL: {url}\n\n"
                            f"Scraped Content:\n"
                            f"{context}\n\n"
                            f"Extracted Data Points:\n"
                            f"{json.dumps(data_points, ensure_ascii=False)}\n\n"
                            f"User Request: {request.query}\n\n"
                            f"Please provide a comprehensive analysis of this data. Include:\n"
                            f"1. Key trends and patterns\n"
                            f"2. Notable insights\n"
                            f"3. Summary of findings\n"
                            f"Base your analysis ONLY on the provided data and content."
                        )
                        user_prompt = analysis_prompt
                    else:
                        # Just analysis, no chart
                        yield json.dumps({"type": "status", "data": "Analyzing data..."}, ensure_ascii=False) + "\n"
                        analysis_prompt = (
                            f"URL: {url}\n\n"
                            f"Scraped Content:\n"
                            f"{context}\n\n"
                            f"User Request: {request.query}\n\n"
                            f"Please provide a comprehensive analysis based on the scraped content."
                        )
                        user_prompt = analysis_prompt
                else:
                    # Regular summary mode
                    yield json.dumps({"type": "status", "data": "Analyzing page content..."}, ensure_ascii=False) + "\n"
                    user_prompt = (
                        f"URL: {url}\n\n"
                        f"Scraped Content:\n"
                        f"{context}\n\n"
                        f"Please provide a comprehensive summary and analysis of this webpage content."
                    )
            else:
                # Regular search mode
                yield json.dumps({"type": "status", "data": "Searching the web..."}, ensure_ascii=False) + "\n"
                search_results = await perform_web_search(request.query, request.max_pages)
                if not search_results:
                    yield json.dumps({
                        "type": "error",
                        "data": "Unable to find relevant web content.",
                    }, ensure_ascii=False) + "\n"
                    return
                yield json.dumps({
                    "type": "search_results",
                    "data": {
                        "count": len(search_results),
                        "results": [r.to_dict() for r in search_results],
                    }
                }, ensure_ascii=False) + "\n"
                urls = [r.url for r in search_results]
                yield json.dumps({
                    "type": "status",
                    "data": f"Scraping {len(urls)} web pages...",
                }, ensure_ascii=False) + "\n"
                scraper = WebScraper(
                    max_concurrent=3,
                    timeout=30,
                    max_retries=2,
                )
                scraped_pages = await deep_scrape_urls(urls, scraper)
                if not scraped_pages:
                    yield json.dumps({
                        "type": "error",
                        "data": "Found search results but unable to scrape web pages.",
                    }, ensure_ascii=False) + "\n"
                    return
                yield json.dumps({
                    "type": "scraped",
                    "data": {
                        "success_count": len(scraped_pages),
                        "pages": [p.to_dict() for p in scraped_pages],
                    }
                }, ensure_ascii=False) + "\n"
                context = build_context_from_pages(scraped_pages)
                sources = [
                    {
                        "title": page.title,
                        "url": page.url,
                    }
                    for page in scraped_pages
                ]
                yield json.dumps({
                    "type": "sources",
                    "data": sources,
                }, ensure_ascii=False) + "\n"
                # If user wants data analysis and chart generation
                if intent["needs_analysis"] or intent["needs_chart"]:
                    yield json.dumps({"type": "status", "data": "Extracting data from content..."},
                                     ensure_ascii=False) + "\n"
                    # Step 1: Use LLM to extract structured data
                    data_extraction_prompt = (
                        f"Query: {request.query}\n\n"
                        f"Scraped Content:\n"
                        f"{context}\n\n"
                        f"Extract structured data points for analysis. Return ONLY JSON array:\n"
                        f'[{{"x": "2021", "y": 100}}, {{"x": "2022", "y": 150}}]\n'
                        f"'x' = labels, 'y' = numeric values.\n"
                        f"Return ONLY the JSON array, no other text."
                    )
                    try:
                        extraction_response = llm_client.chat.completions.create(
                            model=settings.GEMINI_MODEL,
                            messages=[
                                {"role": "system",
                                 "content": "You are a data extraction expert. Return ONLY valid JSON."},
                                {"role": "user", "content": data_extraction_prompt},
                            ],
                            temperature=0.3,
                            max_tokens=4096,
                        )
                        extracted_text = extraction_response.choices[0].message.content.strip()
                        # Parse JSON from response
                        import ast

                        try:
                            # Try to extract JSON from markdown code blocks
                            if '```' in extracted_text:
                                json_str = extracted_text.split('```')[1]
                                if json_str.startswith('json'):
                                    json_str = json_str[4:]
                            else:
                                json_str = extracted_text
                            data_points = json.loads(json_str)
                            if not isinstance(data_points, list) or len(data_points) == 0:
                                raise ValueError("Invalid data format")
                            yield json.dumps({
                                "type": "data_extracted",
                                "data": {
                                    "count": len(data_points),
                                    "points": data_points,
                                }
                            }, ensure_ascii=False) + "\n"
                        except (json.JSONDecodeError, ValueError) as parse_error:
                            logger.error(f"Failed to parse extracted data: {parse_error}")
                            yield json.dumps({
                                "type": "error",
                                "data": f"Failed to extract structured data from content: {str(parse_error)}",
                            }, ensure_ascii=False) + "\n"
                            return
                    except Exception as extraction_error:
                        logger.error(f"Data extraction failed: {extraction_error}")
                        yield json.dumps({
                            "type": "error",
                            "data": f"Failed to extract data: {str(extraction_error)}",
                        }, ensure_ascii=False) + "\n"
                        return
                    # Step 2: Generate chart if requested
                    if intent["needs_chart"]:
                        yield json.dumps({"type": "status", "data": "Generating chart..."}, ensure_ascii=False) + "\n"
                        # Determine chart type from query
                        chart_type = "line"  # default
                        query_lower = request.query.lower()
                        if '\u67f1\u72b6\u56fe' in query_lower or 'bar chart' in query_lower:
                            chart_type = "bar"
                        elif '\u997c\u56fe' in query_lower or 'pie chart' in query_lower:
                            chart_type = "pie"
                        elif '\u6563\u70b9\u56fe' in query_lower or 'scatter' in query_lower:
                            chart_type = "scatter"
                        # Generate chart
                        chart_title = f"Data Analysis - {request.query[:50]}"
                        chart_image = generate_chart_from_data(
                            data_points=data_points,
                            chart_type=chart_type,
                            title=chart_title,
                            x_label="Category",
                            y_label="Value",
                        )
                        if chart_image:
                            yield json.dumps({
                                "type": "chart",
                                "data": {
                                    "image": chart_image,
                                    "chart_type": chart_type,
                                    "title": chart_title,
                                }
                            }, ensure_ascii=False) + "\n"
                        else:
                            yield json.dumps({
                                "type": "warning",
                                "data": "Failed to generate chart",
                            }, ensure_ascii=False) + "\n"
                        # Step 3: Generate analysis text
                        yield json.dumps({"type": "status", "data": "Analyzing data..."}, ensure_ascii=False) + "\n"
                        analysis_prompt = (
                            f"Query: {request.query}\n\n"
                            f"Scraped Content:\n"
                            f"{context}\n\n"
                            f"Data Points:\n"
                            f"{json.dumps(data_points, ensure_ascii=False)}\n\n"
                            f"Please provide a comprehensive analysis of this data. Include:\n"
                            f"1. Key trends and patterns\n"
                            f"2. Notable insights\n"
                            f"3. Summary of findings\n"
                            f"Base your analysis ONLY on the provided data and content."
                        )
                        user_prompt = analysis_prompt
                    else:
                        # Just analysis, no chart
                        yield json.dumps({"type": "status", "data": "Analyzing data..."}, ensure_ascii=False) + "\n"
                        analysis_prompt = (
                            f"Query: {request.query}\n\n"
                            f"Scraped Content:\n"
                            f"{context}\n\n"
                            f"User Request: {request.query}\n\n"
                            f"Please provide a comprehensive analysis based on the scraped content."
                        )
                        user_prompt = analysis_prompt
                else:
                    # Regular summary mode
                    yield json.dumps({"type": "status", "data": "Generating comprehensive answer..."},
                                     ensure_ascii=False) + "\n"
                    user_prompt = (
                        f"Query: {request.query}\n\n"
                        f"Scraped content from web:\n{context}\n\n"
                        f"Answer the query using the content above. Be direct and thorough."
                    )
            full_content = ""
            try:
                stream = llm_client.chat.completions.create(
                    model=settings.GEMINI_MODEL,
                    messages=[
                        {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=16384,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_content += content
                        yield json.dumps({"type": "chunk", "data": content}, ensure_ascii=False) + "\n"
            except Exception as llm_error:
                logger.error(f"LLM error: {llm_error}")
                yield json.dumps({"type": "error", "data": str(llm_error)}, ensure_ascii=False) + "\n"
                return
            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    except Exception as e:
        logger.error(f"Search stream API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/scrape")
async def scrape_url(request: SearchRequest):

    """
    Scrape a specific URL and return cleaned content.
    """
    try:
        if not request.include_urls:
            raise HTTPException(status_code=400, detail="URLs must be provided")
        scraper = WebScraper(
            max_concurrent=1,
            timeout=30,
            max_retries=2,
        )
        url = request.include_urls[0]
        page = await scraper.scrape_url(url)
        if page.error:
            raise HTTPException(status_code=400, detail=page.error)
        return {
            "url": page.url,
            "title": page.title,
            "content": page.content,
            "markdown": page.markdown,
            "metadata": page.metadata,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scrape API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
