import React, { useState } from 'react';
import { Input, Button, Card, Tag, Space, Empty, Spin, message, Collapse, Tooltip, Result, Badge, Divider } from 'antd';
import {
  SearchOutlined,
  GlobalOutlined,
  LinkOutlined,
  LoadingOutlined,
  FileTextOutlined,
  ExportOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  ClearOutlined,
  CaretRightOutlined,
  EyeOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';

const { Panel } = Collapse;

const API_BASE_URL = 'http://localhost:8300/api';

interface SearchResultItem {
  url: string;
  title: string;
  description: string;
  position: number;
}

interface ScrapedPageAnalysis {
  url: string;
  title: string;
  content: string;
  markdown: string;
  analysis: string;
  metadata: Record<string, any>;
  status_code: number;
  error?: string;
}

const QUICK_STARTERS = [
  { label: 'AI Research Papers 2025', query: '2025骞存渶鏂癆I鐮旂┒璁烘枃' },
  { label: 'Global Market Trends', query: '2025鍏ㄧ悆鑲″競璧板娍鍒嗘瀽' },
  { label: 'Latest Tech News', query: '鏈€鏂扮鎶€鏂伴椈' },
  { label: 'Python Web Scraping', query: 'Python web scraping best practices 2025' },
];

const Search: React.FC = () => {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [scrapedPages, setScrapedPages] = useState<Map<string, ScrapedPageAnalysis>>(new Map());
  const [scrapingUrls, setScrapingUrls] = useState<Set<string>>(new Set());
  const [scrapeErrors, setScrapeErrors] = useState<Map<string, string>>(new Map());
  const [expandedPanels, setExpandedPanels] = useState<Set<string>>(new Set());

  const handleQuickStarter = (starterQuery: string) => {
    setQuery(starterQuery);
    setTimeout(() => handleSearch(starterQuery), 100);
  };

  const handleSearch = async (searchQuery?: string) => {
    const q = searchQuery || query;
    if (!q.trim()) {
      message.warning('Please enter a search query');
      return;
    }

    setIsSearching(true);
    setSearchResults([]);
    setScrapedPages(new Map());
    setScrapeErrors(new Map());
    setExpandedPanels(new Set());

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      const response = await fetch(`${API_BASE_URL}/search/search/results`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          max_pages: 10,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      setSearchResults(data.results || []);

      if (!data.results || data.results.length === 0) {
        message.info('No results found. Try a different query.');
      } else {
        message.success(`Found ${data.results.length} results`);
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        message.error('Search timed out. Please try again.');
      } else {
        message.error('Search failed. Check if backend is running.');
      }
    } finally {
      setIsSearching(false);
    }
  };

  const handleScrapePage = async (item: SearchResultItem) => {
    const url = item.url;

    if (scrapedPages.has(url) || scrapingUrls.has(url)) {
      return;
    }

    setScrapingUrls(prev => new Set(prev).add(url));
    setScrapeErrors(prev => {
      const next = new Map(prev);
      next.delete(url);
      return next;
    });

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);

      const response = await fetch(`${API_BASE_URL}/search/search/scrape-page`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data: ScrapedPageAnalysis = await response.json();

      if (data.error && !data.content && !data.markdown) {
        setScrapeErrors(prev => {
          const next = new Map(prev);
          next.set(url, data.error || 'Unknown error');
          return next;
        });
        message.warning(`Scrape failed for ${getDomain(url)}: ${data.error}`);
        return;
      }

      setScrapedPages(prev => {
        const next = new Map(prev);
        next.set(url, data);
        return next;
      });

      setExpandedPanels(prev => new Set(prev).add(url));

      if (data.analysis) {
        message.success(`Successfully scraped and analyzed ${getDomain(url)}`);
      } else {
        message.info(`Scraped ${getDomain(url)} (no AI analysis available)`);
      }
    } catch (err: any) {
      const errorMsg = err.name === 'AbortError'
        ? 'Scrape timed out (120s)'
        : (err.message || 'Unknown error');

      setScrapeErrors(prev => {
        const next = new Map(prev);
        next.set(url, errorMsg);
        return next;
      });
      message.error(`Failed to scrape ${getDomain(url)}: ${errorMsg}`);
    } finally {
      setScrapingUrls(prev => {
        const next = new Set(prev);
        next.delete(url);
        return next;
      });
    }
  };

  const getDomain = (url: string) => {
    try {
      return new URL(url).hostname;
    } catch {
      return url;
    }
  };

  const togglePanel = (url: string) => {
    setExpandedPanels(prev => {
      const next = new Set(prev);
      if (next.has(url)) {
        next.delete(url);
      } else {
        next.add(url);
      }
      return next;
    });
  };

  const scrapedCount = scrapedPages.size;
  const errorCount = scrapeErrors.size;

  return (
    <div className="search-page" style={{ padding: '24px 24px 48px', maxWidth: 1100, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: 40 }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 10,
          marginBottom: 8,
        }}>
          <ThunderboltOutlined style={{ fontSize: 32, color: '#1677ff' }} />
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, color: '#1a1a2e' }}>
            Web Scraper
          </h1>
        </div>
        <p style={{ color: '#8c8c8c', fontSize: 15, margin: '8px 0 0' }}>
          Search the web, scrape pages, and get AI-powered analysis
        </p>
      </div>

      {/* Search Card */}
      <Card
        style={{
          marginBottom: 24,
          borderRadius: 16,
          boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
          border: '1px solid #e8e8e8',
        }}
      >
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <Input
            size="large"
            prefix={<SearchOutlined style={{ color: '#1677ff', fontSize: 18 }} />}
            placeholder="Enter URL or search query to scrape..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            onPressEnter={() => handleSearch()}
            disabled={isSearching}
            style={{
              flex: 1,
              borderRadius: 10,
              height: 48,
              fontSize: 15,
              border: '1px solid #d9d9d9',
            }}
            suffix={
              query && (
                <ClearOutlined
                  onClick={() => setQuery('')}
                  style={{ color: '#bfbfbf', cursor: 'pointer' }}
                />
              )
            }
          />
          <Button
            type="primary"
            size="large"
            icon={isSearching ? <LoadingOutlined /> : <SearchOutlined />}
            onClick={() => handleSearch()}
            loading={isSearching}
            disabled={!query.trim()}
            style={{
              borderRadius: 10,
              height: 48,
              minWidth: 160,
              fontSize: 15,
              fontWeight: 600,
              background: !query.trim() ? undefined : 'linear-gradient(135deg, #1677ff 0%, #0958d9 100%)',
            }}
          >
            Start Scraping
          </Button>
        </div>

        {/* Quick starters */}
        {!isSearching && searchResults.length === 0 && (
          <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ color: '#8c8c8c', fontSize: 13, lineHeight: '28px' }}>Try:</span>
            {QUICK_STARTERS.map(s => (
              <Tag
                key={s.query}
                style={{ cursor: 'pointer', borderRadius: 6, padding: '2px 12px', fontSize: 12 }}
                onClick={() => handleQuickStarter(s.query)}
              >
                {s.label}
              </Tag>
            ))}
          </div>
        )}
      </Card>

      {/* Loading state */}
      {isSearching && (
        <Card style={{ marginBottom: 24, borderRadius: 12, textAlign: 'center', padding: '40px 0' }}>
          <Spin indicator={<LoadingOutlined style={{ fontSize: 32, color: '#1677ff' }} spin />} />
          <p style={{ marginTop: 16, color: '#8c8c8c', fontSize: 15 }}>Searching the web for "{query}"...</p>
        </Card>
      )}

      {/* Stats bar */}
      {!isSearching && searchResults.length > 0 && (
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 20,
          padding: '12px 0',
        }}>
          <Space size="large">
            <span style={{ fontSize: 15, fontWeight: 600, color: '#1a1a2e' }}>
              <SearchOutlined style={{ marginRight: 6, color: '#1677ff' }} />
              {searchResults.length} results for "{query}"
            </span>
            {scrapedCount > 0 && (
              <Tag color="success" icon={<CheckCircleOutlined />}>
                {scrapedCount} scraped
              </Tag>
            )}
            {errorCount > 0 && (
              <Tag color="error" icon={<CloseCircleOutlined />}>
                {errorCount} failed
              </Tag>
            )}
          </Space>
          <Space>
            <Tooltip title="Clear all results">
              <Button
                size="small"
                icon={<ClearOutlined />}
                onClick={() => {
                  setSearchResults([]);
                  setScrapedPages(new Map());
                  setScrapeErrors(new Map());
                  setQuery('');
                }}
              >
                Clear
              </Button>
            </Tooltip>
          </Space>
        </div>
      )}

      {/* Results list */}
      {!isSearching && searchResults.length > 0 && (
        <div>
          {searchResults.map((item) => {
            const isScraping = scrapingUrls.has(item.url);
            const scraped = scrapedPages.get(item.url);
            const scrapeError = scrapeErrors.get(item.url);
            const isExpanded = expandedPanels.has(item.url);

            return (
              <Card
                key={item.url}
                hoverable
                style={{
                  marginBottom: 14,
                  borderRadius: 12,
                  boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
                  border: scraped
                    ? '1px solid #b7eb8f'
                    : scrapeError
                      ? '1px solid #ffa39e'
                      : '1px solid #e8e8e8',
                  transition: 'all 0.3s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}>
                  {/* Result info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      {scraped && (
                        <Badge status="success" />
                      )}
                      {scrapeError && (
                        <Badge status="error" />
                      )}
                      {isScraping && (
                        <Spin size="small" />
                      )}
                      <h4 style={{
                        margin: 0,
                        fontSize: 15,
                        fontWeight: 600,
                        lineHeight: 1.4,
                      }}>
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ color: '#1677ff' }}
                        >
                          {item.title || item.url}
                        </a>
                      </h4>
                    </div>
                    <p style={{ color: '#8c8c8c', margin: '4px 0', fontSize: 12 }}>
                      <LinkOutlined style={{ marginRight: 4 }} />
                      {getDomain(item.url)}
                      <span style={{ marginLeft: 8, color: '#d9d9d9' }}>#{item.position}</span>
                    </p>
                    <p style={{
                      color: '#595959',
                      margin: '8px 0 0',
                      fontSize: 13,
                      lineHeight: 1.6,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      display: '-webkit-box',
                      WebkitLineClamp: 3,
                      WebkitBoxOrient: 'vertical',
                    }}>
                      {item.description}
                    </p>

                    {/* Scrape status */}
                    {isScraping && (
                      <div style={{ marginTop: 12, padding: '8px 0' }}>
                        <Space>
                          <Spin size="small" />
                          <span style={{ color: '#1677ff', fontSize: 13 }}>Scraping page...</span>
                        </Space>
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  <div style={{ flexShrink: 0 }}>
                    <Space direction="vertical" align="end" size={8}>
                      <Button
                        type={scraped ? 'default' : 'primary'}
                        icon={isScraping ? <LoadingOutlined /> : (scraped ? <CheckCircleOutlined /> : <ExportOutlined />)}
                        onClick={() => handleScrapePage(item)}
                        loading={isScraping}
                        disabled={!!scraped || !!scrapeError}
                        style={{
                          borderRadius: 8,
                          minWidth: 135,
                          fontWeight: 500,
                          ...(scraped ? {
                            background: '#f6ffed',
                            borderColor: '#b7eb8f',
                            color: '#52c41a',
                          } : {}),
                          ...(scrapeError ? {
                            background: '#fff2f0',
                            borderColor: '#ffa39e',
                            color: '#ff4d4f',
                          } : {}),
                        }}
                      >
                        {scraped ? 'Scraped' : scrapeError ? 'Failed' : 'Scrape Page'}
                      </Button>

                      {(scraped || scrapeError) && (
                        <Button
                          type="link"
                          size="small"
                          icon={isExpanded ? <EyeOutlined /> : <CaretRightOutlined />}
                          onClick={() => togglePanel(item.url)}
                          style={{ padding: 0, height: 24, fontSize: 12 }}
                        >
                          {isExpanded ? 'Hide' : 'View'} details
                        </Button>
                      )}
                    </Space>
                  </div>
                </div>

                {/* Expandable details */}
                {isExpanded && (
                  <div style={{
                    marginTop: 16,
                    padding: '16px 0 0',
                    borderTop: '1px solid #f0f0f0',
                  }}>
                    {/* Error state */}
                    {scrapeError && (
                      <Result
                        status="error"
                        title="Scrape Failed"
                        subTitle={scrapeError}
                        style={{ padding: '16px 0' }}
                      >
                        <Button
                          type="primary"
                          size="small"
                          icon={<ExportOutlined />}
                          onClick={() => handleScrapePage(item)}
                        >
                          Retry
                        </Button>
                      </Result>
                    )}

                    {/* Success state */}
                    {scraped && (
                      <>
                        {/* Metadata */}
                        <div style={{
                          display: 'flex',
                          gap: 12,
                          flexWrap: 'wrap',
                          marginBottom: 16,
                        }}>
                          <Tag color="blue">Status: {scraped.status_code}</Tag>
                          {scraped.title && <Tag>{scraped.title}</Tag>}
                          {scraped.metadata?.language && (
                            <Tag color="purple">{scraped.metadata.language}</Tag>
                          )}
                          {scraped.content && (
                            <Tag color="cyan">{scraped.content.length.toLocaleString()} chars</Tag>
                          )}
                          {scraped.markdown && (
                            <Tag color="green">{scraped.markdown.length.toLocaleString()} chars (md)</Tag>
                          )}
                        </div>

                        {/* AI Analysis */}
                        {scraped.analysis && (
                          <div style={{
                            padding: 20,
                            background: 'linear-gradient(135deg, #f0f5ff 0%, #e6f4ff 100%)',
                            borderRadius: 10,
                            marginBottom: 14,
                            border: '1px solid #bae0ff',
                          }}>
                            <div style={{
                              fontSize: 13,
                              fontWeight: 600,
                              color: '#1677ff',
                              marginBottom: 10,
                            }}>
                              <ThunderboltOutlined style={{ marginRight: 6 }} />
                              AI Analysis
                            </div>
                            <div style={{ lineHeight: 1.8, fontSize: 14 }}>
                              <ReactMarkdown
                                remarkPlugins={[remarkMath]}
                                rehypePlugins={[rehypeKatex, rehypeRaw]}
                              >
                                {scraped.analysis}
                              </ReactMarkdown>
                            </div>
                          </div>
                        )}

                        {/* Raw content */}
                        <Collapse
                          size="small"
                          ghost
                          expandIconPosition="end"
                          items={[
                            {
                              key: 'raw',
                              label: (
                                <Space>
                                  <FileTextOutlined style={{ color: '#8c8c8c' }} />
                                  <span style={{ fontSize: 13 }}>Raw Content</span>
                                </Space>
                              ),
                              children: (
                                <div
                                  style={{
                                    maxHeight: 350,
                                    overflow: 'auto',
                                    padding: 14,
                                    background: '#fafafa',
                                    borderRadius: 8,
                                    fontSize: 12,
                                    whiteSpace: 'pre-wrap',
                                    fontFamily: 'monospace',
                                    border: '1px solid #f0f0f0',
                                  }}
                                >
                                  {scraped.markdown || scraped.content?.substring(0, 8000) || 'No content extracted'}
                                </div>
                              ),
                            },
                          ]}
                        />
                      </>
                    )}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}

      {/* Empty state */}
      {!isSearching && searchResults.length === 0 && (
        <div style={{ textAlign: 'center', padding: '60px 0' }}>
          <Empty
            image={<SearchOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />}
            description={
              <span style={{ color: '#8c8c8c', fontSize: 15 }}>
                Enter a search query above and click "Start Scraping" to begin
              </span>
            }
          />
        </div>
      )}

      {/* Footer */}
      {!isSearching && (
        <div style={{
          textAlign: 'center',
          marginTop: 48,
          padding: '20px 0',
          borderTop: '1px solid #f0f0f0',
          color: '#bfbfbf',
          fontSize: 12,
        }}>
          Web Scraper -- Multi-engine powered web scraping with AI analysis
        </div>
      )}
    </div>
  );
};

export default Search;
