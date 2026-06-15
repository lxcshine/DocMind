import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Input, Button, Space, Spin, message, Tag, Popconfirm, Modal, Upload, Segmented, Tooltip, Empty } from 'antd';
import {
  SendOutlined,
  UserOutlined,
  RobotOutlined,
  PlusOutlined,
  FileTextOutlined,
  StopOutlined,
  DeleteOutlined,
  LoadingOutlined,
  PaperClipOutlined,
  CheckCircleOutlined,
  CloudUploadOutlined,
  DatabaseOutlined,
  CommentOutlined,
  ThunderboltOutlined,
  SearchOutlined,
  ReloadOutlined,
  BulbOutlined,
} from '@ant-design/icons';
import { useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';
import styles from './Chat.module.css';

const { TextArea } = Input;

const API_BASE_URL = 'http://localhost:8000/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  sources?: any[];
  thinking?: string;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}

interface UploadedFile {
  doc_id: string;
  filename: string;
  file_size: number;
  file_type: string;
  status: string;
}

const SUGGESTIONS = [
  { icon: <BulbOutlined />, text: 'Summarize my recent documents' },
  { icon: <DatabaseOutlined />, text: 'What topics are in my knowledge base?' },
  { icon: <ThunderboltOutlined />, text: 'Find action items from meeting notes' },
  { icon: <SearchOutlined />, text: 'Search for the latest research findings' },
];

const ThinkingAnimation: React.FC<{ text: string }> = ({ text }) => {
  const [dots, setDots] = useState('');
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 400);
    return () => clearInterval(interval);
  }, []);
  return (
    <div className={styles['thinking-pill']}>
      <LoadingOutlined />
      <span>{text}{dots}</span>
    </div>
  );
};

const Chat: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const [thinkingText, setThinkingText] = useState('');
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [chatMode, setChatMode] = useState<'kb' | 'direct'>('kb');
  const [searchQuery, setSearchQuery] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const conversationId = searchParams.get('conversationId') || '';
  const isNew = searchParams.get('isNew') || '';

  const setConversationBoth = useCallback((convId: string, isNewVal: string) => {
    setSearchParams({ conversationId: convId, isNew: isNewVal });
  }, [setSearchParams]);

  useEffect(() => {
    loadSessions().finally(() => setIsInitialLoad(false));
    const handleFocus = () => loadSessions();
    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversations, conversationId, isLoading]);

  useEffect(() => {
    if (conversationId) {
      const conv = conversations.find(c => c.id === conversationId);
      if (conv && conv.messages.length === 0) {
        loadSessionMessages(conversationId);
      }
    }
  }, [conversationId]);

  const currentConversation = conversations.find(c => c.id === conversationId);

  const loadSessions = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      const response = await fetch(`${API_BASE_URL}/chat/sessions`, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (!response.ok) throw new Error('Failed to load sessions');
      const data = await response.json();
      const sessions: Conversation[] = (data.sessions || []).map((s: any) => ({
        id: s.id, title: s.title, messages: [],
      }));
      setConversations(prev => {
        const existingIds = new Set(prev.map(c => c.id));
        const newSessions = sessions.filter(s => !existingIds.has(s.id));
        return [...prev, ...newSessions];
      });
      if (sessions.length > 0 && !conversationId) {
        setConversationBoth(sessions[0].id, '');
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') console.error('Failed to load sessions:', error);
    }
  };

  const loadSessionMessages = async (sessionId: string) => {
    setLoadingMessages(true);
    try {
      const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`);
      const data = await response.json();
      const messages: Message[] = (data.messages || []).map((m: any) => {
        let sources: any[] = [];
        try {
          if (typeof m.sources === 'string') sources = JSON.parse(m.sources);
          else if (Array.isArray(m.sources)) sources = m.sources;
        } catch { sources = []; }
        return {
          id: m.id, role: m.role, content: m.content,
          timestamp: m.timestamp || Date.now(), sources, thinking: '',
        };
      });
      setConversations(prev => prev.map(c => (c.id === sessionId ? { ...c, messages } : c)));
    } catch (error) {
      console.error('Failed to load messages:', error);
      message.error('Failed to load messages');
    } finally {
      setLoadingMessages(false);
    }
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    let sessionId = conversationId;
    const isNewConversation = isNew === 'true' || !conversationId;

    if (isNewConversation) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      setConversationBoth(sessionId, 'true');
    }

    const question = inputValue;
    setInputValue('');
    setIsLoading(true);
    setThinkingText('');

    const userMessage: Message = {
      id: `u_${sessionId}_${Date.now()}`,
      role: 'user', content: question, timestamp: Date.now(),
    };
    const aiMessageId = `a_${sessionId}_${Date.now()}`;
    const aiMessage: Message = {
      id: aiMessageId, role: 'assistant', content: '', timestamp: Date.now(), sources: [], thinking: '',
    };
    const title = question.slice(0, 30) + (question.length > 30 ? '…' : '');

    if (isNewConversation) {
      setConversations([{ id: sessionId, title, messages: [userMessage, aiMessage] }]);
    } else {
      setConversations(prev =>
        prev.map(c => c.id === sessionId
          ? { ...c, messages: [...c.messages, userMessage, aiMessage] }
          : c
        )
      );
    }

    const controller = new AbortController();
    setAbortController(controller);

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: question, session_id: sessionId, session_title: title,
          top_k: 5, chat_mode: chatMode,
        }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error('Chat request failed');

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let fullContent = '';
      let sources: any[] = [];
      let thinkingSteps: string[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(Boolean);
        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            if (parsed.type === 'sources') {
              sources = parsed.data;
            } else if (parsed.type === 'thinking') {
              thinkingSteps.push(parsed.data);
              setThinkingText(parsed.data);
            } else if (parsed.type === 'thinking_done') {
              setThinkingText('');
            } else if (parsed.type === 'chunk') {
              fullContent += parsed.data;
              setConversations(prev => prev.map(c => c.id === sessionId
                ? { ...c, messages: c.messages.map(m =>
                    m.id === aiMessageId ? { ...m, content: fullContent, sources, thinking: thinkingSteps.join('\n') } : m
                  ) }
                : c
              ));
            } else if (parsed.type === 'done' || parsed.type === 'stopped') {
              break;
            } else if (parsed.type === 'error') {
              message.error(parsed.data);
              setConversations(prev => prev.map(c => c.id === sessionId
                ? { ...c, messages: c.messages.map(m => m.id === aiMessageId ? { ...m, content: parsed.data } : m) }
                : c
              ));
              break;
            }
          } catch (e) {
            console.warn('Failed to parse stream chunk:', line);
          }
        }
      }
      if (isNewConversation) setConversationBoth(sessionId, '');
    } catch (error: any) {
      if (error.name === 'AbortError') {
        message.info('Generation stopped');
      } else {
        message.error('Failed to get response. Please check if the backend is running.');
        setConversations(prev => prev.map(c => c.id === sessionId
          ? { ...c, messages: c.messages.map(m => m.id === aiMessageId
              ? { ...m, content: m.content || 'Sorry, I encountered an error. Please try again later.' }
              : m
          ) }
          : c
        ));
      }
    } finally {
      setIsLoading(false);
      setAbortController(null);
      setThinkingText('');
    }
  };

  const handleStop = () => {
    if (abortController) {
      abortController.abort();
      fetch(`${API_BASE_URL}/chat/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: conversationId || 'default' }),
      }).catch(() => {});
    }
  };

  const handleNewConversation = () => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setConversations(prev => [...prev, { id: newSessionId, title: 'New Chat', messages: [] }]);
    setConversationBoth(newSessionId, 'true');
    setInputValue('');
  };

  const handleDeleteSession = async (sessionId: string) => {
    try {
      await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`, { method: 'DELETE' });
      setConversations(prev => prev.filter(c => c.id !== sessionId));
      if (conversationId === sessionId) {
        const remaining = conversations.filter(c => c.id !== sessionId);
        if (remaining.length > 0) setConversationBoth(remaining[0].id, '');
        else setConversationBoth('', '');
      }
      message.success('Conversation deleted');
    } catch {
      message.error('Failed to delete conversation');
    }
  };

  const handleUploadFile = async (file: any) => {
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch(`${API_BASE_URL}/documents/upload`, { method: 'POST', body: formData });
      const result = await response.json();
      if (response.ok) {
        setPendingFiles(prev => [...prev, {
          doc_id: result.doc_id, filename: result.filename,
          file_size: result.file_size, file_type: result.file_type, status: result.status,
        }]);
        setUploadModalVisible(true);
      } else {
        message.error(`Upload failed: ${result.detail || 'Unknown error'}`);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Network error';
      message.error(`Failed to upload ${file.name}: ${errorMessage}`);
    } finally {
      setUploading(false);
    }
    return false;
  };

  const handleAddToKB = async (docId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${docId}/process`, { method: 'POST' });
      const result = await response.json();
      if (response.ok) {
        message.success('Document added to Knowledge Base!');
        setPendingFiles(prev => prev.filter(f => f.doc_id !== docId));
        if (pendingFiles.length <= 1) setUploadModalVisible(false);
      } else {
        message.error(`Failed to process: ${result.detail || 'Unknown error'}`);
      }
    } catch {
      message.error('Failed to add to Knowledge Base');
    }
  };

  const handleSkipKB = (docId: string) => {
    setPendingFiles(prev => prev.filter(f => f.doc_id !== docId));
    if (pendingFiles.length <= 1) setUploadModalVisible(false);
  };

  const handleAddAllToKB = async () => {
    for (const file of pendingFiles) await handleAddToKB(file.doc_id);
  };

  const handleSkipAll = () => {
    setPendingFiles([]);
    setUploadModalVisible(false);
  };

  const handleSuggestionClick = (text: string) => {
    setInputValue(text);
  };

  const filteredConversations = conversations.filter(c =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const fileTypeClass = (type: string) => {
    const t = type.toUpperCase();
    if (t === 'PDF') return styles.pdf;
    if (t === 'MD') return styles.md;
    return styles.txt;
  };

  return (
    <div className={styles['chat-shell']}>
      {isInitialLoad ? (
        <div className={styles['chat-loading']} style={{ gridColumn: '1 / -1' }}>
          <Spin size="large">
            <div style={{ padding: 40, minWidth: 200, textAlign: 'center', color: 'var(--color-text-tertiary)', fontSize: 'var(--text-sm)' }}>
              Loading conversations…
            </div>
          </Spin>
        </div>
      ) : (
        <>
          {/* Conversation sidebar */}
          <aside className={styles['conversations-pane']}>
            <div className={styles['pane-header']}>
              <span className={styles['pane-title']}>Conversations</span>
              <Tooltip title="Reload">
                <Button
                  type="text"
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={() => { loadSessions(); message.success('Refreshed'); }}
                />
              </Tooltip>
            </div>
            <div className={styles['pane-search']}>
              <Input
                size="middle"
                prefix={<SearchOutlined style={{ color: 'var(--color-text-tertiary)' }} />}
                placeholder="Search…"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                allowClear
              />
            </div>
            <div className={styles['conversation-list']}>
              {filteredConversations.length === 0 ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="No conversations"
                  style={{ marginTop: 32 }}
                />
              ) : (
                filteredConversations.map(conv => (
                  <div
                    key={conv.id}
                    className={`${styles['conversation-item']} ${conv.id === conversationId ? styles.active : ''}`}
                    onClick={() => {
                      setConversationBoth(conv.id, '');
                      if (conv.messages.length === 0) loadSessionMessages(conv.id);
                    }}
                  >
                    <div className={styles['conversation-meta']}>
                      <div className={styles['conversation-title']}>{conv.title}</div>
                      <div className={styles['conversation-sub']}>
                        <span>{conv.messages.length} messages</span>
                      </div>
                    </div>
                    <div className={styles['conversation-actions']}>
                      <Popconfirm
                        title="Delete this conversation?"
                        onConfirm={(e) => { e?.stopPropagation(); handleDeleteSession(conv.id); }}
                        onCancel={(e) => e?.stopPropagation()}
                        okText="Delete"
                        cancelText="Cancel"
                      >
                        <Button
                          type="text"
                          size="small"
                          danger
                          icon={<DeleteOutlined />}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </Popconfirm>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div style={{ padding: 'var(--space-3)', borderTop: '1px solid var(--color-border-subtle)' }}>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                block
                onClick={handleNewConversation}
                style={{ height: 38 }}
              >
                New Conversation
              </Button>
            </div>
          </aside>

          {/* Main chat area */}
          <main className={styles['chat-main']}>
            <div className={styles['chat-toolbar']}>
              <span className={styles['chat-toolbar-title']}>
                {currentConversation?.title || 'New Chat'}
              </span>
              <Segmented
                size="small"
                value={chatMode}
                onChange={(value) => setChatMode(value as 'kb' | 'direct')}
                options={[
                  { label: <Space size={4}><DatabaseOutlined /> KB</Space>, value: 'kb' },
                  { label: <Space size={4}><CommentOutlined /> Direct</Space>, value: 'direct' },
                ]}
              />
            </div>

            <div className={styles['messages-container']}>
              {loadingMessages ? (
                <div className={styles['chat-loading']}>
                  <Spin />
                </div>
              ) : currentConversation && currentConversation.messages.length > 0 ? (
                currentConversation.messages.map(msg => (
                  <div key={msg.id} className={`${styles['message-row']} ${styles[msg.role]}`}>
                    <div className={`${styles['message-avatar']} ${styles[msg.role]}`}>
                      {msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                    </div>
                    <div className={styles['message-body']}>
                      <div className={styles['message-name']}>
                        {msg.role === 'user' ? 'You' : 'DocMind'}
                      </div>
                      <div className={styles['message-bubble']}>
                        {msg.role === 'assistant' ? (
                          <>
                            <ReactMarkdown
                              remarkPlugins={[remarkMath]}
                              rehypePlugins={[rehypeKatex, rehypeRaw]}
                            >
                              {msg.content || (isLoading ? '' : '…')}
                            </ReactMarkdown>
                            {msg.sources && msg.sources.length > 0 && (
                              <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--color-border-subtle)' }}>
                                <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-text-tertiary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                                  Sources
                                </div>
                                {msg.sources.map((source: any, idx: number) => (
                                  <div key={idx} style={{ marginBottom: 8, padding: 10, background: 'var(--color-bg)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border-subtle)' }}>
                                    <Space size={4} wrap>
                                      <Tag color="blue">{source.doc_title || 'Unknown'}</Tag>
                                      {source.section_title && <Tag>{source.section_title}</Tag>}
                                    </Space>
                                    <div style={{ marginTop: 6, color: 'var(--color-text-secondary)', fontSize: 'var(--text-sm)', lineHeight: 1.5 }}>
                                      {source.content}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </>
                        ) : (
                          <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className={styles['empty-state']}>
                  <div className={styles['empty-icon']}>
                    <ThunderboltOutlined />
                  </div>
                  <h2 className={styles['empty-title']}>Start a new conversation</h2>
                  <p className={styles['empty-description']}>
                    Ask anything about your knowledge base, or upload a document to get started. DocMind will search, reason, and cite sources automatically.
                  </p>
                  <div className={styles['suggestion-grid']}>
                    {SUGGESTIONS.map((s, i) => (
                      <button
                        key={i}
                        className={styles['suggestion-chip']}
                        onClick={() => handleSuggestionClick(s.text)}
                      >
                        {s.icon}
                        <span>{s.text}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {isLoading && thinkingText && (
                <div className={`${styles['message-row']} ${styles.assistant}`}>
                  <div className={`${styles['message-avatar']} ${styles.assistant}`}>
                    <RobotOutlined />
                  </div>
                  <div className={styles['message-body']}>
                    <ThinkingAnimation text={thinkingText} />
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className={styles['input-area']}>
              <div className={styles['input-shell']}>
                <div className={styles['input-row']}>
                  <Upload
                    accept=".pdf,.md,.txt"
                    showUploadList={false}
                    beforeUpload={handleUploadFile}
                    disabled={uploading}
                  >
                    <Tooltip title="Upload file">
                      <Button
                        type="text"
                        icon={<PaperClipOutlined />}
                        disabled={uploading}
                        style={{ color: 'var(--color-text-secondary)' }}
                      />
                    </Tooltip>
                  </Upload>
                  <TextArea
                    className={styles['input-textarea']}
                    value={inputValue}
                    onChange={e => setInputValue(e.target.value)}
                    onPressEnter={e => {
                      if (!e.shiftKey) {
                        e.preventDefault();
                        handleSend();
                      }
                    }}
                    placeholder="Message DocMind…"
                    autoSize={{ minRows: 1, maxRows: 6 }}
                    disabled={isLoading}
                  />
                  <div className={styles['input-actions']}>
                    {isLoading ? (
                      <Tooltip title="Stop generation">
                        <button
                          className={`${styles['send-btn']} ${styles.stop}`}
                          onClick={handleStop}
                          aria-label="Stop"
                        >
                          <StopOutlined />
                        </button>
                      </Tooltip>
                    ) : (
                      <Tooltip title="Send (Enter)">
                        <button
                          className={styles['send-btn']}
                          onClick={handleSend}
                          disabled={!inputValue.trim()}
                          aria-label="Send"
                        >
                          <SendOutlined />
                        </button>
                      </Tooltip>
                    )}
                  </div>
                </div>
                <div className={styles['input-toolbar']}>
                  <span className={styles['input-hint']}>
                    <span className={styles.kbd}>Enter</span> to send · <span className={styles.kbd}>Shift</span> + <span className={styles.kbd}>Enter</span> for new line
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    {chatMode === 'kb' ? <><DatabaseOutlined /> Knowledge Base</> : <><CommentOutlined /> Direct Chat</>}
                  </span>
                </div>
              </div>
            </div>
          </main>
        </>
      )}

      <Modal
        title={
          <Space>
            <CloudUploadOutlined style={{ color: 'var(--color-primary)' }} />
            <span>Add to Knowledge Base?</span>
          </Space>
        }
        open={uploadModalVisible}
        onCancel={handleSkipAll}
        footer={[
          <Button key="skip" onClick={handleSkipAll}>Skip</Button>,
          <Button key="add" type="primary" icon={<CheckCircleOutlined />} onClick={handleAddAllToKB}>
            Add to KB
          </Button>,
        ]}
        width={560}
      >
        <p style={{ marginBottom: 16, color: 'var(--color-text-secondary)', fontSize: 'var(--text-sm)' }}>
          Your file(s) have been uploaded. Would you like to add them to the Knowledge Base for future reference?
        </p>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {pendingFiles.map(file => (
            <div key={file.doc_id} className={styles['pending-file']}>
              <Space>
                <div className={`${styles['pending-file-icon']} ${fileTypeClass(file.file_type)}`}>
                  <FileTextOutlined />
                </div>
                <div>
                  <div style={{ fontWeight: 500, fontSize: 'var(--text-sm)' }}>{file.filename}</div>
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)' }}>
                    {(file.file_size / 1024).toFixed(1)} KB · {file.file_type}
                  </div>
                </div>
              </Space>
              <Space>
                <Button size="small" onClick={() => handleSkipKB(file.doc_id)}>Skip</Button>
                <Button size="small" type="primary" icon={<CheckCircleOutlined />} onClick={() => handleAddToKB(file.doc_id)}>
                  Add
                </Button>
              </Space>
            </div>
          ))}
        </div>
      </Modal>
    </div>
  );
};

export default Chat;
