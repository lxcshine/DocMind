import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Input, Button, Avatar, Space, Empty, Spin, message, Collapse, Tag, Drawer, Popconfirm, Modal, Upload } from 'antd';
import {
  SendOutlined,
  UserOutlined,
  RobotOutlined,
  PlusOutlined,
  FileTextOutlined,
  MenuOutlined,
  StopOutlined,
  DeleteOutlined,
  LoadingOutlined,
  PaperClipOutlined,
  CheckCircleOutlined,
  CloudUploadOutlined,
} from '@ant-design/icons';
import { useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';

const { TextArea } = Input;
const { Dragger } = Upload;

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

const ThinkingAnimation: React.FC<{ text: string }> = ({ text }) => {
  const [dots, setDots] = useState('');
  
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 400);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#8c8c8c', fontSize: 13, padding: '4px 0' }}>
      <LoadingOutlined style={{ color: '#722ed1', fontSize: 16 }} />
      <span>{text}{dots}</span>
    </div>
  );
};

const Chat: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const [thinkingText, setThinkingText] = useState('');
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const conversationId = searchParams.get('conversationId') || '';
  const isNew = searchParams.get('isNew') || '';

  const setConversationBoth = useCallback((convId: string, isNewVal: string) => {
    setSearchParams({ conversationId: convId, isNew: isNewVal });
  }, [setSearchParams]);

  useEffect(() => {
    loadSessions().finally(() => {
      setIsInitialLoad(false);
    });
    
    const handleFocus = () => loadSessions();
    window.addEventListener('focus', handleFocus);
    
    return () => {
      window.removeEventListener('focus', handleFocus);
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversations, conversationId]);

  const currentConversation = conversations.find(c => c.id === conversationId);

  const loadSessions = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch(`${API_BASE_URL}/chat/sessions`, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error('Failed to load sessions');
      }
      
      const data = await response.json();
      const sessions: Conversation[] = (data.sessions || []).map((s: any) => ({
        id: s.id,
        title: s.title,
        messages: [],
      }));
      
      // Merge sessions instead of replacing - preserve existing conversations with messages
      setConversations(prev => {
        const existingIds = new Set(prev.map(c => c.id));
        const newSessions = sessions.filter(s => !existingIds.has(s.id));
        return [...prev, ...newSessions];
      });
      
      if (sessions.length > 0 && !conversationId) {
        setConversationBoth(sessions[0].id, '');
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.warn('Loading sessions timed out - backend may be slow');
      } else {
        console.error('Failed to load sessions:', error);
      }
    }
  };

  const loadSessionMessages = async (sessionId: string) => {
    console.log('[loadSessionMessages] Loading messages for session:', sessionId);
    setLoadingMessages(true);
    try {
      const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`);
      console.log('[loadSessionMessages] Response status:', response.status);
      const data = await response.json();
      console.log('[loadSessionMessages] Received messages:', data.messages?.length || 0);
      const messages: Message[] = (data.messages || []).map((m: any) => {
        let sources: any[] = [];
        try {
          if (typeof m.sources === 'string') {
            sources = JSON.parse(m.sources);
          } else if (Array.isArray(m.sources)) {
            sources = m.sources;
          }
        } catch {
          sources = [];
        }
        return {
          id: m.id,
          role: m.role,
          content: m.content,
          timestamp: m.timestamp || Date.now(),
          sources,
          thinking: '',
        };
      });
      console.log('[loadSessionMessages] Updating conversations with', messages.length, 'messages');
      setConversations(prev => {
        console.log('[loadSessionMessages] Previous conversations count:', prev.length);
        const updated = prev.map(c => (c.id === sessionId ? { ...c, messages } : c));
        console.log('[loadSessionMessages] Updated conversations:', updated.map(c => ({ id: c.id, messages: c.messages.length })));
        return updated;
      });
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
      role: 'user',
      content: question,
      timestamp: Date.now(),
    };

    const aiMessageId = `a_${sessionId}_${Date.now()}`;
    const aiMessage: Message = {
      id: aiMessageId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      sources: [],
      thinking: '',
    };

    const title = question.slice(0, 30) + (question.length > 30 ? '...' : '');

    // Create the conversation with messages immediately
    const newConversation: Conversation = {
      id: sessionId,
      title,
      messages: [userMessage, aiMessage],
    };

    if (isNewConversation) {
      setConversations([newConversation]);
    } else {
      setConversations(prev =>
        prev.map(c =>
          c.id === sessionId
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
          message: question,
          session_id: sessionId,
          session_title: title,
          top_k: 5,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error('Chat request failed');
      }

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
              setConversations(prev =>
                prev.map(c =>
                  c.id === sessionId
                    ? {
                        ...c,
                        messages: c.messages.map(m =>
                          m.id === aiMessageId ? { ...m, content: fullContent, sources, thinking: thinkingSteps.join('\n') } : m
                        ),
                      }
                    : c
                )
              );
            } else if (parsed.type === 'done' || parsed.type === 'stopped') {
              break;
            } else if (parsed.type === 'error') {
              message.error(parsed.data);
              setConversations(prev =>
                prev.map(c =>
                  c.id === sessionId
                    ? {
                        ...c,
                        messages: c.messages.map(m =>
                          m.id === aiMessageId ? { ...m, content: parsed.data } : m
                        ),
                      }
                    : c
                )
              );
              break;
            }
          } catch (e) {
            console.warn('Failed to parse stream chunk:', line);
          }
        }
      }

      // Update URL to remove isNew flag after first message
      if (isNewConversation) {
        setConversationBoth(sessionId, '');
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        message.info('Generation stopped');
      } else {
        message.error('Failed to get response. Please check if the backend is running.');
        setConversations(prev =>
          prev.map(c =>
            c.id === sessionId
              ? {
                  ...c,
                  messages: c.messages.map(m =>
                    m.id === aiMessageId
                      ? { ...m, content: m.content || 'Sorry, I encountered an error. Please try again later.' }
                      : m
                  ),
                }
              : c
          )
        );
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
    const newConv: Conversation = {
      id: newSessionId,
      title: 'New Chat',
      messages: [],
    };
    setConversations(prev => [...prev, newConv]);
    setConversationBoth(newSessionId, 'true');
    setSidebarOpen(false);
    setInputValue('');
  };

  const handleDeleteSession = async (sessionId: string) => {
    try {
      await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`, { method: 'DELETE' });
      setConversations(prev => prev.filter(c => c.id !== sessionId));
      if (conversationId === sessionId) {
        const remaining = conversations.filter(c => c.id !== sessionId);
        if (remaining.length > 0) {
          setConversationBoth(remaining[0].id, '');
        } else {
          setConversationBoth('', '');
        }
      }
      message.success('Conversation deleted');
    } catch (error) {
      message.error('Failed to delete conversation');
    }
  };

  const handleUploadFile = async (file: any) => {
    setUploading(true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        const uploadedFile: UploadedFile = {
          doc_id: result.doc_id,
          filename: result.filename,
          file_size: result.file_size,
          file_type: result.file_type,
          status: result.status,
        };
        
        setPendingFiles(prev => [...prev, uploadedFile]);
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
      const response = await fetch(`${API_BASE_URL}/documents/${docId}/process`, {
        method: 'POST',
      });

      const result = await response.json();

      if (response.ok) {
        message.success('Document added to Knowledge Base!');
        setPendingFiles(prev => prev.filter(f => f.doc_id !== docId));
        if (pendingFiles.length <= 1) {
          setUploadModalVisible(false);
        }
      } else {
        message.error(`Failed to process: ${result.detail || 'Unknown error'}`);
      }
    } catch (error) {
      message.error('Failed to add to Knowledge Base');
    }
  };

  const handleSkipKB = (docId: string) => {
    setPendingFiles(prev => prev.filter(f => f.doc_id !== docId));
    message.info('File saved but not added to Knowledge Base');
    if (pendingFiles.length <= 1) {
      setUploadModalVisible(false);
    }
  };

  const handleAddAllToKB = async () => {
    for (const file of pendingFiles) {
      await handleAddToKB(file.doc_id);
    }
  };

  const handleSkipAll = () => {
    setPendingFiles([]);
    setUploadModalVisible(false);
    message.info('Files saved but not added to Knowledge Base');
  };

  const ConversationSidebarContent = (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Button
        type="primary"
        icon={<PlusOutlined />}
        block
        onClick={handleNewConversation}
        style={{ marginBottom: 16 }}
      >
        New Conversation
      </Button>

      <div className="conversation-list" style={{ overflowY: 'auto', flex: 1 }}>
        {conversations.map(conv => (
          <div
            key={conv.id}
            className={`conversation-item ${conv.id === conversationId ? 'active' : ''}`}
            onClick={() => {
              console.log('[Chat] Clicking conversation:', conv.id, 'messages length:', conv.messages.length);
              setConversationBoth(conv.id, '');
              if (conv.messages.length === 0) {
                console.log('[Chat] Loading session messages for:', conv.id);
                loadSessionMessages(conv.id);
              }
              setSidebarOpen(false);
            }}
            style={{
              padding: '10px 12px',
              marginBottom: 8,
              borderRadius: 6,
              cursor: 'pointer',
              background: conv.id === conversationId ? '#e6f4ff' : 'white',
              border: '1px solid #f0f0f0',
              position: 'relative',
            }}
          >
            <div style={{ fontWeight: 500, fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', paddingRight: 24 }}>
              {conv.title}
            </div>
            <div style={{ fontSize: 12, color: '#8c8c8c', marginTop: 4 }}>
              {conv.messages.length} messages
            </div>
            <Popconfirm
              title="Delete this conversation?"
              onConfirm={(e) => {
                e?.stopPropagation();
                handleDeleteSession(conv.id);
              }}
              onCancel={(e) => e?.stopPropagation()}
              okText="Yes"
              cancelText="No"
            >
              <Button
                type="text"
                size="small"
                danger
                icon={<DeleteOutlined />}
                onClick={(e) => e.stopPropagation()}
                style={{ position: 'absolute', right: 4, top: 8, padding: '2px 4px' }}
              />
            </Popconfirm>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="chat-page" style={{ display: 'flex', height: 'calc(100vh - 64px)', gap: 0, position: 'relative' }}>
      {isInitialLoad ? (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%' }}>
          <Spin size="large" tip="Loading..." />
        </div>
      ) : (
        <>
      <Drawer
        title="Conversations"
        placement="left"
        onClose={() => setSidebarOpen(false)}
        open={sidebarOpen}
        width={280}
        styles={{ body: { padding: 16 } }}
      >
        {ConversationSidebarContent}
      </Drawer>

      <div className="chat-main" style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'white' }}>
        <div style={{ padding: '8px 16px', borderBottom: '1px solid #f0f0f0', display: 'flex', alignItems: 'center', gap: 12, minHeight: 48 }}>
          <Button
            type="text"
            icon={<MenuOutlined />}
            onClick={() => setSidebarOpen(true)}
            size="small"
          />
          <span style={{ fontWeight: 500, fontSize: 15, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {currentConversation?.title || 'Select a conversation'}
          </span>
          <Button
            type="text"
            icon={<PlusOutlined />}
            onClick={() => {
              loadSessions();
              message.success('Refreshed');
            }}
            size="small"
            title="Refresh conversations"
          />
          {isLoading && (
            <Button
              type="primary"
              danger
              size="small"
              icon={<StopOutlined />}
              onClick={handleStop}
            >
              Stop
            </Button>
          )}
        </div>

        <div className="messages-container" style={{ flex: 1, overflowY: 'auto', padding: '16px 24px' }}>
          {loadingMessages ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <Spin size="large" tip="Loading conversation..." />
            </div>
          ) : !currentConversation || currentConversation.messages.length === 0 ? (
            <Empty description="Start a new conversation or select an existing one" />
          ) : (
            currentConversation.messages.map(msg => (
              <div
                key={msg.id}
                className={`message-item ${msg.role}`}
                style={{
                  display: 'flex',
                  gap: 12,
                  marginBottom: 16,
                  flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                }}
              >
                <Avatar
                  size={32}
                  icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                  style={{ background: msg.role === 'user' ? '#1677ff' : '#722ed1', flexShrink: 0 }}
                />
                <div
                  className="message-content"
                  style={{
                    maxWidth: '80%',
                    padding: '10px 14px',
                    borderRadius: 12,
                    background: msg.role === 'user' ? '#e6f4ff' : '#f5f5f5',
                    fontSize: 14,
                    lineHeight: 1.6,
                  }}
                >
                  {msg.role === 'assistant' ? (
                    <>
                      {msg.thinking && !msg.content && (
                        <ThinkingAnimation text={msg.thinking} />
                      )}
                      
                      {msg.content ? (
                        <ReactMarkdown
                          remarkPlugins={[remarkMath]}
                          rehypePlugins={[rehypeKatex, rehypeRaw]}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      ) : (
                        !msg.thinking && <Spin size="small" />
                      )}
                      
                      {msg.sources && msg.sources.length > 0 && (
                        <Collapse
                          size="small"
                          style={{ marginTop: 8, background: 'transparent', border: 'none' }}
                          items={[{
                            key: 'sources',
                            label: (
                              <Space>
                                <FileTextOutlined />
                                <span style={{ fontSize: 12, color: '#8c8c8c' }}>
                                  {msg.sources.length} source(s)
                                </span>
                              </Space>
                            ),
                            children: (
                              <div style={{ fontSize: 12 }}>
                                {msg.sources.map((source, idx) => (
                                  <div key={idx} style={{ marginBottom: 8, padding: 8, background: '#fff', borderRadius: 4 }}>
                                    <Tag color="blue">{source.doc_title || 'Unknown'}</Tag>
                                    {source.section_title && <Tag>{source.section_title}</Tag>}
                                    <div style={{ marginTop: 4, color: '#595959', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                      {source.content}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ),
                          }]}
                        />
                      )}
                    </>
                  ) : (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  )}
                </div>
              </div>
            ))
          )}

          {isLoading && thinkingText && (
            <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
              <Avatar
                size={32}
                icon={<RobotOutlined />}
                style={{ background: '#722ed1', flexShrink: 0 }}
              />
              <div style={{ padding: '10px 14px', borderRadius: 12, background: '#f9f0ff', maxWidth: '80%' }}>
                <ThinkingAnimation text={thinkingText} />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="input-area" style={{ padding: '10px 16px', borderTop: '1px solid #f0f0f0', background: '#fafafa' }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
            <Upload
              accept=".pdf,.md,.txt"
              showUploadList={false}
              beforeUpload={handleUploadFile}
              disabled={uploading}
            >
              <Button
                type="text"
                icon={<PaperClipOutlined />}
                size="large"
                title="Upload file"
                disabled={uploading}
              />
            </Upload>
            <TextArea
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onPressEnter={e => {
                if (!e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Enter your question... (Shift+Enter for new line)"
              autoSize={{ minRows: 1, maxRows: 4 }}
              style={{ flex: 1 }}
              disabled={isLoading}
            />
            {isLoading ? (
              <Button
                type="primary"
                danger
                icon={<StopOutlined />}
                onClick={handleStop}
              >
                Stop
              </Button>
            ) : (
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={handleSend}
                disabled={!inputValue.trim()}
              >
                Send
              </Button>
            )}
          </div>
        </div>
      </div>

      <Modal
        title={
          <Space>
            <CloudUploadOutlined />
            <span>Add to Knowledge Base?</span>
          </Space>
        }
        open={uploadModalVisible}
        onCancel={handleSkipAll}
        footer={[
          <Button key="skip" onClick={handleSkipAll}>
            Skip
          </Button>,
          <Button key="add" type="primary" icon={<CheckCircleOutlined />} onClick={handleAddAllToKB}>
            Add to KB
          </Button>,
        ]}
        width={600}
      >
        <p style={{ marginBottom: 16, color: '#595959' }}>
          Your file(s) have been uploaded. Would you like to add them to the Knowledge Base for future reference?
        </p>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {pendingFiles.map(file => (
            <div
              key={file.doc_id}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 16px',
                marginBottom: 8,
                background: '#fafafa',
                borderRadius: 6,
                border: '1px solid #f0f0f0',
              }}
            >
              <Space>
                {file.file_type === 'PDF' ? (
                  <FileTextOutlined style={{ color: '#ff4d4f', fontSize: 20 }} />
                ) : (
                  <FileTextOutlined style={{ color: '#1677ff', fontSize: 20 }} />
                )}
                <div>
                  <div style={{ fontWeight: 500 }}>{file.filename}</div>
                  <div style={{ fontSize: 12, color: '#8c8c8c' }}>
                    {(file.file_size / 1024).toFixed(1)} KB ? {file.file_type}
                  </div>
                </div>
              </Space>
              <Space>
                <Button
                  size="small"
                  onClick={() => handleSkipKB(file.doc_id)}
                >
                  Skip
                </Button>
                <Button
                  size="small"
                  type="primary"
                  icon={<CheckCircleOutlined />}
                  onClick={() => handleAddToKB(file.doc_id)}
                >
                  Add to KB
                </Button>
              </Space>
            </div>
          ))}
        </div>
      </Modal>
        </>
      )}
    </div>
  );
};

export default Chat;
