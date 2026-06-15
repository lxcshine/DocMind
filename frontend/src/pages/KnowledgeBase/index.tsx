import React, { useState, useEffect, useMemo } from 'react';
import { Upload, Button, Space, message, Progress, Popconfirm, Dropdown, Empty, Tooltip } from 'antd';
import {
  InboxOutlined,
  FilePdfOutlined,
  FileTextOutlined,
  DeleteOutlined,
  CloudUploadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  FolderAddOutlined,
  ThunderboltOutlined,
  DownOutlined,
  DatabaseOutlined,
  RocketOutlined,
  ClockCircleOutlined,
  AppstoreOutlined,
  BarsOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import type { UploadFile } from 'antd';
import type { ProcessingMode } from '../../types/api';
import { BatchImportModal } from '../../components';
import styles from './KnowledgeBase.module.css';

const { Dragger } = Upload;

const API_BASE_URL = '/api';

interface Document {
  key: string;
  name: string;
  type: string;
  size: string;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  progress: number;
  current_step: string;
  completed_steps: number;
  total_steps: number;
  sections: number;
  updatedAt: string;
}

type ViewMode = 'grid' | 'list';

const KnowledgeBase: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [batchModalOpen, setBatchModalOpen] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('list');

  useEffect(() => {
    fetchDocuments();
    return () => {
      if (pollingInterval) clearInterval(pollingInterval);
    };
  }, []);

  // Auto-poll while any document is processing
  useEffect(() => {
    const hasProcessing = documents.some(d => d.status === 'processing');
    if (hasProcessing && !pollingInterval) {
      const interval = setInterval(fetchDocuments, 2000);
      setPollingInterval(interval);
    } else if (!hasProcessing && pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  }, [documents, pollingInterval]);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/list`);
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

  const stats = useMemo(() => {
    const total = documents.length;
    const indexed = documents.filter(d => d.status === 'completed').length;
    const processing = documents.filter(d => d.status === 'processing').length;
    const totalSections = documents.reduce((sum, d) => sum + (d.sections || 0), 0);
    return { total, indexed, processing, totalSections };
  }, [documents]);

  const handleUpload = async (file: UploadFile) => {
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file as any);
    try {
      const response = await fetch(`${API_BASE_URL}/documents/upload`, { method: 'POST', body: formData });
      const result = await response.json();
      if (response.ok && result.success) {
        message.success(`Saved ${result.data.filename}. Click "Add to KB" to process.`);
        fetchDocuments();
      } else {
        message.error(`Upload failed: ${result.message || 'Unknown error'}`);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Network error';
      message.error(`Failed to upload ${file.name}: ${errorMessage}`);
    } finally {
      setUploading(false);
    }
    return false;
  };

  const handleAddToKB = async (docId: string, mode: ProcessingMode = 'standard') => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${docId}/process?mode=${mode}`, { method: 'POST' });
      const result = await response.json();
      if (response.ok) {
        const modeLabels = { fast: 'Fast', standard: 'Standard', full: 'Full' };
        message.success(`Processing started in ${modeLabels[mode]} mode!`);
        fetchDocuments();
      } else {
        message.error(`Failed to process: ${result.detail || 'Unknown error'}`);
      }
    } catch {
      message.error('Failed to start processing');
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${docId}`, { method: 'DELETE' });
      if (response.ok) {
        message.success('Document deleted');
        fetchDocuments();
      } else {
        message.error('Failed to delete document');
      }
    } catch {
      message.error('Failed to delete document');
    }
  };

  const renderStatus = (doc: Document) => {
    const map: Record<Document['status'], { label: string; cls: string; icon: React.ReactNode }> = {
      uploaded:   { label: 'Uploaded',  cls: 'uploaded',   icon: null },
      processing: { label: 'Processing',cls: 'processing', icon: null },
      completed:  { label: 'Indexed',   cls: 'completed',  icon: <CheckCircleOutlined /> },
      failed:     { label: 'Failed',    cls: 'failed',     icon: <CloseCircleOutlined /> },
    };
    const s = map[doc.status];
    return (
      <span className={`${styles['status-pill']} ${styles[s.cls]}`}>
        <span className={styles.dot} />
        {s.label}
      </span>
    );
  };

  const renderActionMenu = (doc: Document) => {
    if (doc.status === 'uploaded' || doc.status === 'failed') {
      const isRetry = doc.status === 'failed';
      return (
        <Dropdown
          menu={{
            items: [
              {
                key: 'fast',
                label: (
                  <Space size={6}>
                    <ThunderboltOutlined />
                    <span>Fast</span>
                    <span style={{ color: 'var(--color-text-tertiary)', fontSize: 11 }}>Vector · ~1-2 min</span>
                  </Space>
                ),
                onClick: () => handleAddToKB(doc.key, 'fast'),
              },
              {
                key: 'standard',
                label: (
                  <Space size={6}>
                    <CloudUploadOutlined />
                    <span>Standard</span>
                    <span style={{ color: 'var(--color-text-tertiary)', fontSize: 11 }}>KG + Vector · ~3-8 min</span>
                  </Space>
                ),
                onClick: () => handleAddToKB(doc.key, 'standard'),
              },
              {
                key: 'full',
                label: (
                  <Space size={6}>
                    <RocketOutlined />
                    <span>Full</span>
                    <span style={{ color: 'var(--color-text-tertiary)', fontSize: 11 }}>+ Multimodal · ~10-30 min</span>
                  </Space>
                ),
                onClick: () => handleAddToKB(doc.key, 'full'),
              },
            ],
          }}
        >
          <Button type="primary" size="small" icon={isRetry ? <SyncOutlined /> : <CloudUploadOutlined />}>
            {isRetry ? 'Retry' : 'Add to KB'} <DownOutlined />
          </Button>
        </Dropdown>
      );
    }
    if (doc.status === 'processing') {
      return (
        <Button type="default" size="small" icon={<SyncOutlined spin />} disabled>
          Processing…
        </Button>
      );
    }
    return (
      <span style={{ color: 'var(--color-success)', fontSize: 'var(--text-sm)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
        <CheckCircleOutlined /> Ready
      </span>
    );
  };

  const renderDocIcon = (type: string) => {
    const cls = type === 'PDF' ? styles.pdf : (type === 'MD' ? styles.md : styles.txt);
    const Icon = type === 'PDF' ? FilePdfOutlined : FileTextOutlined;
    return (
      <div className={`${styles['doc-icon']} ${cls}`}>
        <Icon />
      </div>
    );
  };

  return (
    <div className="page-shell">
      <div className="page-header">
        <div className="page-header-text">
          <span className="page-eyebrow">Library</span>
          <h1 className="page-title">Knowledge Base</h1>
          <p className="page-description">
            Upload documents, then process them into vector + graph indexes. Your knowledge base powers the AI chat and semantic search across the app.
          </p>
        </div>
        <div className="page-header-actions">
          <Button icon={<FolderAddOutlined />} onClick={() => setBatchModalOpen(true)}>
            Batch Import
          </Button>
        </div>
      </div>

      <div className={styles['kb-shell']}>
        {/* Stat cards */}
        <div className={styles['kb-stats']}>
          <div className={`${styles['stat-card']} ${styles.brand}`}>
            <div className={styles['stat-head']}>
              <div className={`${styles['stat-icon']} ${styles.brand}`}><DatabaseOutlined /></div>
              <span className={styles['stat-trend']}>All time</span>
            </div>
            <div className={styles['stat-value']}>{stats.total}</div>
            <div className={styles['stat-label']}>Total documents</div>
          </div>
          <div className={`${styles['stat-card']} ${styles.success}`}>
            <div className={styles['stat-head']}>
              <div className={`${styles['stat-icon']} ${styles.success}`}><CheckCircleOutlined /></div>
              <span className={styles['stat-trend']}>Searchable</span>
            </div>
            <div className={styles['stat-value']}>{stats.indexed}</div>
            <div className={styles['stat-label']}>Indexed</div>
          </div>
          <div className={`${styles['stat-card']} ${styles.warning}`}>
            <div className={styles['stat-head']}>
              <div className={`${styles['stat-icon']} ${styles.warning}`}><ClockCircleOutlined /></div>
              <span className={styles['stat-trend']}>Live</span>
            </div>
            <div className={styles['stat-value']}>{stats.processing}</div>
            <div className={styles['stat-label']}>Processing</div>
          </div>
          <div className={`${styles['stat-card']} ${styles.accent}`}>
            <div className={styles['stat-head']}>
              <div className={`${styles['stat-icon']} ${styles.accent}`}><RocketOutlined /></div>
              <span className={styles['stat-trend']}>Across all docs</span>
            </div>
            <div className={styles['stat-value']}>{stats.totalSections}</div>
            <div className={styles['stat-label']}>Total sections</div>
          </div>
        </div>

        {/* Upload zone */}
        <div className={styles['upload-zone']}>
          <Dragger
            accept=".pdf,.md,.txt"
            multiple
            beforeUpload={handleUpload}
            maxCount={10}
            disabled={uploading}
            showUploadList={false}
          >
            <div className={styles['upload-icon']}>
              <InboxOutlined />
            </div>
            <div className={styles['upload-title']}>
              {uploading ? 'Uploading…' : 'Drop files here or click to upload'}
            </div>
            <div className={styles['upload-subtitle']}>
              Save documents to your library. Process them into the knowledge base with one click.
            </div>
            <div className={styles['upload-hint']}>
              PDF · Markdown · TXT · Up to 10 files at once
            </div>
          </Dragger>
        </div>

        {/* Document list panel */}
        <div className={styles['kb-panel']}>
          <div className={styles['kb-panel-header']}>
            <div>
              <div className={styles['kb-panel-title']}>Documents</div>
              <div className={styles['kb-panel-subtitle']}>
                {stats.total} {stats.total === 1 ? 'document' : 'documents'} · {stats.indexed} indexed
              </div>
            </div>
            <div className={styles['kb-panel-actions']}>
              <Tooltip title="Refresh">
                <Button
                  type="text"
                  icon={<ReloadOutlined />}
                  onClick={fetchDocuments}
                  size="small"
                />
              </Tooltip>
              <div style={{ display: 'flex', background: 'var(--color-bg-sunken)', borderRadius: 'var(--radius-md)', padding: 2 }}>
                <Button
                  type={viewMode === 'list' ? 'primary' : 'text'}
                  size="small"
                  icon={<BarsOutlined />}
                  onClick={() => setViewMode('list')}
                  style={viewMode === 'list' ? { background: 'var(--color-surface)', boxShadow: 'var(--shadow-xs)' } : { background: 'transparent' }}
                />
                <Button
                  type={viewMode === 'grid' ? 'primary' : 'text'}
                  size="small"
                  icon={<AppstoreOutlined />}
                  onClick={() => setViewMode('grid')}
                  style={viewMode === 'grid' ? { background: 'var(--color-surface)', boxShadow: 'var(--shadow-xs)' } : { background: 'transparent' }}
                />
              </div>
            </div>
          </div>

          {documents.length === 0 ? (
            <div className={styles['kb-empty']}>
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="No documents yet. Upload files above to get started."
              />
            </div>
          ) : (
            <div>
              {documents.map(doc => (
                <div key={doc.key} className={styles['doc-row']}>
                  {renderDocIcon(doc.type)}
                  <div className={styles['doc-meta']}>
                    <div className={styles['doc-name']}>{doc.name}</div>
                    <div className={styles['doc-sub']}>
                      <span>{doc.type}</span>
                      <span>·</span>
                      <span>{doc.size}</span>
                      {doc.sections > 0 && <><span>·</span><span>{doc.sections} sections</span></>}
                      {doc.updatedAt && <><span>·</span><span>{doc.updatedAt}</span></>}
                    </div>
                  </div>
                  {doc.status === 'processing' || doc.progress > 0 ? (
                    <div className={styles['doc-progress']}>
                      <div style={{ flex: 1, minWidth: 100 }}>
                        <Progress
                          percent={doc.progress}
                          size="small"
                          status={doc.status === 'failed' ? 'exception' : 'active'}
                          showInfo={false}
                        />
                        {doc.current_step && (
                          <div className={styles['doc-step']}>{doc.current_step}</div>
                        )}
                      </div>
                      <span className={styles['doc-progress-text']}>{doc.progress}%</span>
                    </div>
                  ) : (
                    <div style={{ minWidth: 200 }}>{renderStatus(doc)}</div>
                  )}
                  <div className={styles['doc-actions']}>
                    {renderActionMenu(doc)}
                    <Popconfirm
                      title="Delete this document?"
                      description="This will remove the file and all processed sections."
                      onConfirm={() => handleDelete(doc.key)}
                      okText="Delete"
                      cancelText="Cancel"
                    >
                      <Button type="text" size="small" danger icon={<DeleteOutlined />} />
                    </Popconfirm>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <BatchImportModal
        open={batchModalOpen}
        onClose={() => setBatchModalOpen(false)}
        onComplete={fetchDocuments}
      />
    </div>
  );
};

export default KnowledgeBase;
