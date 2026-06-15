import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Modal,
  Input,
  Button,
  Table,
  Progress,
  Tag,
  Steps,
  Alert,
  message,
  Tooltip,
  Radio,
} from 'antd';
import {
  FolderOpenOutlined,
  ScanOutlined,
  ThunderboltOutlined,
  StopOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  InboxOutlined,
  ClockCircleOutlined,
  CloudUploadOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import type { BatchJob, BatchDocItem, BatchScanResult, ProcessingMode } from '../types/api';
import { documentsApi } from '../services/api';
import styles from './BatchImportModal.module.css';

interface Props {
  open: boolean;
  onClose: () => void;
  onComplete: () => void;
}

type Step = 'input' | 'scanning' | 'confirm' | 'processing' | 'done';

const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const docStatusConfig: Record<string, { color: string; icon: React.ReactNode; text: string }> = {
  completed: { color: 'success', icon: <CheckCircleOutlined />, text: 'Completed' },
  processing: { color: 'processing', icon: <SyncOutlined spin />, text: 'Processing' },
  uploading: { color: 'processing', icon: <SyncOutlined spin />, text: 'Uploading' },
  queued: { color: 'warning', icon: <ClockCircleOutlined />, text: 'Queued' },
  pending: { color: 'default', icon: <InboxOutlined />, text: 'Pending' },
  failed: { color: 'error', icon: <CloseCircleOutlined />, text: 'Failed' },
  skipped: { color: 'default', icon: <InboxOutlined />, text: 'Skipped' },
};

const BatchImportModal: React.FC<Props> = ({ open, onClose, onComplete }) => {
  const [step, setStep] = useState<Step>('input');
  const [directory, setDirectory] = useState('');
  const [scanResult, setScanResult] = useState<BatchScanResult | null>(null);
  const [batchJob, setBatchJob] = useState<BatchJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('fast');
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup polling on unmount or close
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const resetState = useCallback(() => {
    setStep('input');
    setDirectory('');
    setScanResult(null);
    setBatchJob(null);
    setLoading(false);
    setError(null);
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const handleClose = () => {
    resetState();
    onClose();
  };

  // Step 1: Scan directory
  const handleScan = async () => {
    if (!directory.trim()) {
      message.warning('Please enter a directory path');
      return;
    }

    setLoading(true);
    setError(null);
    setStep('scanning');

    try {
      const result = await documentsApi.batchScan(directory.trim());
      setScanResult(result as unknown as BatchScanResult);
      setStep('confirm');
    } catch (err: any) {
      setError(err.message || 'Failed to scan directory');
      setStep('input');
    } finally {
      setLoading(false);
    }
  };

  // Step 2: Start batch processing
  const handleStart = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await documentsApi.batchStart(directory.trim(), processingMode);
      const job = (result as unknown) as BatchJob;
      setBatchJob(job);
      setStep('processing');
      startPolling(job.batch_id);
    } catch (err: any) {
      setError(err.message || 'Failed to start batch processing');
      setStep('confirm');
    } finally {
      setLoading(false);
    }
  };

  // Poll batch status
  const startPolling = (batchId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      try {
        const result = await documentsApi.batchGetStatus(batchId);
        const job = (result as unknown) as BatchJob;
        setBatchJob(job);

        // Stop polling when done
        if (job.status === 'completed' || job.status === 'failed' || job.status === 'partial') {
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
          setStep('done');
          onComplete();
        }
      } catch (err) {
        console.error('Poll error:', err);
      }
    }, 3000);
  };

  // Cancel batch
  const handleCancel = async () => {
    if (!batchJob) return;
    try {
      await documentsApi.batchCancel(batchJob.batch_id);
      message.info('Batch processing cancelled');
    } catch (err: any) {
      message.error(err.message || 'Failed to cancel');
    }
  };

  // Scan result table columns
  const scanColumns = [
    {
      title: 'File',
      dataIndex: 'filename',
      key: 'filename',
      ellipsis: true,
    },
    {
      title: 'Size',
      dataIndex: 'file_size',
      key: 'file_size',
      width: 100,
      render: (size: number) => formatSize(size),
    },
    {
      title: 'Type',
      dataIndex: 'extension',
      key: 'extension',
      width: 80,
      render: (ext: string) => ext.toUpperCase(),
    },
  ];

  // Processing table columns
  const processColumns = [
    {
      title: 'File',
      dataIndex: 'filename',
      key: 'filename',
      ellipsis: true,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 130,
      render: (status: string) => {
        const cfg = docStatusConfig[status] || docStatusConfig.pending;
        return <Tag color={cfg.color} icon={cfg.icon}>{cfg.text}</Tag>;
      },
    },
    {
      title: 'Progress',
      key: 'progress',
      width: 200,
      render: (_: any, record: BatchDocItem) => {
        if (record.status === 'completed') {
          return <span style={{ color: '#52c41a', fontWeight: 500 }}>Done</span>;
        }
        if (record.status === 'processing') {
          return (
            <div>
              <Progress percent={record.progress} size="small" status="active" />
              {record.current_step && (
                <div style={{ fontSize: 11, color: '#8c8c8c', marginTop: 2 }}>
                  {record.current_step}
                </div>
              )}
            </div>
          );
        }
        if (record.status === 'failed') {
          return (
            <Tooltip title={record.error}>
              <span style={{ color: '#ff4d4f', cursor: 'help' }}>
                Failed {record.retry_count > 0 ? `(${record.retry_count} retries)` : ''}
              </span>
            </Tooltip>
          );
        }
        return <span style={{ color: '#8c8c8c' }}>Waiting...</span>;
      },
    },
  ];

  // Render step indicator
  const currentStepIndex = ['input', 'scanning', 'confirm', 'processing', 'done'].indexOf(step);

  return (
    <Modal
      title="Batch Import Documents"
      open={open}
      onCancel={handleClose}
      width={840}
      footer={null}
      destroyOnHidden
      className={styles['batch-modal']}
    >
      <Steps
        current={currentStepIndex}
        size="small"
        className={styles['batch-steps']}
        items={[
          { title: 'Select Directory' },
          { title: 'Scan' },
          { title: 'Confirm' },
          { title: 'Processing' },
          { title: 'Done' },
        ]}
      />

      {error && (
        <Alert
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
          className={styles['batch-error']}
        />
      )}

      {/* Step 1: Input directory path */}
      {step === 'input' && (
        <div className={styles['batch-step-body']}>
          <p className={styles['batch-step-description']}>
            Enter a directory path on the server. The system will scan for all supported document files
            (PDF, Markdown, TXT, Word, PowerPoint, Excel, Images).
          </p>
          <div className={styles['batch-input-row']}>
            <Input
              placeholder="e.g., D:\Documents\Research or /home/user/papers"
              value={directory}
              onChange={(e) => setDirectory(e.target.value)}
              onPressEnter={handleScan}
              prefix={<FolderOpenOutlined />}
              size="large"
              style={{ flex: 1 }}
            />
            <Button
              type="primary"
              size="large"
              icon={<ScanOutlined />}
              onClick={handleScan}
              loading={loading}
              style={{ height: 44 }}
            >
              Scan Directory
            </Button>
          </div>
        </div>
      )}

      {/* Step 2: Scanning */}
      {step === 'scanning' && (
        <div className={styles['batch-loading']}>
          <SyncOutlined spin className={styles['batch-loading-icon']} />
          <p className={styles['batch-step-description']}>Scanning directory for documents…</p>
        </div>
      )}

      {/* Step 3: Confirm and start */}
      {step === 'confirm' && scanResult && (
        <div className={styles['batch-step-body']}>
          <Alert
            type="info"
            message={`Found ${scanResult.total} supported documents in ${directory}`}
            description="Choose a processing mode. Fast mode is recommended for batch imports."
          />

          <div className={styles['batch-mode-card']}>
            <div className={styles['batch-mode-card-title']}>Processing Mode</div>
            <Radio.Group
              value={processingMode}
              onChange={(e) => setProcessingMode(e.target.value)}
              style={{ width: '100%' }}
            >
              <div className={styles['batch-mode-list']}>
                <Radio value="fast">
                  <div className={styles['batch-mode-row']}>
                    <ThunderboltOutlined className={styles['icon-fast']} />
                    <span>Fast</span>
                    <span className={styles.muted}>— Vector only (~1-2 min/doc)</span>
                  </div>
                  <div className={styles['batch-mode-desc']}>
                    Parse + chunk + embed. No knowledge graph. Best for quick search.
                  </div>
                </Radio>
                <Radio value="standard">
                  <div className={styles['batch-mode-row']}>
                    <CloudUploadOutlined className={styles['icon-standard']} />
                    <span>Standard</span>
                    <span className={styles.muted}>— KG + Vector (~3-8 min/doc)</span>
                  </div>
                  <div className={styles['batch-mode-desc']}>
                    Parse + knowledge graph + vector. Good balance of speed and quality.
                  </div>
                </Radio>
                <Radio value="full">
                  <div className={styles['batch-mode-row']}>
                    <EyeOutlined className={styles['icon-full']} />
                    <span>Full</span>
                    <span className={styles.muted}>— KG + Vector + Multimodal (~10-30 min/doc)</span>
                  </div>
                  <div className={styles['batch-mode-desc']}>
                    Full pipeline with image/table/equation analysis. Best quality, slowest.
                  </div>
                </Radio>
              </div>
            </Radio.Group>
          </div>

          <Table
            columns={scanColumns}
            dataSource={scanResult.files}
            rowKey="file_path"
            pagination={{ pageSize: 10 }}
            size="small"
          />
          <div className={styles['batch-action-row']}>
            <Button onClick={() => setStep('input')}>Back</Button>
            <Button
              type="primary"
              icon={<ThunderboltOutlined />}
              onClick={handleStart}
              loading={loading}
              disabled={scanResult.total === 0}
            >
              Start Processing ({scanResult.total} docs, {processingMode} mode)
            </Button>
          </div>
        </div>
      )}

      {/* Step 4: Processing */}
      {(step === 'processing' || step === 'done') && batchJob && (
        <div className={styles['batch-step-body']}>
          {/* Overall progress */}
          <div>
            <div className={styles['batch-progress-summary']}>
              <span className={styles['batch-progress-title']}>Overall Progress</span>
              <div className={styles['batch-progress-stats']}>
                <Tag color="success">{batchJob.completed} completed</Tag>
                <Tag color="error">{batchJob.failed} failed</Tag>
                <Tag>{batchJob.skipped} skipped</Tag>
                <Tag color="processing">{batchJob.total - batchJob.completed - batchJob.failed - batchJob.skipped} remaining</Tag>
              </div>
            </div>
            <Progress
              percent={batchJob.progress_percent}
              status={step === 'done' ? (batchJob.failed > 0 ? 'exception' : 'success') : 'active'}
            />
          </div>

          <Table
            columns={processColumns}
            dataSource={batchJob.documents}
            rowKey="doc_id"
            pagination={{ pageSize: 10 }}
            size="small"
          />

          <div className={styles['batch-action-row']}>
            {step === 'processing' && (
              <Button danger icon={<StopOutlined />} onClick={handleCancel}>
                Cancel Batch
              </Button>
            )}
            {step === 'done' && (
              <Button type="primary" onClick={handleClose}>
                Close
              </Button>
            )}
          </div>
        </div>
      )}
    </Modal>
  );
};

export { BatchImportModal };
export default BatchImportModal;
