import React, { useState, useRef } from 'react';
import { Upload, Button, Card, Progress, Space, Tag, Empty, Spin, message, Collapse, Tooltip } from 'antd';
import {
  ScanOutlined,
  FilePdfOutlined,
  FileImageOutlined,
  FilePptOutlined,
  UploadOutlined,
  LoadingOutlined,
  CopyOutlined,
  DownloadOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons';
import type { UploadFile, RcFile } from 'antd/es/upload';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';

const { Panel } = Collapse;

const API_BASE_URL = '/api';

interface OCRResult {
  doc_id: string;
  filename: string;
  total_pages: number;
  raw_text: string;
  ocr_text: string;
  raw_char_count: number;
  corrected_char_count: number;
  llm_corrected: boolean;
  status: string;
}

interface FileTypeInfo {
  ext: string;
  icon: React.ReactNode;
  color: string;
}

const FILE_TYPE_MAP: Record<string, FileTypeInfo> = {
  '.pdf': { ext: 'PDF', icon: <FilePdfOutlined />, color: '#f5222d' },
  '.png': { ext: 'PNG', icon: <FileImageOutlined />, color: '#52c41a' },
  '.jpg': { ext: 'JPG', icon: <FileImageOutlined />, color: '#52c41a' },
  '.jpeg': { ext: 'JPEG', icon: <FileImageOutlined />, color: '#52c41a' },
  '.tiff': { ext: 'TIFF', icon: <FileImageOutlined />, color: '#1677ff' },
  '.bmp': { ext: 'BMP', icon: <FileImageOutlined />, color: '#722ed1' },
  '.ppt': { ext: 'PPT', icon: <FilePptOutlined />, color: '#fa8c16' },
  '.pptx': { ext: 'PPTX', icon: <FilePptOutlined />, color: '#fa8c16' },
};

const OCR: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<UploadFile | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState('');
  const [stageProgress, setStageProgress] = useState(0);
  const [stageLabel, setStageLabel] = useState('');
  const [totalPages, setTotalPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [rawChars, setRawChars] = useState(0);
  const [correctedChars, setCorrectedChars] = useState(0);
  const [result, setResult] = useState<OCRResult | null>(null);
  const [error, setError] = useState('');
  const [isCorrecting, setIsCorrecting] = useState(false);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const rawFileRef = useRef<RcFile | null>(null);
  const currentDocIdRef = useRef<string>('');

  const getFileTypeInfo = (filename: string): FileTypeInfo => {
    const ext = filename.toLowerCase().substring(filename.lastIndexOf('.'));
    return FILE_TYPE_MAP[ext] || { ext: 'FILE', icon: <UploadOutlined />, color: '#8c8c8c' };
  };

  const resetState = () => {
    setUploadProgress(0);
    setCurrentStage('');
    setStageProgress(0);
    setStageLabel('');
    setTotalPages(0);
    setCurrentPage(0);
    setRawChars(0);
    setCorrectedChars(0);
    setResult(null);
    setError('');
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const handleProcess = async () => {
    if (!uploadedFile) {
      message.warning('Please upload a file first');
      return;
    }

    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    resetState();
    setIsProcessing(true);
    setUploadProgress(100);
    setCurrentStage('upload');
    setStageLabel('Uploading file...');

    try {
      const formData = new FormData();
      const rawFile = rawFileRef.current;
      if (!rawFile) {
        throw new Error('File not available');
      }
      formData.append('file', rawFile, uploadedFile.name);

      const response = await fetch(`${API_BASE_URL}/ocr/process-poll`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || 'OCR request failed');
      }

      const { doc_id } = await response.json();

      currentDocIdRef.current = doc_id;

      setCurrentStage('processing');
      setStageLabel('Processing started...');

      pollTimerRef.current = setInterval(async () => {
        try {
          const progressRes = await fetch(`${API_BASE_URL}/ocr/progress/${doc_id}`);
          if (!progressRes.ok) return;

          const p = await progressRes.json();

          setStageProgress(p.progress || 0);
          setStageLabel(p.current_step || '');

          if (p.status === 'completed') {
            if (pollTimerRef.current) {
              clearInterval(pollTimerRef.current);
              pollTimerRef.current = null;
            }

            const resultRes = await fetch(`${API_BASE_URL}/ocr/result/${doc_id}`);
            if (resultRes.ok) {
              const resultData = await resultRes.json();
              setResult(resultData);
              setCurrentStage('done');
              setStageProgress(100);
              setStageLabel('OCR completed');
              setTotalPages(resultData.total_pages || 0);
              setRawChars(resultData.raw_char_count || 0);
              setCorrectedChars(resultData.corrected_char_count || 0);
              message.success('OCR completed! Click "AI Correction" to enhance.');
            }
            setIsProcessing(false);
          } else if (p.status === 'failed') {
            if (pollTimerRef.current) {
              clearInterval(pollTimerRef.current);
              pollTimerRef.current = null;
            }
            setError(p.error || 'OCR processing failed');
            setCurrentStage('error');
            message.error(p.error || 'OCR processing failed');
            setIsProcessing(false);
          }
        } catch {
          // polling errors are expected occasionally, ignore
        }
      }, 1000);

    } catch (err: any) {
      setError(err.message || 'OCR processing failed');
      setCurrentStage('error');
      message.error('OCR failed. Check if backend is running.');
      setIsProcessing(false);
    }
  };

  const handleCopy = () => {
    if (result?.ocr_text) {
      navigator.clipboard.writeText(result.ocr_text);
      message.success('Copied to clipboard');
    }
  };

  const handleDownload = () => {
    if (result?.ocr_text) {
      const blob = new Blob([result.ocr_text], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${result.filename.replace(/\.[^.]+$/, '')}_ocr.md`;
      a.click();
      URL.revokeObjectURL(url);
      message.success('Downloaded');
    }
  };

  const handleClear = () => {
    setUploadedFile(null);
    rawFileRef.current = null;
    resetState();
  };

  const handleLLMCorrect = async () => {
    const docId = currentDocIdRef.current;
    if (!docId) {
      message.warning('No document to correct');
      return;
    }

    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }

    setIsCorrecting(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/ocr/correct/${docId}`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || 'Correction request failed');
      }

      setCurrentStage('llm');
      setStageLabel('LLM correction in progress...');
      setStageProgress(0);

      pollTimerRef.current = setInterval(async () => {
        try {
          const progressRes = await fetch(`${API_BASE_URL}/ocr/progress/${docId}`);
          if (!progressRes.ok) return;

          const p = await progressRes.json();

          setStageProgress(p.progress || 0);
          setStageLabel(p.current_step || '');

          if (p.status === 'completed') {
            if (pollTimerRef.current) {
              clearInterval(pollTimerRef.current);
              pollTimerRef.current = null;
            }

            const resultRes = await fetch(`${API_BASE_URL}/ocr/result/${docId}`);
            if (resultRes.ok) {
              const resultData = await resultRes.json();
              setResult(resultData);
              setCorrectedChars(resultData.corrected_char_count || 0);
              setCurrentStage('done');
              setStageProgress(100);
              setStageLabel('AI correction completed');
              message.success('AI correction completed!');
            }
            setIsCorrecting(false);
          } else if (p.status === 'failed') {
            if (pollTimerRef.current) {
              clearInterval(pollTimerRef.current);
              pollTimerRef.current = null;
            }
            setError(p.error || 'LLM correction failed');
            message.error(p.error || 'LLM correction failed');
            setIsCorrecting(false);
          }
        } catch {
          // polling errors are expected occasionally, ignore
        }
      }, 1000);

    } catch (err: any) {
      setError(err.message || 'LLM correction failed');
      message.error('LLM correction failed. Check API quota or network.');
      setIsCorrecting(false);
    }
  };

  const fileTypeInfo = uploadedFile ? getFileTypeInfo(uploadedFile.name) : null;
  const overallProgress = currentStage === 'done' ? 100
    : currentStage === 'llm' ? stageProgress
    : currentStage === 'ocr' ? stageProgress
    : currentStage === 'preprocess' ? stageProgress
    : currentStage === 'convert' ? stageProgress
    : currentStage === 'upload' ? 10
    : currentStage === 'processing' ? stageProgress
    : 0;

  const stageDisplay = currentStage === 'convert' ? 'Converting'
    : currentStage === 'preprocess' ? 'Preprocessing'
    : currentStage === 'ocr' ? 'OCR Extracting'
    : currentStage === 'llm' ? 'LLM Smart Correction'
    : currentStage === 'done' ? 'Completed'
    : currentStage === 'processing' ? 'Processing'
    : currentStage === 'upload' ? 'Uploading'
    : 'Ready';

  const stageColor = currentStage === 'done' ? '#52c41a'
    : currentStage === 'llm' ? '#722ed1'
    : currentStage === 'ocr' ? '#1677ff'
    : currentStage === 'processing' ? '#1677ff'
    : currentStage === 'error' ? '#f5222d'
    : '#fa8c16';

  return (
    <div className="ocr-page">
      <div className="page-header">
        <h1 className="page-title">
          <ScanOutlined style={{ marginRight: 12 }} />
          Intelligent OCR
        </h1>
        <p className="page-description">
          Upload PDF, images (PNG/JPG/TIFF/BMP), or PPT files -- Tesseract OCR extraction + optional AI correction
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, minHeight: 500 }}>
        {/* Left: Upload & Progress */}
        <div>
          <Card
            className="upload-card"
            style={{
              border: uploadedFile ? '2px solid #d9d9d9' : '2px dashed #d9d9d9',
              borderRadius: 8,
              height: '100%',
            }}
          >
            {!uploadedFile ? (
              <Upload.Dragger
                name="file"
                multiple={false}
                accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp,.ppt,.pptx"
                beforeUpload={(file) => {
                  const allowed = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.ppt', '.pptx'];
                  const ext = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
                  if (!allowed.includes(ext)) {
                    message.error(`Unsupported file type: ${ext}`);
                    return Upload.LIST_IGNORE;
                  }
                  if (file.size > 100 * 1024 * 1024) {
                    message.error('File too large (max 100MB)');
                    return Upload.LIST_IGNORE;
                  }
                  setUploadedFile({ ...file, uid: '-1', name: file.name, size: file.size, originFileObj: file } as UploadFile);
                  rawFileRef.current = file as RcFile;
                  return false;
                }}
                showUploadList={false}
                disabled={isProcessing}
                style={{ padding: '40px 20px' }}
              >
                <p className="ant-upload-drag-icon" style={{ fontSize: 48, marginBottom: 16 }}>
                  <ScanOutlined style={{ color: '#1677ff' }} />
                </p>
                <p className="ant-upload-text" style={{ fontSize: 16, fontWeight: 500 }}>
                  Click or drag file here
                </p>
                <p className="ant-upload-hint" style={{ color: '#8c8c8c' }}>
                  Supports PDF, PNG, JPG, JPEG, TIFF, BMP, PPT, PPTX (max 100MB)
                </p>
              </Upload.Dragger>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                {/* File Info */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: 16,
                  background: '#fafafa',
                  borderRadius: 8,
                  marginBottom: 16,
                }}>
                  <span style={{ fontSize: 32, marginRight: 12, color: fileTypeInfo?.color }}>
                    {fileTypeInfo?.icon}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 600, fontSize: 15 }}>{uploadedFile.name}</div>
                    <Space size={4}>
                      <Tag color={fileTypeInfo?.color}>{fileTypeInfo?.ext}</Tag>
                      <span style={{ color: '#8c8c8c', fontSize: 12 }}>
                        {(uploadedFile.size! / 1024 / 1024).toFixed(1)} MB
                      </span>
                    </Space>
                  </div>
                  {!isProcessing && (
                    <Button
                      type="text"
                      danger
                      icon={<DeleteOutlined />}
                      onClick={handleClear}
                    />
                  )}
                </div>

                {/* Progress */}
                {isProcessing ? (
                  <div style={{ flex: 1 }}>
                    <div style={{ marginBottom: 16 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <span style={{ fontWeight: 600 }}>
                          <Tag color={stageColor}>{stageDisplay}</Tag>
                        </span>
                        <span style={{ color: '#8c8c8c' }}>{overallProgress}%</span>
                      </div>
                      <Progress
                        percent={overallProgress}
                        strokeColor={stageColor}
                        status={error ? 'exception' : currentStage === 'done' ? 'success' : 'active'}
                        showInfo={false}
                      />
                      <div style={{ color: '#8c8c8c', fontSize: 13, marginTop: 8 }}>
                        {stageLabel}
                      </div>
                    </div>

                    {/* Per-stage details */}
                    <Space direction="vertical" style={{ width: '100%' }}>
                      {totalPages > 0 && (
                        <div style={{ fontSize: 13, color: '#8c8c8c' }}>
                          Pages: {currentPage} / {totalPages}
                          {currentPage > 0 && (
                            <Progress
                              percent={Math.round((currentPage / totalPages) * 100)}
                              size="small"
                              style={{ marginTop: 4 }}
                            />
                          )}
                        </div>
                      )}

                      <div style={{ fontSize: 13, color: '#8c8c8c' }}>
                        {currentStage === 'convert' && 'Converting document to images...'}
                        {currentStage === 'preprocess' && 'Enhancing image quality for better OCR...'}
                        {currentStage === 'ocr' && 'Tesseract OCR extracting text per page...'}
                        {currentStage === 'llm' && 'LLM analyzing and correcting OCR errors...'}
                        {currentStage === 'done' && (
                          <span>
                            <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 4 }} />
                            All done! {correctedChars > 0 && `${correctedChars.toLocaleString()} characters`}
                          </span>
                        )}
                        {currentStage === 'error' && (
                          <span>
                            <CloseCircleOutlined style={{ color: '#f5222d', marginRight: 4 }} />
                            {error}
                          </span>
                        )}
                      </div>
                    </Space>

                    <div style={{ marginTop: 24 }}>
                      <Button
                        onClick={() => {
                          if (pollTimerRef.current) {
                            clearInterval(pollTimerRef.current);
                            pollTimerRef.current = null;
                          }
                          setIsProcessing(false);
                          setError('Cancelled');
                          setCurrentStage('');
                        }}
                        danger
                        size="small"
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Button
                      type="primary"
                      size="large"
                      icon={<ScanOutlined />}
                      onClick={handleProcess}
                      style={{ height: 48, paddingInline: 32, fontSize: 16 }}
                    >
                      Start OCR
                    </Button>
                  </div>
                )}
              </div>
            )}
          </Card>
        </div>

        {/* Right: Result */}
        <div>
          <Card
            title={
              <Space>
                <span>OCR Result</span>
                {result && (
                  <Tag color="green">
                    {result.total_pages} page{result.total_pages > 1 ? 's' : ''}
                  </Tag>
                )}
              </Space>
            }
            extra={
              result && (
                <Space>
                  <Tooltip title="Copy to clipboard">
                    <Button icon={<CopyOutlined />} size="small" onClick={handleCopy} />
                  </Tooltip>
                  <Tooltip title="Download as Markdown">
                    <Button icon={<DownloadOutlined />} size="small" onClick={handleDownload} />
                  </Tooltip>
                </Space>
              )
            }
            style={{ borderRadius: 8, height: '100%' }}
            bodyStyle={{ padding: result ? 16 : 32, height: 'calc(100% - 57px)', overflow: 'auto' }}
          >
            {!result && !isProcessing && (
              <Empty
                image={<ScanOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />}
                description={
                  <span style={{ color: '#bfbfbf' }}>
                    Upload a file and start OCR to see results here
                  </span>
                }
              />
            )}

            {isProcessing && !result && (
              <div style={{ textAlign: 'center', padding: 40 }}>
                <Spin indicator={<LoadingOutlined style={{ fontSize: 32 }} spin />} />
                <div style={{ marginTop: 16, color: '#8c8c8c' }}>
                  Processing your document...
                </div>
              </div>
            )}

            {result && (
              <div>
                {/* Stats */}
                <Collapse
                  size="small"
                  ghost
                  style={{ marginBottom: 16 }}
                  items={[{
                    key: 'stats',
                    label: 'OCR Statistics',
                    children: (
                      <Space direction="vertical" size={4}>
                        <div>Raw OCR characters: <strong>{result.raw_char_count.toLocaleString()}</strong></div>
                        {result.llm_corrected ? (
                          <>
                            <div>After LLM correction: <strong>{result.corrected_char_count.toLocaleString()}</strong></div>
                            <div>Correction ratio: <strong>
                              {result.raw_char_count > 0
                                ? ((result.corrected_char_count / result.raw_char_count - 1) * 100).toFixed(1)
                                : 0}%
                            </strong></div>
                          </>
                        ) : (
                          <div style={{ color: '#fa8c16' }}>
                            <Tag color="orange">LLM correction not applied</Tag>
                          </div>
                        )}
                        <div>Total pages: <strong>{result.total_pages}</strong></div>
                      </Space>
                    ),
                  }]}
                />

                {/* AI Correction button */}
                {!result.llm_corrected && !isCorrecting && (
                  <div style={{ marginBottom: 12 }}>
                    <Button
                      type="primary"
                      icon={<ScanOutlined />}
                      onClick={handleLLMCorrect}
                      style={{ background: '#722ed1', borderColor: '#722ed1' }}
                    >
                      AI 鏅鸿兘绾犻敊
                    </Button>
                    <span style={{ marginLeft: 8, color: '#8c8c8c', fontSize: 12 }}>
                      Fix OCR errors and reorganize text structure
                    </span>
                  </div>
                )}

                {isCorrecting && (
                  <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Spin size="small" />
                    <span style={{ color: '#722ed1', fontSize: 13 }}>AI correction running...</span>
                  </div>
                )}

                {result.llm_corrected && (
                  <Tag color="purple" style={{ marginBottom: 12 }}>LLM Corrected</Tag>
                )}

                {/* Content */}
                <div className="ocr-result-content" style={{ lineHeight: 1.7 }}>
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex, rehypeRaw]}
                    components={{
                      table: ({ children }) => (
                        <table style={{
                          borderCollapse: 'collapse',
                          width: '100%',
                          margin: '8px 0',
                        }}>
                          {children}
                        </table>
                      ),
                      th: ({ children }) => (
                        <th style={{
                          border: '1px solid #e8e8e8',
                          padding: '6px 12px',
                          background: '#fafafa',
                          fontWeight: 600,
                        }}>
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td style={{
                          border: '1px solid #e8e8e8',
                          padding: '6px 12px',
                        }}>
                          {children}
                        </td>
                      ),
                    }}
                  >
                    {result.ocr_text}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            {error && !isProcessing && (
              <div style={{ padding: 16, background: '#fff2f0', borderRadius: 8 }}>
                <CloseCircleOutlined style={{ color: '#f5222d', marginRight: 8 }} />
                <span style={{ color: '#f5222d' }}>{error}</span>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default OCR;
