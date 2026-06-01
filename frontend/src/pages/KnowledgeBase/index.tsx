import React, { useState, useEffect, useCallback } from 'react';
import { Card, Upload, Button, Table, Tag, Space, message, Progress, Popconfirm, Modal } from 'antd';
import {
  InboxOutlined,
  FilePdfOutlined,
  FileTextOutlined,
  DeleteOutlined,
  CloudUploadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import type { UploadFile } from 'antd';

const { Dragger } = Upload;

const API_BASE_URL = 'http://localhost:8000/api';

interface Document {
  key: string;
  name: string;
  type: string;
  size: string;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  progress: number;
  current_step: string;
  sections: number;
  updatedAt: string;
}

const KnowledgeBase: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    fetchDocuments();
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/list`);
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

  const handleUpload = async (file: UploadFile) => {
    setUploading(true);
    
    const formData = new FormData();
    formData.append('file', file as any);

    try {
      console.log('Uploading file:', file.name, 'Size:', file.size);
      
      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      console.log('Upload response:', response.status, result);

      if (response.ok) {
        message.success(`File '${result.filename}' saved successfully! Click 'Add to KB' to process it.`);
        fetchDocuments();
      } else {
        message.error(`Upload failed: ${result.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Network error. Is backend running?';
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
        message.success('Document processing started!');
        
        if (!pollingInterval) {
          const interval = setInterval(() => {
            fetchDocuments();
          }, 2000);
          setPollingInterval(interval);
        }
      } else {
        message.error(`Failed to process: ${result.detail || 'Unknown error'}`);
      }
    } catch (error) {
      message.error('Failed to start processing');
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${docId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        message.success('Document deleted successfully');
        fetchDocuments();
      } else {
        message.error('Failed to delete document');
      }
    } catch (error) {
      message.error('Failed to delete document');
    }
  };

  const getStatusTag = (record: Document) => {
    switch (record.status) {
      case 'uploaded':
        return <Tag color="default" icon={<CheckCircleOutlined />}>Uploaded</Tag>;
      case 'processing':
        return <Tag color="processing" icon={<SyncOutlined spin />}>Processing</Tag>;
      case 'completed':
        return <Tag color="success" icon={<CheckCircleOutlined />}>In Knowledge Base</Tag>;
      case 'failed':
        return <Tag color="error" icon={<CloseCircleOutlined />}>Failed</Tag>;
      default:
        return <Tag>Unknown</Tag>;
    }
  };

  const columns = [
    {
      title: 'Document Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Document) => (
        <Space>
          {record.type === 'PDF' ? <FilePdfOutlined style={{ color: '#ff4d4f' }} /> : <FileTextOutlined />}
          <span>{text}</span>
        </Space>
      ),
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      width: 100,
    },
    {
      title: 'Size',
      dataIndex: 'size',
      key: 'size',
      width: 100,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 180,
      render: (_: string, record: Document) => getStatusTag(record),
    },
    {
      title: 'Progress',
      key: 'progress',
      width: 200,
      render: (_: any, record: Document) => {
        if (record.status === 'completed') {
          return <span style={{ color: '#52c41a', fontWeight: 500 }}>Completed</span>;
        }
        if (record.status === 'processing' || record.progress > 0) {
          return (
            <div>
              <Progress percent={record.progress} size="small" status="active" />
              {record.current_step && (
                <div style={{ fontSize: 12, color: '#8c8c8c', marginTop: 4 }}>
                  {record.current_step}
                </div>
              )}
            </div>
          );
        }
        if (record.status === 'failed') {
          return <span style={{ color: '#ff4d4f' }}>Failed</span>;
        }
        return '-';
      },
    },
    {
      title: 'Sections',
      dataIndex: 'sections',
      key: 'sections',
      width: 100,
      render: (sections: number) => sections > 0 ? sections : '-',
    },
    {
      title: 'Updated At',
      dataIndex: 'updatedAt',
      key: 'updatedAt',
      width: 180,
    },
    {
      title: 'Actions',
      key: 'action',
      width: 250,
      render: (_: any, record: Document) => (
        <Space>
          {record.status === 'uploaded' && (
            <Button 
              type="primary" 
              size="small" 
              icon={<CloudUploadOutlined />}
              onClick={() => handleAddToKB(record.key)}
            >
              Add to KB
            </Button>
          )}
          {record.status === 'processing' && (
            <Button 
              type="default" 
              size="small" 
              icon={<SyncOutlined spin />}
              disabled
            >
              Processing...
            </Button>
          )}
          {record.status === 'completed' && (
            <Tag color="success">Ready</Tag>
          )}
          {record.status === 'failed' && (
            <Button 
              type="primary" 
              size="small" 
              danger
              icon={<CloudUploadOutlined />}
              onClick={() => handleAddToKB(record.key)}
            >
              Retry
            </Button>
          )}
          <Popconfirm
            title="Delete this document?"
            description="This will remove the file and all processed sections"
            onConfirm={() => handleDelete(record.key)}
            okText="Yes"
            cancelText="No"
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              Delete
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div className="knowledge-base-page">
      <div className="page-header">
        <h2 className="page-title">Knowledge Base</h2>
        <p className="page-description">Upload academic documents, then add them to the knowledge base for RAG processing</p>
      </div>

      <Card title="Upload Documents" style={{ marginBottom: 24 }}>
        <Dragger
          accept=".pdf,.md,.txt"
          multiple
          beforeUpload={handleUpload}
          maxCount={10}
          disabled={uploading}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: 48, color: '#1677ff' }} />
          </p>
          <p className="ant-upload-text">Click or drag files to upload</p>
          <p className="ant-upload-hint">
            Support PDF, Markdown, TXT formats. Files are saved permanently. Click 'Add to KB' to process.
          </p>
        </Dragger>
      </Card>

      <Card 
        title="Document List" 
        extra={
          <Space>
            <Button type="primary" onClick={fetchDocuments}>Refresh</Button>
          </Space>
        }
      >
        <Table
          columns={columns}
          dataSource={documents}
          pagination={{ pageSize: 10 }}
          rowKey="key"
        />
      </Card>
    </div>
  );
};

export default KnowledgeBase;
