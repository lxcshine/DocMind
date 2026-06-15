import React from 'react';
import { Tag } from 'antd';
import {
  CheckCircleOutlined,
  SyncOutlined,
  CloseCircleOutlined,
  PauseCircleOutlined,
  InboxOutlined,
} from '@ant-design/icons';

interface StatusTagProps {
  status: string;
  progress?: number;
}

/**
 * Unified status tag component for document/chat states.
 */
const StatusTag: React.FC<StatusTagProps> = ({ status, progress }) => {
  const config: Record<string, { color: string; icon: React.ReactNode; text: string }> = {
    completed: { color: 'success', icon: <CheckCircleOutlined />, text: 'Completed' },
    processing: { color: 'processing', icon: <SyncOutlined spin />, text: 'Processing' },
    failed: { color: 'error', icon: <CloseCircleOutlined />, text: 'Failed' },
    uploaded: { color: 'default', icon: <InboxOutlined />, text: 'Uploaded' },
    pending: { color: 'warning', icon: <PauseCircleOutlined />, text: 'Pending' },
  };

  const { color, icon, text } = config[status] || config.uploaded;

  return (
    <Tag color={color} icon={icon}>
      {text}
      {progress !== undefined && progress > 0 && ` (${progress}%)`}
    </Tag>
  );
};

export default StatusTag;
export { StatusTag };
