import React from 'react';
import { Empty, Button } from 'antd';
import { InboxOutlined, FileTextOutlined, CommentOutlined } from '@ant-design/icons';

interface EmptyStateProps {
  type?: 'documents' | 'messages' | 'default';
  title?: string;
  description?: string;
  action?: {
    text: string;
    onClick: () => void;
  };
}

/**
 * Consistent empty state component for different contexts.
 */
const EmptyState: React.FC<EmptyStateProps> = ({
  type = 'default',
  title,
  description,
  action,
}) => {
  const defaults: Record<string, { icon: React.ReactNode; title: string; description: string }> = {
    documents: {
      icon: <InboxOutlined style={{ fontSize: 48, color: '#bfbfbf' }} />,
      title: 'No documents yet',
      description: 'Upload your first document to get started',
    },
    messages: {
      icon: <CommentOutlined style={{ fontSize: 48, color: '#bfbfbf' }} />,
      title: 'No messages yet',
      description: 'Start a conversation to get answers from your documents',
    },
    default: {
      icon: <FileTextOutlined style={{ fontSize: 48, color: '#bfbfbf' }} />,
      title: 'No data',
      description: 'There is no data to display',
    },
  };

  const { icon, title: defaultTitle, description: defaultDescription } = defaults[type];

  return (
    <Empty
      image={null}
      description={
        <div>
          <div style={{ marginBottom: 8 }}>{icon}</div>
          <div style={{ fontWeight: 500, fontSize: 16, color: 'rgba(0,0,0,0.85)' }}>
            {title || defaultTitle}
          </div>
          <div style={{ fontSize: 13, color: 'rgba(0,0,0,0.45)', marginTop: 4 }}>
            {description || defaultDescription}
          </div>
          {action && (
            <Button type="primary" onClick={action.onClick} style={{ marginTop: 16 }}>
              {action.text}
            </Button>
          )}
        </div>
      }
    />
  );
};

export default EmptyState;
export { EmptyState };
