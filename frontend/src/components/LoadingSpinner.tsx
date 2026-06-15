import React from 'react';
import { Spin } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

interface LoadingSpinnerProps {
  size?: 'small' | 'default' | 'large';
  text?: string;
  fullScreen?: boolean;
}

/**
 * Consistent loading spinner component.
 *
 * Note: in antd v5+, `<Spin tip=…>` only renders the tip text inside
 * the `nest` pattern (Spin with children that get dimmed) or the
 * `fullscreen` pattern. We avoid the deprecation by rendering the
 * spinner and the tip as sibling elements ourselves.
 */
const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'large',
  text = 'Loading...',
  fullScreen = false,
}) => {
  const fontSize = size === 'large' ? 32 : size === 'small' ? 16 : 24;
  const indicator = <LoadingOutlined style={{ fontSize }} spin />;
  const wrapperStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    ...(fullScreen
      ? { width: '100vw', height: '100vh', background: 'rgba(255, 255, 255, 0.8)' }
      : { padding: 'var(--space-6)' }),
  };

  return (
    <div style={wrapperStyle}>
      <Spin indicator={indicator} size={size} />
      {text && (
        <div
          style={{
            color: 'var(--color-text-tertiary)',
            fontSize: 'var(--text-sm)',
            fontWeight: 500,
          }}
        >
          {text}
        </div>
      )}
    </div>
  );
};

export default LoadingSpinner;
export { LoadingSpinner };
