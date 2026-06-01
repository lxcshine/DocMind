import React, { useState } from 'react';
import { Layout, Menu, Avatar, Dropdown, Tooltip, Spin } from 'antd';
import {
  BookOutlined,
  MessageOutlined,
  SearchOutlined,
  HistoryOutlined,
  ScanOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  UserOutlined,
  SettingOutlined,
  LogoutOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import type { MenuProps } from 'antd';
import './MainLayout.css';

const { Sider, Header, Content } = Layout;

const MainLayout: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [isNavigating, setIsNavigating] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems: MenuProps['items'] = [
    {
      key: '/knowledge',
      icon: <BookOutlined />,
      label: 'Knowledge Base',
    },
    {
      key: '/chat',
      icon: <MessageOutlined />,
      label: 'Chat',
    },
    {
      key: '/search',
      icon: <SearchOutlined />,
      label: 'Search',
    },
    {
      key: '/memory',
      icon: <HistoryOutlined />,
      label: 'Memory',
    },
    {
      key: '/ocr',
      icon: <ScanOutlined />,
      label: 'OCR',
    },
  ];

  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: 'Profile',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Settings',
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
      danger: true,
    },
  ];

  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key);
  };

  const handleNewChat = () => {
    // Use URL search params like RAGFlow for reliable navigation
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    navigate(`/chat?conversationId=${newSessionId}&isNew=true`);
  };

  // Reset navigating state when route changes
  React.useEffect(() => {
    if (isNavigating) {
      const timer = setTimeout(() => setIsNavigating(false), 500);
      return () => clearTimeout(timer);
    }
  }, [location.pathname, isNavigating]);

  return (
    <Layout className="main-layout">
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        className="layout-sider"
        width={240}
        theme="light"
      >
        <div className="logo-container">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          {!collapsed && <span className="logo-text">DocMind</span>}
        </div>

        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
          className="sidebar-menu"
        />

        <div className="sidebar-footer">
          <Tooltip title={collapsed ? 'New Chat' : ''} placement="right">
            <button className="new-chat-btn" onClick={handleNewChat}>
              <PlusOutlined />
              {!collapsed && <span>New Chat</span>}
            </button>
          </Tooltip>
        </div>
      </Sider>

      <Layout className="main-content">
        <Header className="layout-header">
          <div className="header-left">
            <button
              className="collapse-btn"
              onClick={() => setCollapsed(!collapsed)}
            >
              {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            </button>
          </div>

          <div className="header-right">
            <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
              <div className="user-avatar">
                <Avatar size="small" icon={<UserOutlined />} />
                <span className="username">User</span>
              </div>
            </Dropdown>
          </div>
        </Header>

        <Content className="layout-content">
          <Outlet />
          
          {/* Loading overlay during navigation */}
          {isNavigating && (
            <div style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(255, 255, 255, 0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 9999,
            }}>
              <Spin size="large" tip="Loading chat..." />
            </div>
          )}
        </Content>
      </Layout>
    </Layout>
  );
};

export default MainLayout;
