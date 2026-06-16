import React, { useState, useMemo } from 'react';
import { Layout, Menu, Dropdown, Tooltip, Spin } from 'antd';
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
  BellOutlined,
  GithubOutlined,
  ThunderboltOutlined,
  ApartmentOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import type { MenuProps } from 'antd';
import './MainLayout.css';

const { Sider, Header, Content } = Layout;

interface NavItem {
  key: string;
  label: string;
  description?: string;
  icon: React.ReactNode;
}

const PRIMARY_NAV: NavItem[] = [
  { key: '/chat',       label: 'Chat',           description: 'Talk to your knowledge base', icon: <MessageOutlined /> },
  { key: '/knowledge',  label: 'Knowledge Base', description: 'Documents & ingestion',      icon: <BookOutlined /> },
  { key: '/search',     label: 'Search',         description: 'Full-text + semantic',       icon: <SearchOutlined /> },
];

const SECONDARY_NAV: NavItem[] = [
  { key: '/memory', label: 'Memory', description: 'Long-term context', icon: <HistoryOutlined /> },
  { key: '/ocr',    label: 'OCR',    description: 'Extract text from scans', icon: <ScanOutlined /> },
  { key: '/knowledge-graph', label: 'Knowledge Graph', description: 'Visualize entities & relations', icon: <ApartmentOutlined /> },
];

const routeMeta: Record<string, { title: string; eyebrow: string; icon: React.ReactNode }> = {
  '/chat':      { title: 'Chat',           eyebrow: 'Conversation',  icon: <MessageOutlined /> },
  '/knowledge': { title: 'Knowledge Base', eyebrow: 'Library',       icon: <BookOutlined /> },
  '/search':    { title: 'Search',         eyebrow: 'Discovery',     icon: <SearchOutlined /> },
  '/memory':    { title: 'Memory',         eyebrow: 'Long-term',     icon: <HistoryOutlined /> },
  '/ocr':       { title: 'OCR',            eyebrow: 'Extraction',    icon: <ScanOutlined /> },
  '/knowledge-graph': { title: 'Knowledge Graph', eyebrow: 'Visualization', icon: <ApartmentOutlined /> },
};

const MainLayout: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [isNavigating, setIsNavigating] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const userMenuItems: MenuProps['items'] = [
    { key: 'profile',  icon: <UserOutlined />,    label: 'Profile' },
    { key: 'settings', icon: <SettingOutlined />, label: 'Settings' },
    { type: 'divider' },
    { key: 'logout',   icon: <LogoutOutlined />,  label: 'Sign out', danger: true },
  ];

  const handleNewChat = () => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    navigate(`/chat?conversationId=${newSessionId}&isNew=true`);
  };

  // Match the deepest path (handles `/chat?...`)
  const activePath = useMemo(() => {
    const path = location.pathname;
    const known = Object.keys(routeMeta).sort((a, b) => b.length - a.length);
    return known.find(k => path === k || path.startsWith(k + '/')) || '/chat';
  }, [location.pathname]);

  const current = routeMeta[activePath];

  const handleMenuClick = ({ key }: { key: string }) => {
    setIsNavigating(true);
    navigate(key);
    // small artificial delay so the transition is visible
    setTimeout(() => setIsNavigating(false), 320);
  };

  return (
    <Layout className="main-layout" hasSider>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        className="layout-sider"
        width={260}
        collapsedWidth={68}
        theme="light"
      >
        <div className="logo-container">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          {!collapsed && <span className="logo-text">DocMind</span>}
        </div>

        <div className="sidebar-menu-wrap">
          {!collapsed && <div className="sidebar-section-label">Workspace</div>}
          <Menu
            mode="inline"
            selectedKeys={[activePath]}
            items={PRIMARY_NAV.map(i => ({ key: i.key, icon: i.icon, label: i.label }))}
            onClick={handleMenuClick}
            className="sidebar-menu"
          />

          {!collapsed && <div className="sidebar-section-label">Tools</div>}
          <Menu
            mode="inline"
            selectedKeys={[activePath]}
            items={SECONDARY_NAV.map(i => ({ key: i.key, icon: i.icon, label: i.label }))}
            onClick={handleMenuClick}
            className="sidebar-menu"
          />
        </div>

        <div className="sidebar-footer">
          <Tooltip title={collapsed ? 'New Chat' : ''} placement="right">
            <button className="new-chat-btn" onClick={handleNewChat}>
              {collapsed ? <PlusOutlined /> : (
                <>
                  <ThunderboltOutlined />
                  <span>New Chat</span>
                </>
              )}
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
              aria-label="Toggle sidebar"
            >
              {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            </button>
            <div className="breadcrumb">
              <span className="breadcrumb-icon">{current?.icon}</span>
              <span style={{ color: 'var(--color-text-tertiary)' }}>{current?.eyebrow}</span>
              <span style={{ color: 'var(--color-text-tertiary)' }}>/</span>
              <span className="breadcrumb-title">{current?.title}</span>
            </div>
          </div>

          <div className="header-right">
            <Tooltip title="Notifications">
              <button className="header-icon-btn" aria-label="Notifications">
                <BellOutlined />
              </button>
            </Tooltip>
            <Tooltip title="Source">
              <button
                className="header-icon-btn"
                aria-label="Source"
                onClick={() => window.open('https://github.com/lxcshine/DocMind', '_blank')}
              >
                <GithubOutlined />
              </button>
            </Tooltip>
            <Dropdown menu={{ items: userMenuItems }} placement="bottomRight" trigger={['click']}>
              <div className="user-avatar" role="button" tabIndex={0}>
                <div className="avatar-bubble">DM</div>
                <span className="username">User</span>
              </div>
            </Dropdown>
          </div>
        </Header>

        <Content className="layout-content">
          <Outlet />
          {isNavigating && (
            <div className="nav-overlay">
              <Spin size="large" tip="Loading…">
                <div style={{ minWidth: 200, minHeight: 120, padding: 24 }} />
              </Spin>
            </div>
          )}
        </Content>
      </Layout>
    </Layout>
  );
};

export default MainLayout;
