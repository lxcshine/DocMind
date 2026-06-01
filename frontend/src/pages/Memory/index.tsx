import React, { useState, useEffect, useCallback } from 'react';
import {
  Card, Table, Tag, Space, Button, Switch, Empty, Select, Modal,
  message, Popconfirm, Tabs, Checkbox, InputNumber, Slider, Spin, Tooltip, Typography, Descriptions,
} from 'antd';
import {
  DeleteOutlined,
  EyeOutlined,
  ClearOutlined,
  SettingOutlined,
  UnorderedListOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Text, Paragraph } = Typography;
const API_BASE_URL = 'http://localhost:8000/api';

interface MemoryTypeInfo {
  name: string;
  label: string;
  color: string;
}

interface MemoryConfig {
  active_types: string[];
  max_entries: number;
  max_tokens: number;
  temperature: number;
  forgetting_policy: string;
  system_prompt: string;
}

interface MemoryEntry {
  id: string;
  type: string;
  type_label: string;
  type_color: string;
  content: string;
  keywords: string[];
  source_session_id: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

const Memory: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'entries' | 'config'>('entries');
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [filterType, setFilterType] = useState<string>('all');
  const [enabledOnly, setEnabledOnly] = useState(false);
  const [loading, setLoading] = useState(false);

  const [config, setConfig] = useState<MemoryConfig>({
    active_types: ['raw', 'semantic', 'episodic', 'procedural'],
    max_entries: 200,
    max_tokens: 50000,
    temperature: 0.3,
    forgetting_policy: 'FIFO',
    system_prompt: '',
  });
  const [configLoading, setConfigLoading] = useState(false);
  const [types, setTypes] = useState<MemoryTypeInfo[]>([]);
  const [viewingEntry, setViewingEntry] = useState<MemoryEntry | null>(null);

  const fetchEntries = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filterType !== 'all') params.set('memory_type', filterType);
      if (enabledOnly) params.set('enabled_only', 'true');
      params.set('page', String(page));
      params.set('page_size', String(pageSize));

      const response = await fetch(`${API_BASE_URL}/memory/entries?${params}`);
      const data = await response.json();
      setEntries(data.entries || []);
      setTotal(data.total || 0);
    } catch (error) {
      console.error('Failed to fetch entries:', error);
      message.error('Failed to load memory entries');
    } finally {
      setLoading(false);
    }
  }, [filterType, enabledOnly, page, pageSize]);

  const fetchConfig = async () => {
    setConfigLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/memory/config`);
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error('Failed to fetch config:', error);
    } finally {
      setConfigLoading(false);
    }
  };

  const fetchTypes = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/memory/types`);
      const data = await response.json();
      setTypes(data.types || []);
    } catch (error) {
      console.error('Failed to fetch types:', error);
    }
  };

  useEffect(() => {
    fetchEntries();
  }, [fetchEntries]);

  useEffect(() => {
    if (activeTab === 'config') {
      fetchConfig();
      fetchTypes();
    }
  }, [activeTab]);

  const handleEnableToggle = async (entryId: string, enabled: boolean) => {
    try {
      const endpoint = enabled ? 'enable' : 'disable';
      const response = await fetch(`${API_BASE_URL}/memory/entries/${entryId}/${endpoint}`, {
        method: 'PUT',
      });
      if (response.ok) {
        message.success(enabled ? 'Entry enabled' : 'Entry disabled');
        fetchEntries();
      } else {
        message.error('Failed to update entry');
      }
    } catch (error) {
      message.error('Failed to update entry');
    }
  };

  const handleForgetEntry = async (entryId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/memory/entries/${entryId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        message.success('Entry forgotten');
        fetchEntries();
      } else {
        message.error('Failed to forget entry');
      }
    } catch (error) {
      message.error('Failed to forget entry');
    }
  };

  const handleClearAll = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/memory/clear`, {
        method: 'DELETE',
      });
      if (response.ok) {
        message.success('All memory cleared');
        fetchEntries();
      } else {
        message.error('Failed to clear memory');
      }
    } catch (error) {
      message.error('Failed to clear memory');
    }
  };

  const handleSaveConfig = async (partial: Partial<MemoryConfig>) => {
    try {
      const response = await fetch(`${API_BASE_URL}/memory/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(partial),
      });
      if (response.ok) {
        message.success('Configuration saved');
        fetchConfig();
      } else {
        message.error('Failed to save configuration');
      }
    } catch (error) {
      message.error('Failed to save configuration');
    }
  };

  const columns: ColumnsType<MemoryEntry> = [
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      width: 160,
      render: (type: string, record: MemoryEntry) => (
        <Tag color={record.type_color || 'default'}>{record.type_label}</Tag>
      ),
    },
    {
      title: 'Content',
      dataIndex: 'content',
      key: 'content',
      ellipsis: true,
      render: (text: string) => (
        <Tooltip title={text}>
          <Text ellipsis style={{ maxWidth: 400 }}>{text}</Text>
        </Tooltip>
      ),
    },
    {
      title: 'Keywords',
      dataIndex: 'keywords',
      key: 'keywords',
      width: 180,
      render: (keywords: string[]) => (
        <Space size={4} wrap>
          {(keywords || []).slice(0, 3).map(kw => (
            <Tag key={kw} color="blue" style={{ fontSize: 11 }}>{kw}</Tag>
          ))}
          {(keywords || []).length > 3 && (
            <Tag style={{ fontSize: 11 }}>+{keywords.length - 3}</Tag>
          )}
        </Space>
      ),
    },
    {
      title: 'Enabled',
      dataIndex: 'enabled',
      key: 'enabled',
      width: 90,
      align: 'center',
      render: (enabled: boolean, record: MemoryEntry) => (
        <Switch
          size="small"
          checked={enabled}
          onChange={(checked) => handleEnableToggle(record.id, checked)}
        />
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
    },
    {
      title: 'Actions',
      key: 'action',
      width: 120,
      render: (_: any, record: MemoryEntry) => (
        <Space size={4}>
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => setViewingEntry(record)}
          >
            View
          </Button>
          <Popconfirm
            title="Forget this memory?"
            description="This will permanently remove this memory entry"
            onConfirm={() => handleForgetEntry(record.id)}
            okText="Forget"
            cancelText="Cancel"
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div className="memory-page">
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h2 className="page-title">Memory Management</h2>
            <p className="page-description">
              Agent memory with LLM-driven knowledge extraction. Semantic, Episodic, and Procedural memories
              are automatically extracted from conversations and used to personalize future answers.
            </p>
          </div>
          <Popconfirm
            title="Clear all memory entries?"
            description="This action cannot be undone. All extracted memories will be lost."
            onConfirm={handleClearAll}
            okText="Clear All"
            cancelText="Cancel"
          >
            <Button danger icon={<ClearOutlined />}>
              Clear All Memory
            </Button>
          </Popconfirm>
        </div>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={(key) => setActiveTab(key as 'entries' | 'config')}
        items={[
          {
            key: 'entries',
            label: (
              <span><UnorderedListOutlined /> Memory Entries</span>
            ),
            children: (
              <>
                <Card style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
                    <span style={{ color: '#8c8c8c' }}>Type:</span>
                    <Select
                      value={filterType}
                      onChange={(v) => { setFilterType(v); setPage(1); }}
                      style={{ width: 180 }}
                      options={[
                        { value: 'all', label: 'All Types' },
                        ...types.map(t => ({ value: t.name, label: t.label })),
                      ]}
                    />
                    <Space>
                      <Switch
                        checked={enabledOnly}
                        onChange={setEnabledOnly}
                        checkedChildren="Enabled"
                        unCheckedChildren="All"
                      />
                      <span style={{ color: '#8c8c8c', fontSize: 13 }}>
                        Show enabled only
                      </span>
                    </Space>
                    <div style={{ flex: 1 }} />
                    <Button
                      icon={<SearchOutlined />}
                      onClick={fetchEntries}
                    >
                      Refresh
                    </Button>
                  </div>
                </Card>

                <Card
                  title={
                    <Space>
                      <UnorderedListOutlined />
                      <span>Memory Entries ({total})</span>
                    </Space>
                  }
                >
                  {entries.length > 0 ? (
                    <Table
                      columns={columns}
                      dataSource={entries}
                      pagination={{
                        current: page,
                        pageSize,
                        total,
                        onChange: (p, ps) => { setPage(p); setPageSize(ps); },
                        showSizeChanger: true,
                        showTotal: (t) => `${t} entries`,
                      }}
                      rowKey="id"
                      loading={loading}
                      size="middle"
                    />
                  ) : (
                    <Empty
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                      description={
                        <div>
                          <p style={{ marginBottom: 8, color: '#8c8c8c' }}>
                            No memory entries yet
                          </p>
                          <p style={{ fontSize: 12, color: '#bfbfbf' }}>
                            Memory entries are automatically extracted when you chat with the AI.
                            Start a conversation in the Chat page to build your memory.
                          </p>
                        </div>
                      }
                    />
                  )}
                </Card>

                <Modal
                  title="Memory Entry Details"
                  open={!!viewingEntry}
                  onCancel={() => setViewingEntry(null)}
                  footer={[
                    viewingEntry && (
                      <Switch
                        key="toggle"
                        checked={viewingEntry.enabled}
                        checkedChildren="Enabled"
                        unCheckedChildren="Disabled"
                        onChange={(checked) => {
                          handleEnableToggle(viewingEntry.id, checked);
                          setViewingEntry(null);
                        }}
                      />
                    ),
                    <Button key="close" onClick={() => setViewingEntry(null)}>
                      Close
                    </Button>,
                  ]}
                  width={600}
                >
                  {viewingEntry && (
                    <Descriptions column={1} bordered size="small">
                      <Descriptions.Item label="Type">
                        <Tag color={viewingEntry.type_color}>{viewingEntry.type_label}</Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label="Content">
                        <Paragraph style={{ margin: 0 }}>{viewingEntry.content}</Paragraph>
                      </Descriptions.Item>
                      <Descriptions.Item label="Keywords">
                        <Space wrap>
                          {(viewingEntry.keywords || []).map(kw => (
                            <Tag key={kw}>{kw}</Tag>
                          ))}
                        </Space>
                      </Descriptions.Item>
                      <Descriptions.Item label="Source Session">
                        {viewingEntry.source_session_id}
                      </Descriptions.Item>
                      <Descriptions.Item label="Created">
                        {viewingEntry.created_at}
                      </Descriptions.Item>
                      <Descriptions.Item label="Updated">
                        {viewingEntry.updated_at}
                      </Descriptions.Item>
                    </Descriptions>
                  )}
                </Modal>
              </>
            ),
          },
          {
            key: 'config',
            label: (
              <span><SettingOutlined /> Configuration</span>
            ),
            children: (
              <Spin spinning={configLoading}>
                <Card title={<span><SettingOutlined /> Memory Settings</span>} style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 24 }}>
                    <h4 style={{ marginBottom: 8 }}>Active Memory Types</h4>
                    <p style={{ fontSize: 13, color: '#8c8c8c', marginBottom: 12 }}>
                      Select which types of memory the system should extract from conversations.
                      Raw conversations are always saved.
                    </p>
                    <Space size={24} wrap>
                      {types.map(t => (
                        <Checkbox
                          key={t.name}
                          checked={config.active_types.includes(t.name)}
                          disabled={t.name === 'raw'}
                          onChange={(e) => {
                            const newTypes = e.target.checked
                              ? [...config.active_types, t.name]
                              : config.active_types.filter(tt => tt !== t.name);
                            handleSaveConfig({ active_types: newTypes });
                          }}
                        >
                          <Space size={4}>
                            <Tag color={t.color} style={{ marginRight: 0 }}>{t.name}</Tag>
                            <span style={{ fontSize: 13 }}>{t.label}</span>
                          </Space>
                        </Checkbox>
                      ))}
                    </Space>
                  </div>

                  <div style={{ marginBottom: 24 }}>
                    <h4 style={{ marginBottom: 8 }}>Max Entries</h4>
                    <p style={{ fontSize: 13, color: '#8c8c8c', marginBottom: 8 }}>
                      Maximum number of memory entries. Oldest entries are forgotten first (FIFO).
                    </p>
                    <Space>
                      <InputNumber
                        min={10}
                        max={1000}
                        value={config.max_entries}
                        onChange={(v) => handleSaveConfig({ max_entries: v ?? undefined })}
                        style={{ width: 120 }}
                      />
                      <span style={{ color: '#8c8c8c', fontSize: 13 }}>entries</span>
                    </Space>
                  </div>

                  <div style={{ marginBottom: 24 }}>
                    <h4 style={{ marginBottom: 8 }}>Max Tokens</h4>
                    <p style={{ fontSize: 13, color: '#8c8c8c', marginBottom: 8 }}>
                      Approximate total token budget for memory context.
                    </p>
                    <Space>
                      <InputNumber
                        min={5000}
                        max={200000}
                        step={5000}
                        value={config.max_tokens}
                        onChange={(v) => handleSaveConfig({ max_tokens: v ?? undefined })}
                        style={{ width: 120 }}
                      />
                      <span style={{ color: '#8c8c8c', fontSize: 13 }}>tokens</span>
                    </Space>
                  </div>

                  <div style={{ marginBottom: 24 }}>
                    <h4 style={{ marginBottom: 8 }}>Extraction Temperature</h4>
                    <p style={{ fontSize: 13, color: '#8c8c8c', marginBottom: 8 }}>
                      LLM temperature for knowledge extraction. Lower = more factual, higher = more creative.
                    </p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <Slider
                        min={0}
                        max={1}
                        step={0.05}
                        value={config.temperature}
                        onChange={(v) => handleSaveConfig({ temperature: v })}
                        style={{ width: 200 }}
                      />
                      <span style={{ fontSize: 13, color: '#8c8c8c' }}>{config.temperature}</span>
                    </div>
                  </div>

                  <div>
                    <h4 style={{ marginBottom: 8 }}>Forgetting Policy</h4>
                    <p style={{ fontSize: 13, color: '#8c8c8c', marginBottom: 8 }}>
                      How old memories are removed when capacity is reached.
                    </p>
                    <Select
                      value={config.forgetting_policy}
                      onChange={(v) => handleSaveConfig({ forgetting_policy: v })}
                      style={{ width: 120 }}
                      options={[
                        { value: 'FIFO', label: 'FIFO (First In, First Out)' },
                      ]}
                    />
                  </div>
                </Card>
              </Spin>
            ),
          },
        ]}
      />
    </div>
  );
};

export default Memory;
