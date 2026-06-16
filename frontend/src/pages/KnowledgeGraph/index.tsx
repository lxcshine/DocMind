import React, { useEffect, useRef, useState } from 'react';
import { Card, Select, Button, Space, Spin, Empty, message, Tag, Input, Switch } from 'antd';
import { ReloadOutlined, ZoomInOutlined, ZoomOutOutlined, FullscreenOutlined } from '@ant-design/icons';
import * as d3 from 'd3';
import { apiFetch } from '../../services/api';

const { Search } = Input;

interface KGNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: 'entity' | 'relation' | 'document' | 'chunk';
  properties?: Record<string, any>;
}

interface KGEdge extends d3.SimulationLinkDatum<KGNode> {
  source: string | KGNode;
  target: string | KGNode;
  relation: string;
  weight?: number;
}

interface KGData {
  nodes: KGNode[];
  edges: KGEdge[];
}

const KnowledgeGraph: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [loading, setLoading] = useState(false);
  const [graphData, setGraphData] = useState<KGData | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string>('');
  const [documents, setDocuments] = useState<Array<{ id: string; name: string }>>([]);
  const [highlightNodes, setHighlightNodes] = useState<Set<string>>(new Set());
  const [showLabels, setShowLabels] = useState(true);
  const [zoom, setZoom] = useState(1);

  // 加载文档列表
  useEffect(() => {
    loadDocuments();
  }, []);

  // 加载图谱数据
  useEffect(() => {
    if (selectedDocId) {
      loadGraph(selectedDocId);
    } else {
      loadFullGraph();
    }
  }, [selectedDocId]);

  // D3 渲染
  useEffect(() => {
    if (graphData && svgRef.current) {
      renderGraph();
    }
  }, [graphData, showLabels, highlightNodes]);

  const loadDocuments = async () => {
    try {
      const data = await apiFetch<any>('/documents/list');
      // apiFetch unwraps the response, so data is already the inner data object
      const docs = data?.documents ?? data ?? [];
      setDocuments(docs.map((doc: any) => ({ id: doc.key || doc.id || doc.doc_id, name: doc.name || doc.filename })));
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  const loadGraph = async (docId: string) => {
    setLoading(true);
    try {
      const data = await apiFetch<any>(`/knowledge-graph/document/${docId}`);
      // apiFetch unwraps the response, data is already {nodes, edges, stats}
      setGraphData(data);
    } catch (error) {
      console.error('Failed to load graph:', error);
      message.error('加载图谱失败');
    } finally {
      setLoading(false);
    }
  };

  const loadFullGraph = async () => {
    setLoading(true);
    try {
      const data = await apiFetch<any>('/knowledge-graph/full');
      // apiFetch unwraps the response, data is already {nodes, edges, stats}
      setGraphData(data);
    } catch (error) {
      console.error('Failed to load full graph:', error);
      message.error('加载图谱失败');
    } finally {
      setLoading(false);
    }
  };

  const renderGraph = () => {
    if (!svgRef.current || !graphData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    const g = svg.append('g');

    // 缩放行为
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        g.attr('transform', event.transform.toString());
        setZoom(event.transform.k);
      });

    svg.call(zoomBehavior);

    // 准备数据（d3 需要对象引用，不能展开）
    const nodes = graphData.nodes;
    const links = graphData.edges;

    // 力导向模拟
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // 绘制边
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', 1);

    // 边标签
    const linkText = g.append('g')
      .selectAll('text')
      .data(links)
      .join('text')
      .attr('font-size', '10px')
      .attr('fill', '#666')
      .text((d: KGEdge) => d.relation);

    // 绘制节点
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call((d3.drag<SVGGElement, KGNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended)) as any);

    // 节点圆形
    node.append('circle')
      .attr('r', (d: KGNode) => d.type === 'entity' ? 8 : 6)
      .attr('fill', (d: KGNode) => {
        if (highlightNodes.has(d.id)) return '#ff4d4f';
        switch (d.type) {
          case 'entity': return '#1890ff';
          case 'relation': return '#52c41a';
          case 'document': return '#faad14';
          case 'chunk': return '#722ed1';
          default: return '#999';
        }
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    // 节点标签
    if (showLabels) {
      node.append('text')
        .attr('dx', 12)
        .attr('dy', 4)
        .attr('font-size', '12px')
        .attr('fill', '#333')
        .text((d: KGNode) => d.label);
    }

    // 节点悬停提示
    node.append('title')
      .text((d: KGNode) => `${d.label}\n类型: ${d.type}`);

    // 更新位置
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      linkText
        .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
        .attr('y', (d: any) => (d.source.y + d.target.y) / 2);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event: d3.D3DragEvent<SVGGElement, KGNode, KGNode>, d: KGNode) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, KGNode, KGNode>, d: KGNode) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, KGNode, KGNode>, d: KGNode) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };

  const handleSearch = (value: string) => {
    if (!value || !graphData) {
      setHighlightNodes(new Set());
      return;
    }

    const matched = new Set<string>();
    graphData.nodes.forEach(node => {
      if (node.label.toLowerCase().includes(value.toLowerCase())) {
        matched.add(node.id);
      }
    });
    setHighlightNodes(matched);
  };

  const handleZoomIn = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      const currentTransform = d3.zoomTransform(svg.node()!);
      const newZoom = currentTransform.k * 1.2;
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().transform,
        d3.zoomIdentity.scale(newZoom)
      );
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      const currentTransform = d3.zoomTransform(svg.node()!);
      const newZoom = currentTransform.k / 1.2;
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().transform,
        d3.zoomIdentity.scale(newZoom)
      );
    }
  };

  const handleReset = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().transform,
        d3.zoomIdentity
      );
    }
  };

  return (
    <div style={{ padding: 24, height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Card
        title="知识图谱可视化"
        extra={
          <Space>
            <Select
              placeholder="选择文档"
              style={{ width: 200 }}
              value={selectedDocId || undefined}
              onChange={setSelectedDocId}
              allowClear
            >
              {documents.map(doc => (
                <Select.Option key={doc.id} value={doc.id}>
                  {doc.name}
                </Select.Option>
              ))}
            </Select>
            <Button icon={<ReloadOutlined />} onClick={() => selectedDocId ? loadGraph(selectedDocId) : loadFullGraph()}>
              刷新
            </Button>
          </Space>
        }
        style={{ marginBottom: 16 }}
      >
        <Space style={{ marginBottom: 16 }}>
          <Search
            placeholder="搜索节点"
            allowClear
            onSearch={handleSearch}
            style={{ width: 200 }}
          />
          <Switch
            checked={showLabels}
            onChange={setShowLabels}
            checkedChildren="显示标签"
            unCheckedChildren="隐藏标签"
          />
          <Button icon={<ZoomInOutlined />} onClick={handleZoomIn}>放大</Button>
          <Button icon={<ZoomOutOutlined />} onClick={handleZoomOut}>缩小</Button>
          <Button icon={<FullscreenOutlined />} onClick={handleReset}>重置视图</Button>
          <Tag color="blue">节点: {graphData?.nodes.length || 0}</Tag>
          <Tag color="green">边: {graphData?.edges.length || 0}</Tag>
          <Tag color="orange">缩放: {(zoom * 100).toFixed(0)}%</Tag>
        </Space>
      </Card>

      <Card style={{ flex: 1, overflow: 'hidden' }}>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Spin size="large" tip="加载图谱数据..." />
          </div>
        ) : graphData && graphData.nodes.length > 0 ? (
          <svg
            ref={svgRef}
            style={{ width: '100%', height: '100%', border: '1px solid #f0f0f0', borderRadius: 4 }}
          />
        ) : (
          <Empty description="暂无图谱数据" />
        )}
      </Card>
    </div>
  );
};

export default KnowledgeGraph;
