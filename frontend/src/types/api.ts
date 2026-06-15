// API types for DocMind

export interface ApiResponse<T = any> {
  success: boolean;
  code: string;
  message: string;
  data?: T;
  correlation_id?: string;
  errors?: Array<{ field: string; message: string; type: string }>;
}

export interface PaginatedResponse<T = any> {
  success: boolean;
  code: string;
  message: string;
  data: T[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

// Document types
export interface Document {
  key: string;
  name: string;
  type: string;
  size: string;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  progress: number;
  current_step: string;
  completed_steps: number;
  total_steps: number;
  sections: number;
  updatedAt: string;
}

export interface UploadedFile {
  doc_id: string;
  filename: string;
  file_size: number;
  file_type: string;
  status: string;
}

export interface DocumentStats {
  total_documents: number;
  completed: number;
  processing: number;
  failed: number;
  total_sections: number;
}

export interface ProgressInfo {
  doc_id: string;
  progress: number;
  current_step: string;
  status: string;
  message: string;
}

// Chat types
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  sources?: SourceItem[];
  thinking?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}

export interface SessionInfo {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  session_title?: string;
  query_mode?: 'local' | 'global' | 'hybrid' | 'naive' | 'mix' | 'bypass';
  chat_mode?: 'kb' | 'direct';
}

export interface ChatResponse {
  response: string;
  sources: SourceItem[];
}

export interface SourceItem {
  content: string;
  source: string;
  score: number;
}

export interface QueryMode {
  id: string;
  name: string;
  description: string;
}

// Health types
export interface HealthInfo {
  success: boolean;
  status: string;
  version: string;
  architecture: string;
  rag_status: string;
}

export interface AppInfo {
  success: boolean;
  name: string;
  version: string;
  description: string;
  architecture: string;
  features: string[];
}

// Batch import types
export interface BatchDocItem {
  doc_id: string;
  filename: string;
  file_path: string;
  file_size: number;
  status: 'pending' | 'uploading' | 'queued' | 'processing' | 'completed' | 'failed' | 'skipped';
  progress: number;
  current_step: string;
  error?: string;
  started_at?: string;
  completed_at?: string;
  retry_count: number;
}

export interface BatchJob {
  batch_id: string;
  directory: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'partial';
  total: number;
  completed: number;
  failed: number;
  skipped: number;
  progress_percent: number;
  documents: BatchDocItem[];
  created_at?: string;
  started_at?: string;
  finished_at?: string;
}

export interface BatchScanResult {
  directory: string;
  files: Array<{
    filename: string;
    file_path: string;
    file_size: number;
    extension: string;
  }>;
  total: number;
  supported_extensions: string[];
}

export type ProcessingMode = 'fast' | 'standard' | 'full';
