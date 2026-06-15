// API client for DocMind backend
// Provides typed, error-handled API calls with correlation ID support

import type {
  ApiResponse,
  Document,
  DocumentStats,
  ProgressInfo,
  UploadedFile,
  ChatRequest,
  ChatResponse,
  SessionInfo,
  QueryMode,
  HealthInfo,
  AppInfo,
  Message,
  BatchJob,
  BatchScanResult,
  ProcessingMode,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

/**
 * Generic fetch wrapper with error handling and correlation ID
 */
export async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
  };

  // Add correlation ID for request tracing
  if (!headers['X-Correlation-ID']) {
    headers['X-Correlation-ID'] = generateCorrelationId();
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    // Handle non-JSON responses (SSE streams)
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      return response as unknown as T;
    }

    const data = await response.json();

    if (!response.ok) {
      const error = new Error(data.message || data.detail || 'Request failed');
      (error as any).code = data.code;
      (error as any).correlationId = data.correlation_id;
      (error as any).errors = data.errors;
      (error as any).status = response.status;
      throw error;
    }

    // Unwrap standardized response
    if (data.success !== undefined && 'data' in data) {
      return data.data as T;
    }

    return data as T;
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      const networkError = new Error('Network error. Please check if the backend is running.');
      (networkError as any).isNetworkError = true;
      throw networkError;
    }
    throw error;
  }
}

/**
 * Generate a short correlation ID for request tracing
 */
function generateCorrelationId(): string {
  return Math.random().toString(36).substring(2, 10);
}

// ===== Document API =====

export const documentsApi = {
  upload: (file: File): Promise<ApiResponse<UploadedFile>> => {
    const formData = new FormData();
    formData.append('file', file);
    return apiFetch('/documents/upload', {
      method: 'POST',
      body: formData,
    });
  },

  process: (docId: string, mode: ProcessingMode = 'standard'): Promise<ApiResponse<{ doc_id: string; status: string; mode: string }>> => {
    return apiFetch(`/documents/${docId}/process?mode=${mode}`, { method: 'POST' });
  },

  list: (): Promise<ApiResponse<{ documents: Document[]; total: number }>> => {
    return apiFetch('/documents/list');
  },

  stats: (): Promise<ApiResponse<DocumentStats>> => {
    return apiFetch('/documents/stats');
  },

  delete: (docId: string): Promise<ApiResponse<{ doc_id: string }>> => {
    return apiFetch(`/documents/${docId}`, { method: 'DELETE' });
  },

  getProgress: (docId: string): Promise<ApiResponse<ProgressInfo>> => {
    return apiFetch(`/documents/${docId}/progress`);
  },

  getAllProgress: (): Promise<ApiResponse<ProgressInfo[]>> => {
    return apiFetch('/documents/progress');
  },

  // ===== Batch Import =====

  batchScan: (directory: string): Promise<ApiResponse<BatchScanResult>> => {
    return apiFetch(`/documents/batch/scan?directory=${encodeURIComponent(directory)}`, { method: 'POST' });
  },

  batchStart: (directory: string, mode: ProcessingMode = 'fast'): Promise<ApiResponse<BatchJob>> => {
    return apiFetch(`/documents/batch/start?directory=${encodeURIComponent(directory)}&mode=${mode}`, { method: 'POST' });
  },

  batchGetStatus: (batchId: string): Promise<ApiResponse<BatchJob>> => {
    return apiFetch(`/documents/batch/${batchId}`);
  },

  batchList: (): Promise<ApiResponse<{ batches: BatchJob[] }>> => {
    return apiFetch('/documents/batches');
  },

  batchCancel: (batchId: string): Promise<ApiResponse<{ batch_id: string }>> => {
    return apiFetch(`/documents/batch/${batchId}/cancel`, { method: 'POST' });
  },
};

// ===== Chat API =====

export const chatApi = {
  send: (request: ChatRequest): Promise<ApiResponse<ChatResponse>> => {
    return apiFetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  },

  stream: (request: ChatRequest, signal: AbortSignal): Promise<Response> => {
    return fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    });
  },

  listSessions: (): Promise<ApiResponse<{ sessions: SessionInfo[]; total: number }>> => {
    return apiFetch('/chat/sessions');
  },

  getSessionMessages: (sessionId: string): Promise<ApiResponse<{ messages: Message[]; total: number }>> => {
    return apiFetch(`/chat/sessions/${sessionId}`);
  },

  deleteSession: (sessionId: string): Promise<ApiResponse> => {
    return apiFetch(`/chat/sessions/${sessionId}`, { method: 'DELETE' });
  },

  getQueryModes: (): Promise<ApiResponse<{ modes: QueryMode[] }>> => {
    return apiFetch('/chat/query_modes');
  },
};

// ===== Health & Info API =====

export const healthApi = {
  check: (): Promise<HealthInfo> => {
    return apiFetch('/health');
  },

  info: (): Promise<AppInfo> => {
    return apiFetch('/info');
  },
};

// ===== SSE Stream Parser =====

export interface StreamEvent {
  type: 'chunk' | 'sources' | 'thinking' | 'thinking_done' | 'status' | 'done' | 'error' | 'stopped';
  data?: any;
}

export async function* parseStream(
  response: Response,
): AsyncGenerator<StreamEvent, void, unknown> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          yield JSON.parse(line) as StreamEvent;
        } catch {
          console.warn('Failed to parse stream chunk:', line);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export default {
  documents: documentsApi,
  chat: chatApi,
  health: healthApi,
};
