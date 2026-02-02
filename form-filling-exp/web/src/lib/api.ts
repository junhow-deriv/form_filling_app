import { AnalyzeResponse, StreamEvent, FormField, ChatMessage } from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ============================================================================
// API Key Validation
// ============================================================================

export async function validateAnthropicApiKey(apiKey: string): Promise<{ valid: boolean; message: string }> {
  const formData = new FormData();
  formData.append('api_key', apiKey);

  const response = await fetch(`${API_BASE}/validate-anthropic-key`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to validate API key' }));
    throw new Error(error.detail || 'Failed to validate API key');
  }

  return response.json();
}

export async function* analyzePdf(file: File, sessionId: string): AsyncGenerator<StreamEvent> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', sessionId);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to analyze PDF' }));
    throw new Error(error.detail || 'Failed to analyze PDF');
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonStr = line.slice(6).trim();
        if (jsonStr) {
          try {
            const event: StreamEvent = JSON.parse(jsonStr);
            yield event;
          } catch {
            console.error('Failed to parse SSE:', jsonStr);
          }
        }
      }
    }
  }
}

export interface StreamAgentFillOptions {
  file: File;
  instructions: string;
  filledPdfBytes?: Uint8Array | null;  // Use filled PDF for continuations
  isContinuation?: boolean;
  previousEdits?: Record<string, unknown> | null;
  resumeSessionId?: string | null;  // Agent session ID to resume conversation context
  userSessionId?: string | null;  // User's form-filling session ID (for concurrent user support)
  anthropicApiKey?: string | null;  // User's Anthropic API key for Claude calls
  contextFileIds?: Array<{ filename: string; document_id: string }>;  // Context file document IDs from /parse-files
}

export async function* streamAgentFill(
  options: StreamAgentFillOptions
): AsyncGenerator<StreamEvent> {
  const { file, instructions, filledPdfBytes, isContinuation, previousEdits, resumeSessionId, userSessionId, anthropicApiKey, contextFileIds } = options;

  const formData = new FormData();

  // For continuations, use the filled PDF bytes instead of original file
  if (isContinuation && filledPdfBytes) {
    const filledBlob = new Blob([new Uint8Array(filledPdfBytes)], { type: 'application/pdf' });
    formData.append('file', filledBlob, file.name);
  } else {
    formData.append('file', file);
  }

  formData.append('instructions', instructions);
  formData.append('is_continuation', String(isContinuation || false));

  if (previousEdits) {
    formData.append('previous_edits', JSON.stringify(previousEdits));
  }

  if (resumeSessionId) {
    formData.append('resume_session_id', resumeSessionId);
  }

  if (userSessionId) {
    formData.append('user_session_id', userSessionId);
  }

  if (anthropicApiKey) {
    formData.append('anthropic_api_key', anthropicApiKey);
  }

  if (contextFileIds && contextFileIds.length > 0) {
    const docIds = contextFileIds.map(f => f.document_id);
    formData.append('context_file_ids', JSON.stringify(docIds));
  }

  const response = await fetch(`${API_BASE}/fill-agent-stream`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to start agent' }));
    throw new Error(error.detail || 'Failed to start agent');
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonStr = line.slice(6).trim();
        if (jsonStr) {
          try {
            const event: StreamEvent = JSON.parse(jsonStr);
            yield event;
          } catch {
            console.error('Failed to parse SSE:', jsonStr);
          }
        }
      }
    }
  }
}

export function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
  }
  return bytes;
}

export function downloadPdf(bytes: Uint8Array, filename: string) {
  // Create a new Uint8Array copy to ensure we have a standard ArrayBuffer
  const copy = new Uint8Array(bytes);
  const blob = new Blob([copy], { type: 'application/pdf' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function createPdfUrl(bytes: Uint8Array): string {
  // Create a new Uint8Array copy to ensure we have a standard ArrayBuffer
  const copy = new Uint8Array(bytes);
  const blob = new Blob([copy], { type: 'application/pdf' });
  return URL.createObjectURL(blob);
}

// ============================================================================
// Session Restoration
// ============================================================================

export interface ContextFile {
  filename: string;
  document_id: string;
}

export interface SessionInfo {
  session_id: string;
  user_session_id: string;
  agent_session_id: string | null;
  has_filled_pdf: boolean;
  has_original_pdf: boolean;
  applied_edits: Record<string, unknown>;
  fields: FormField[];
  messages: ChatMessage[];
  pdf_filename: string | null;
  context_files: ContextFile[];
}

export async function getSessionInfo(sessionId: string): Promise<SessionInfo | null> {
  try {
    const response = await fetch(`${API_BASE}/session/${sessionId}`);
    if (!response.ok) {
      return null;
    }
    return response.json();
  } catch (error) {
    console.error('Failed to get session info:', error);
    return null;
  }
}

export async function getSessionPdf(sessionId: string): Promise<Uint8Array | null> {
  try {
    const response = await fetch(`${API_BASE}/session/${sessionId}/pdf`);
    if (!response.ok) {
      return null;
    }
    const arrayBuffer = await response.arrayBuffer();
    return new Uint8Array(arrayBuffer);
  } catch (error) {
    console.error('Failed to get session PDF:', error);
    return null;
  }
}

export async function getSessionOriginalPdf(sessionId: string): Promise<Uint8Array | null> {
  try {
    const response = await fetch(`${API_BASE}/session/${sessionId}/original-pdf`);
    if (!response.ok) {
      return null;
    }
    const arrayBuffer = await response.arrayBuffer();
    return new Uint8Array(arrayBuffer);
  } catch (error) {
    console.error('Failed to get session original PDF:', error);
    return null;
  }
}

// ============================================================================
// Context Files Parsing
// ============================================================================

export interface ParsedContextFile {
  filename: string;
  document_id: string;
}

export interface ParseFilesProgress {
  type: 'start' | 'progress' | 'complete' | 'error' | 'init' | 'system_message';
  total?: number;
  current?: number;
  filename?: string;
  status?: 'parsing' | 'reading_text' | 'docling' | 'complete' | 'error';
  error?: string;
  message?: string;
  document_id?: string;
  results?: Array<{
    filename: string;
    document_id?: string;
    success?: boolean;
    error?: string;
  }>;
  success_count?: number;
  error_count?: number;
  count?: number;
  storage_type?: string;
  content?: string;
  role?: 'system' | 'user' | 'assistant';
  timestamp?: string;
}

export async function* streamParseFiles(
  files: File[],
  userSessionId?: string | null
): AsyncGenerator<ParseFilesProgress> {
  const formData = new FormData();

  for (const file of files) {
    formData.append('files', file);
  }
  formData.append('is_ephemeral', 'true');

  if (userSessionId) {
    formData.append('user_session_id', userSessionId);
  }

  const response = await fetch(`${API_BASE}/parse-files`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to parse files' }));
    // Handle case where detail is an object/array (e.g. validation error)
    const errorMessage = typeof error.detail === 'string' 
      ? error.detail 
      : JSON.stringify(error.detail) || 'Failed to parse files';
    throw new Error(errorMessage);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonStr = line.slice(6).trim();
        if (jsonStr) {
          try {
            const event: ParseFilesProgress = JSON.parse(jsonStr);
            yield event;
          } catch {
            console.error('Failed to parse SSE:', jsonStr);
          }
        }
      }
    }
  }
}

export async function getParseStatus(): Promise<{
  docling_available: boolean;
  docling_error: string | null;
}> {
  const response = await fetch(`${API_BASE}/parse-status`);
  if (!response.ok) {
    throw new Error('Failed to get parse status');
  }
  return response.json();
}

// ============================================================================
// Field Detection
// ============================================================================

export async function detectFields(file: File): Promise<Uint8Array> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/detect-fields`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to detect fields' }));
    throw new Error(error.detail || 'Failed to detect fields');
  }

  // The endpoint returns PDF bytes directly
  const arrayBuffer = await response.arrayBuffer();
  return new Uint8Array(arrayBuffer);
}
