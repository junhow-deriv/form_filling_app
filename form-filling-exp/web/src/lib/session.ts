import { ChatMessage } from '@/types';

// Create a new chat message
export function createMessage(
  role: 'user' | 'assistant' | 'system',
  content: string,
  status?: ChatMessage['status']
): ChatMessage {
  return {
    id: Math.random().toString(36).substring(2, 15),
    role,
    content,
    timestamp: new Date(),
    status,
    toolCalls: [],
  };
}

// URL session persistence helpers
export function getSessionIdFromUrl(): string | null {
  if (typeof window === 'undefined') return null;
  const params = new URLSearchParams(window.location.search);
  return params.get('session');
}

export function setSessionIdInUrl(sessionId: string) {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  url.searchParams.set('session', sessionId);
  window.history.replaceState({}, '', url.toString());
}
