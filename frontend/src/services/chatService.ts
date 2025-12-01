import type { ChatChunk, ChatRequest } from '../types/chat';

const API_BASE_URL = 'http://localhost:8000/api';

export class ChatService {
  async *streamChat(request: ChatRequest): AsyncGenerator<ChatChunk, void, unknown> {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: request.query,
          auto_confirm_ultra: request.auto_confirm_ultra ?? false,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const chunk = JSON.parse(line.slice(6)) as ChatChunk;
              yield chunk;
            } catch (e) {
              console.error('Failed to parse chunk:', e);
            }
          }
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      yield {
        type: 'error',
        message: 'Connection failed',
        details: errorMessage,
      };
    }
  }

  async checkHealth(): Promise<{ status: string; stats: any }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    return response.json();
  }
}

export const chatService = new ChatService();