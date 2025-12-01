import { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { StatusIndicator } from './components/StatusIndicator';
import { ErrorDisplay } from './components/ErrorDisplay';
import { chatService } from './services/chatService';
import type { ChatChunk } from './types/chat';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  metadata?: ChatChunk;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentAssistantMessage, setCurrentAssistantMessage] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<{ message: string; details?: string } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentAssistantMessage]);

  const handleSendMessage = async (userMessage: string) => {
    setError(null);
    setIsStreaming(true);
    setStatusMessage('');
    setCurrentAssistantMessage('');

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    try {
      let assistantMessage = '';
      let finalMetadata: ChatChunk | undefined;

      for await (const chunk of chatService.streamChat({ query: userMessage })) {
        switch (chunk.type) {
          case 'status':
            setStatusMessage(chunk.message);
            break;

          case 'classification':
            setStatusMessage(`Query type: ${chunk.query_type} (${(chunk.confidence * 100).toFixed(0)}% confidence)`);
            break;

          case 'token':
            assistantMessage += chunk.content;
            setCurrentAssistantMessage(assistantMessage);
            break;

          case 'metadata':
            finalMetadata = chunk;
            break;

          case 'complete':
            setStatusMessage('');
            setMessages(prev => [
              ...prev,
              { role: 'assistant', content: assistantMessage, metadata: finalMetadata }
            ]);
            setCurrentAssistantMessage('');
            break;

          case 'error':
            setError({ message: chunk.message, details: chunk.details });
            setStatusMessage('');
            break;

          case 'confirmation_required':
            setStatusMessage(chunk.message);
            // TODO: Handle user confirmation for ultra-complex queries
            break;
        }
      }
    } catch (err) {
      setError({
        message: 'Failed to connect to chatbot',
        details: err instanceof Error ? err.message : 'Unknown error'
      });
    } finally {
      setIsStreaming(false);
      setStatusMessage('');
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 p-4">
        <h1 className="text-2xl font-bold">HR Chatbot</h1>
        <p className="text-sm text-gray-400">Ask about employees, benefits, and policies</p>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 && (
            <div className="text-center text-gray-500 mt-20">
              <p className="text-lg mb-2">👋 Welcome to HR Chatbot</p>
              <p className="text-sm">Ask questions about employees, insurance plans, or company policies</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <ChatMessage key={idx} {...msg} />
          ))}

          {currentAssistantMessage && (
            <ChatMessage role="assistant" content={currentAssistantMessage} />
          )}

          {statusMessage && <StatusIndicator message={statusMessage} />}

          {error && <ErrorDisplay {...error} />}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input */}
      <ChatInput onSubmit={handleSendMessage} disabled={isStreaming} />
    </div>
  );
}

export default App;