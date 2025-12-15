import { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { StatusIndicator } from './components/StatusIndicator';
import { ErrorDisplay } from './components/ErrorDisplay';
import { LoginForm } from './components/LoginForm';
import { RegisterForm } from './components/RegisterForm';
import { chatService } from './services/chatService';
import { authService } from './services/authService';
import type { ChatChunk } from './types/chat';
import { ChatHistory } from './components/ChatHistory';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  metadata?: ChatChunk;
}

type AuthView = 'login' | 'register' | 'chat';

function App() {
  const [authView, setAuthView] = useState<AuthView>('login');
  const [currentUser, setCurrentUser] = useState<{ email: string; full_name: string } | null>(null);
  const [authError, setAuthError] = useState<string>('');
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentAssistantMessage, setCurrentAssistantMessage] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<{ message: string; details?: string } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showHistory, setShowHistory] = useState(false);

  // Check authentication on mount
  useEffect(() => {
    const checkAuth = async () => {
      if (authService.isAuthenticated()) {
        try {
          const user = await authService.getCurrentUser();
          setCurrentUser(user);
          setAuthView('chat');
        } catch {
          authService.logout();
          setAuthView('login');
        }
      }
    };
    checkAuth();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentAssistantMessage]);

  // Auth handlers
  const handleLogin = async (email: string, password: string) => {
    try {
      setAuthError('');
      await authService.login({ email, password });
      const user = await authService.getCurrentUser();
      setCurrentUser(user);
      setAuthView('chat');
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Login failed');
      throw err;
    }
  };

  const handleRegister = async (email: string, password: string, full_name: string) => {
    try {
      setAuthError('');
      await authService.register({ email, password, full_name });
      // Auto-login after registration
      await authService.login({ email, password });
      const user = await authService.getCurrentUser();
      setCurrentUser(user);
      setAuthView('chat');
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Registration failed');
      throw err;
    }
  };

  const handleLogout = () => {
    authService.logout();
    setCurrentUser(null);
    setMessages([]);
    setAuthView('login');
  };

  // Chat handler
  const handleSendMessage = async (userMessage: string) => {
    setError(null);
    setIsStreaming(true);
    setStatusMessage('');
    setCurrentAssistantMessage('');

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

  // Render authentication views
  if (authView === 'login') {
    return (
      <LoginForm
        onLogin={handleLogin}
        onSwitchToRegister={() => {
          setAuthError('');
          setAuthView('register');
        }}
        error={authError}
      />
    );
  }

  if (authView === 'register') {
    return (
      <RegisterForm
        onRegister={handleRegister}
        onSwitchToLogin={() => {
          setAuthError('');
          setAuthView('login');
        }}
        error={authError}
      />
    );
  }

  // Render chat interface (authenticated)
  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      {/* Header */}
      <header className="sticky top-0 bg-gray-900 border-b border-gray-800 p-4 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">HR Chatbot</h1>
          <p className="text-sm text-gray-400">Ask about employees, benefits, and policies</p>
        </div>
        <div className="flex items-center gap-4">
          <button
              onClick={() => setShowHistory(!showHistory)}
              className="bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2"
            >
              <span>📜</span>
              <span>Chat History</span>
            </button>
          <span className="text-sm text-gray-400">
            Welcome, {currentUser?.full_name}
          </span>
          <button
            onClick={handleLogout}
            className="bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm transition-colors"
          >
            Logout
          </button>
        </div>
      </header>

      {showHistory && (
        <ChatHistory
          onClose={() => setShowHistory(false)}
          onLoadMessage={(query, response) => {
            setMessages([
              { role: 'user', content: query },
              { role: 'assistant', content: response }
            ]);
            setShowHistory(false);
          }}
        />
      )}

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