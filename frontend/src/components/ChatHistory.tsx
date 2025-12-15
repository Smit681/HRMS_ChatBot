import { useState, useEffect } from 'react';
import { chatService } from '../services/chatService';

interface HistoryMessage {
  id: string;
  query: string;
  response: string;
  query_type?: string;
  pipeline?: string;
  timestamp: string;
}

interface ChatHistoryProps {
  onClose: () => void;
  onLoadMessage: (query: string, response: string) => void;
}

export function ChatHistory({ onClose, onLoadMessage }: ChatHistoryProps) {
  const [history, setHistory] = useState<HistoryMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const data = await chatService.getHistory();
      setHistory(data.history);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-lg w-full max-w-4xl h-[80vh] flex flex-col border border-gray-800">
        {/* Header */}
        <div className="p-4 border-b border-gray-800 flex justify-between items-center">
          <h2 className="text-xl font-bold">Chat History</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading && (
            <div className="text-center text-gray-400 py-8">
              Loading history...
            </div>
          )}

          {error && (
            <div className="bg-red-900/20 border border-red-800 rounded-lg p-4">
              <p className="text-red-400">{error}</p>
            </div>
          )}

          {!loading && !error && history.length === 0 && (
            <div className="text-center text-gray-500 py-8">
              No chat history yet. Start a conversation!
            </div>
          )}

          {!loading && !error && history.length > 0 && (
            <div className="space-y-4">
              {history.map((msg) => (
                <div
                  key={msg.id}
                  className="bg-gray-800 rounded-lg p-4 hover:bg-gray-750 transition-colors cursor-pointer"
                  onClick={() => onLoadMessage(msg.query, msg.response)}
                >
                  {/* User Query */}
                  <div className="mb-3">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-blue-400 font-medium">You:</span>
                      <span className="text-xs text-gray-500">
                        {formatDate(msg.timestamp)}
                      </span>
                    </div>
                    <p className="text-gray-200">{msg.query}</p>
                  </div>

                  {/* Assistant Response */}
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-green-400 font-medium">Assistant:</span>
                      {msg.query_type && (
                        <span className="text-xs bg-gray-700 px-2 py-0.5 rounded">
                          {msg.query_type}
                        </span>
                      )}
                    </div>
                    <p className="text-gray-300 line-clamp-3">{msg.response}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800 text-sm text-gray-400 text-center">
          Showing last {history.length} messages
        </div>
      </div>
    </div>
  );
}