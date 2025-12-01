import type { ChatChunk } from '../types/chat';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  metadata?: ChatChunk;
}

export function ChatMessage({ role, content }: ChatMessageProps) {
  return (
    <div className={`flex ${role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2 ${
          role === 'user'
            ? 'bg-blue-600 text-white'
            : 'bg-gray-800 text-gray-100'
        }`}
      >
        <p className="whitespace-pre-wrap break-words">{content}</p>
    
      </div>
    </div>
  );
}