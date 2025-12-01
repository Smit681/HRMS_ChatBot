interface StatusIndicatorProps {
  message: string;
}

export function StatusIndicator({ message }: StatusIndicatorProps) {
  return (
    <div className="flex items-center gap-2 text-sm text-gray-400 mb-4">
      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
      <span>{message}</span>
    </div>
  );
}