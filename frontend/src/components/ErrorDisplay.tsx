interface ErrorDisplayProps {
  message: string;
  details?: string;
}

export function ErrorDisplay({ message, details }: ErrorDisplayProps) {
  return (
    <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 mb-4">
      <div className="flex items-start gap-3">
        <span className="text-red-500 text-xl">⚠</span>
        <div>
          <p className="text-red-400 font-medium">{message}</p>
          {details && (
            <p className="text-red-500/70 text-sm mt-1">{details}</p>
          )}
        </div>
      </div>
    </div>
  );
}