export type ChunkType = 
  | 'status' 
  | 'classification' 
  | 'token' 
  | 'metadata' 
  | 'complete' 
  | 'error' 
  | 'confirmation_required';

export interface BaseChunk {
  type: ChunkType;
}

export interface StatusChunk extends BaseChunk {
  type: 'status';
  message: string;
}

export interface ClassificationChunk extends BaseChunk {
  type: 'classification';
  query_type: string;
  confidence: number;
}

export interface TokenChunk extends BaseChunk {
  type: 'token';
  content: string;
}

export interface MetadataChunk extends BaseChunk {
  type: 'metadata';
  sources?: any[];
  num_sources?: number;
  confidence?: number;
  pipeline?: string;
  mongodb_operation?: any;
  result?: any;
  total_analyzed?: number;
}

export interface CompleteChunk extends BaseChunk {
  type: 'complete';
  query_type: string;
  processing_time: number;
}

export interface ErrorChunk extends BaseChunk {
  type: 'error';
  message: string;
  details?: string;
}

export interface ConfirmationChunk extends BaseChunk {
  type: 'confirmation_required';
  message: string;
  query_type: string;
}

export type ChatChunk = 
  | StatusChunk 
  | ClassificationChunk 
  | TokenChunk 
  | MetadataChunk 
  | CompleteChunk 
  | ErrorChunk 
  | ConfirmationChunk;

export interface ChatRequest {
  query: string;
  auto_confirm_ultra?: boolean;
}