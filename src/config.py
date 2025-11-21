"""
HR RAG Chatbot - Centralized Configuration
===========================================

This configuration file contains all critical settings for the HR RAG system.
Modify these values to change behavior across the entire application.

IMPORTANT: After changing any settings here, you may need to:
1. Regenerate embeddings (if embedding model changes)
2. Restart Ollama (if LLM model changes)
3. Clear caches (if path changes)
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """
    Centralized configuration for HR RAG Chatbot
    
    Usage:
        from config import Config
        
        # Access settings
        model = Config.EMBEDDING_MODEL
        
        # Or use as dictionary
        settings = Config.to_dict()
    """
    
    # ============================================================================
    # PATHS - File System Locations
    # ============================================================================
    
    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # ChromaDB storage location (WHERE YOUR EMBEDDINGS ARE STORED)
    CHROMA_DB_PATH = str(DATA_DIR / "embeddings")
    
    # Logs directory
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_FILE = LOG_DIR / "hr_chatbot.log"
    
    # Cache directory (for future caching implementation)
    CACHE_DIR = DATA_DIR / "cache"
    
    # ============================================================================
    # EMBEDDING MODEL - Semantic Search Configuration
    # ============================================================================
    
    # Model name (MUST match what was used in production-etl.py)
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    
    # Model dimensions (automatically set based on model)
    EMBEDDING_DIMENSIONS = {
        "BAAI/bge-large-en-v1.5": 1024,
        "all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768
    }
    
    # Device for embedding model
    EMBEDDING_DEVICE = "cuda"  # Options: "cuda", "cpu", "mps" (Mac)
    
    # Batch size for embedding generation
    EMBEDDING_BATCH_SIZE = 32
    
    # ============================================================================
    # LLM MODEL - Text Generation Configuration
    # ============================================================================
    
    # Ollama model name
    LLM_MODEL = "qwen2.5:14b"
    
    # Alternative models (uncomment to use):
    # LLM_MODEL = "deepseek-r1:7b"
    # LLM_MODEL = "llama2:7b"
    # LLM_MODEL = "mistral:7b"
    
    # Ollama API endpoint
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # LLM generation parameters
    LLM_TEMPERATURE = 0.2  # Lower = more factual, Higher = more creative
    LLM_MAX_TOKENS = 1000  # Maximum response length
    LLM_TIMEOUT = 120      # Seconds before timeout
    
    # Temperature settings by query type
    TEMPERATURE_BY_TYPE = {
        "aggregation": 0.1,      # Very factual for counting/calculations
        "comparison": 0.2,       # Slightly creative for comparisons
        "simple_lookup": 0.2,    # Factual for direct lookups
        "conversational": 0.4    # More creative for chat
    }
    
    # ============================================================================
    # RETRIEVAL SETTINGS - How Documents Are Retrieved
    # ============================================================================
    
    # Number of documents to retrieve
    TOP_K = 5
    
    # Hybrid search weights (must sum to 1.0)
    SEMANTIC_WEIGHT = 0.7  # Weight for semantic similarity
    KEYWORD_WEIGHT = 0.3   # Weight for keyword matching
    
    # Retrieval strategy
    DEFAULT_RETRIEVAL_STRATEGY = "hybrid"  # Options: "semantic", "keyword", "hybrid"
    
    # Maximum context size (in tokens)
    MAX_CONTEXT_TOKENS = 2000
    
    # Character approximation per token
    CHARS_PER_TOKEN = 4
    
    # ============================================================================
    # CHROMADB COLLECTIONS - Collection Names
    # ============================================================================
    
    # Collection names (MUST match production-etl.py)
    COLLECTIONS = {
        "employees": "employees",
        "medical": "medical_plans",
        "dental": "dental_plans",
        "vision": "vision_plans",
        "employment": "employment_agreements",
        "faq": "faq"
    }
    
    # All collection names as list
    ALL_COLLECTIONS = list(COLLECTIONS.values())
    
    # Collection-specific settings
    COLLECTION_SETTINGS = {
        "employees": {
            "description": "Employee records and visa information",
            "searchable_fields": ["employee_id", "position", "visa_type"]
        },
        "medical_plans": {
            "description": "Medical insurance plans and coverage",
            "searchable_fields": ["plan_name", "copay", "deductible"]
        },
        "dental_plans": {
            "description": "Dental insurance coverage",
            "searchable_fields": ["plan_type", "coverage"]
        },
        "vision_plans": {
            "description": "Vision insurance benefits",
            "searchable_fields": ["exam_coverage", "frame_allowance"]
        },
        "employment_agreements": {
            "description": "Employment policies and agreements",
            "searchable_fields": ["policy_type", "benefit"]
        },
        "faq": {
            "description": "Frequently asked questions",
            "searchable_fields": ["question", "category"]
        }
    }
    
    # ============================================================================
    # QUERY CLASSIFICATION - How Queries Are Categorized
    # ============================================================================
    
    # Query complexity thresholds
    COMPLEXITY_THRESHOLD = {
        "simple": ["what is", "show me", "tell me about"],
        "complex": ["how many", "compare", "calculate", "average", "total"]
    }
    
    # Auto-routing enabled
    AUTO_ROUTE_QUERIES = True
    
    # Default to complex path if uncertain
    DEFAULT_TO_COMPLEX = False
    
    # ============================================================================
    # MULTI-AGENT SETTINGS - Agent Behavior
    # ============================================================================
    
    # Enable/disable specific agents
    ENABLE_CALCULATOR = True
    ENABLE_VALIDATOR = True
    ENABLE_QUERY_CLASSIFIER = True
    
    # Validation thresholds
    MIN_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to consider response valid
    MIN_SOURCE_SCORE = 0.3          # Minimum retrieval score to use source
    
    # Calculator operations enabled
    CALCULATOR_OPERATIONS = [
        "count", "sum", "average", "min", "max", 
        "date_diff", "percentage"
    ]
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    
    # Logging level
    LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Log format
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Enable file logging
    ENABLE_FILE_LOGGING = True
    
    # Enable console logging
    ENABLE_CONSOLE_LOGGING = True
    
    # Max log file size (MB)
    MAX_LOG_SIZE_MB = 10
    
    # Number of backup log files
    LOG_BACKUP_COUNT = 5
    
    # ============================================================================
    # PERFORMANCE SETTINGS
    # ============================================================================
    
    # Enable GPU acceleration
    USE_GPU = True
    
    # Batch processing size
    BATCH_SIZE = 50
    
    # Connection pool size for Ollama
    OLLAMA_CONNECTION_POOL = 10
    
    # Enable response caching
    ENABLE_CACHING = False  # TODO: Implement Redis caching
    
    # Cache TTL (seconds)
    CACHE_TTL = 3600
    
    # ============================================================================
    # DATA PROCESSING - ETL Settings
    # ============================================================================
    
    # Semantic chunking settings
    SEMANTIC_CHUNKING = {
        "enabled": True,
        "buffer_size": 1,
        "breakpoint_threshold": 95,
        "max_chunk_size": 512  # tokens
    }
    
    # Text cleaning settings
    TEXT_CLEANING = {
        "remove_extra_whitespace": True,
        "normalize_dates": True,
        "handle_missing_values": True,
        "missing_value_replacements": {
            "NaT": "[Not Available]",
            "nan": "[Not Specified]",
            "None": "[Not Recorded]"
        }
    }
    
    # ============================================================================
    # COMPANY-SPECIFIC SETTINGS
    # ============================================================================
    
    # Company information
    COMPANY_NAME = "Itlize Global, LLC"
    COMPANY_ADDRESS = "242 Old New Brunswick Road, Suite#250, Piscataway, NJ 08854"
    COMPANY_PHONE = "732.529.6129"
    
    # HR contact
    HR_EMAIL = "insurance@itlize.com"
    
    # Benefits information
    BENEFITS = {
        "sick_days": 5,
        "vacation_days": 8,
        "federal_holidays": 6,
        "off_project_stipend": 1300  # per month
    }
    
    # Visa types tracked
    VISA_TYPES = [
        "H-1B", "H-1B Extension", "H-1B Amendment", "H-1B Audit",
        "Green Card", "OPT", "OPT-Extension", "CPT", 
        "Citizen", "TN Visa"
    ]
    
    # Employment types
    EMPLOYMENT_TYPES = [
        "FullTime", "PartTime", "PermanentContractor"
    ]
    
    # Pay types
    PAY_TYPES = [
        "Compensation Package", "W2 contract", 
        "C2C contract", "1099 contract"
    ]
    
    # ============================================================================
    # SYSTEM PROMPTS - Prompt Templates
    # ============================================================================
    
    SYSTEM_PROMPTS = {
        "default": """You are an HR assistant for Itlize Global, LLC.
Answer questions accurately based on the provided context.
Be concise, professional, and helpful.
If the answer is not in the context, say so clearly.
Always cite which source(s) you used.""",
        
        "aggregation": """You are an HR analytics assistant.
Answer the question using the provided context and calculation results.
Be precise with numbers - use the exact calculation result.
Format: Give the answer directly, then briefly explain the context.
Keep it concise and professional.""",
        
        "comparison": """You are an HR comparison assistant.
Compare the items objectively using the provided context.
Structure: Present key differences clearly in a table or list format.
Be balanced and factual.""",
        
        "conversational": """You are a friendly HR assistant for Itlize Global.
Provide helpful, conversational responses based on the context.
Be warm and professional while maintaining accuracy."""
    }
    
    # ============================================================================
    # FEATURE FLAGS - Enable/Disable Features
    # ============================================================================
    
    FEATURES = {
        "multi_agent": True,          # Use multi-agent orchestrator
        "simple_path": True,          # Enable simple RAG path
        "streaming_responses": False, # Stream LLM responses (TODO)
        "conversation_history": False, # Track conversation (TODO)
        "user_feedback": False,       # Collect feedback (TODO)
        "analytics": False            # Track usage analytics (TODO)
    }
    
    # ============================================================================
    # VALIDATION RULES
    # ============================================================================
    
    VALIDATION_RULES = {
        "check_citations": True,
        "check_completeness": True,
        "check_numbers": True,
        "check_hallucinations": True,
        "min_response_length": 20
    }
    
    # ============================================================================
    # ERROR MESSAGES - User-Facing Messages
    # ============================================================================
    
    ERROR_MESSAGES = {
        "no_results": "I couldn't find relevant information to answer your question. Please try rephrasing or contact HR directly at {hr_email}.",
        "timeout": "I apologize, the request took too long. Please try again or simplify your question.",
        "connection_error": "I'm having trouble connecting to my knowledge base. Please try again in a moment.",
        "invalid_query": "I didn't understand that question. Could you please rephrase it?"
    }
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dictionary of all configuration settings
        """
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def get_embedding_dim(cls) -> int:
        """
        Get embedding dimensions for current model
        
        Returns:
            Embedding dimension size
        """
        return cls.EMBEDDING_DIMENSIONS.get(cls.EMBEDDING_MODEL, 1024)
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration settings
        
        Returns:
            True if valid, raises Exception if invalid
        """
        # Check paths exist
        if not cls.RAW_DATA_DIR.exists():
            raise ValueError(f"Raw data directory not found: {cls.RAW_DATA_DIR}")
        
        # Check weights sum to 1
        if abs((cls.SEMANTIC_WEIGHT + cls.KEYWORD_WEIGHT) - 1.0) > 0.01:
            raise ValueError("Semantic and keyword weights must sum to 1.0")
        
        # Check embedding model is supported
        if cls.EMBEDDING_MODEL not in cls.EMBEDDING_DIMENSIONS:
            raise ValueError(f"Unsupported embedding model: {cls.EMBEDDING_MODEL}")
        
        return True
    
    @classmethod
    def create_directories(cls):
        """
        Create necessary directories if they don't exist
        """
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.LOG_DIR,
            cls.CACHE_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ All directories created/verified")
    
    @classmethod
    def print_config(cls):
        """
        Print current configuration in a readable format
        """
        print("=" * 70)
        print("HR RAG CHATBOT - CONFIGURATION")
        print("=" * 70)
        print(f"\n📁 PATHS:")
        print(f"  Project Root: {cls.PROJECT_ROOT}")
        print(f"  ChromaDB: {cls.CHROMA_DB_PATH}")
        print(f"  Logs: {cls.LOG_DIR}")
        
        print(f"\n🤖 EMBEDDING MODEL:")
        print(f"  Model: {cls.EMBEDDING_MODEL}")
        print(f"  Dimensions: {cls.get_embedding_dim()}")
        print(f"  Device: {cls.EMBEDDING_DEVICE}")
        
        print(f"\n💬 LLM MODEL:")
        print(f"  Model: {cls.LLM_MODEL}")
        print(f"  Temperature: {cls.LLM_TEMPERATURE}")
        print(f"  Max Tokens: {cls.LLM_MAX_TOKENS}")
        print(f"  Endpoint: {cls.OLLAMA_BASE_URL}")
        
        print(f"\n🔍 RETRIEVAL:")
        print(f"  Strategy: {cls.DEFAULT_RETRIEVAL_STRATEGY}")
        print(f"  Top-K: {cls.TOP_K}")
        print(f"  Semantic Weight: {cls.SEMANTIC_WEIGHT}")
        print(f"  Keyword Weight: {cls.KEYWORD_WEIGHT}")
        
        print(f"\n📚 COLLECTIONS:")
        for key, value in cls.COLLECTIONS.items():
            count = cls.COLLECTION_SETTINGS.get(key, {}).get("description", "")
            print(f"  {value}: {count}")
        
        print(f"\n⚙️  FEATURES:")
        for feature, enabled in cls.FEATURES.items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {feature}")
        
        print("\n" + "=" * 70)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_config() -> Config:
    """
    Get configuration instance
    
    Usage:
        from config import get_config
        config = get_config()
        print(config.EMBEDDING_MODEL)
    """
    return Config

def load_config_from_env():
    """
    Override configuration from environment variables
    
    Usage:
        export HR_RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
        export HR_RAG_LLM_MODEL="llama2:7b"
    """
    # Override with environment variables if present
    if os.getenv("HR_RAG_EMBEDDING_MODEL"):
        Config.EMBEDDING_MODEL = os.getenv("HR_RAG_EMBEDDING_MODEL")
    
    if os.getenv("HR_RAG_LLM_MODEL"):
        Config.LLM_MODEL = os.getenv("HR_RAG_LLM_MODEL")
    
    if os.getenv("HR_RAG_CHROMA_PATH"):
        Config.CHROMA_DB_PATH = os.getenv("HR_RAG_CHROMA_PATH")
    
    if os.getenv("HR_RAG_LOG_LEVEL"):
        Config.LOG_LEVEL = os.getenv("HR_RAG_LOG_LEVEL")
    
    print("✅ Configuration loaded from environment variables")


# ============================================================================
# MAIN - For Testing Configuration
# ============================================================================

if __name__ == "__main__":
    """
    Test configuration by running: python config.py
    """
    print("\n🧪 Testing Configuration...\n")
    
    # Create directories
    Config.create_directories()
    
    # Validate config
    try:
        Config.validate_config()
        print("✅ Configuration is valid\n")
    except Exception as e:
        print(f"❌ Configuration error: {e}\n")
        sys.exit(1)
    
    # Print configuration
    Config.print_config()
    
    # Test dictionary conversion
    print("\n📋 Configuration as Dictionary:")
    config_dict = Config.to_dict()
    print(f"  Total settings: {len(config_dict)}")
    
    print("\n✅ Configuration test complete!")