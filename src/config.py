from pathlib import Path

class Config:
    
    # PATHS    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    CHROMA_DB_PATH = str(DATA_DIR / "embeddings")

    # MONGODB
    MONGODB_URI = "mongodb://localhost:27017/"
    MONGODB_DB_NAME = "hr_chatbot"

    
    # EMBEDDING MODEL
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DEVICE = "cuda" 
    EMBEDDING_BATCH_SIZE = 50
    
    # LLM MODEL    
    LLM_MODEL = "qwen2.5:14b"
    OLLAMA_BASE_URL = "http://localhost:11434"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 1000
    
    # RETRIEVAL SETTINGS    
    TOP_K = 3
    SEMANTIC_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3
    MAX_CONTEXT_TOKENS = 4000
    
    # CHROMADB COLLECTIONS    
    COLLECTIONS = [
        "employees",
        "medical_plans",
        "dental_plans",
        "vision_plans",
        "employment_agreements",
        "faq"
    ]

    # BATCH PROCESSING
    BATCH_SIZE = 10              # Employees per batch for ultra-complex
    MAX_PARALLEL_WORKERS = 3     # Parallel LLM calls
    ULTRA_COMPLEX_TIMEOUT = 300  # 5 minutes max


    # Redis Configuration
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0  # Use database 0 for chatbot cache
    REDIS_CACHE_TTL_SECONDS = 3600  # 1 hour TTL for all cached entries

    # SYSTEM PROMPTS
    SYSTEM_PROMPTS = {
        'default': """You are an HR assistant for Itlize Global, LLC. Answer questions accurately based on the provided context. Be concise, professional, and helpful. If the user's question references previous conversation (e.g., "what about that employee", "the same plan"), use the recent conversation history to understand the full context. If query is asking about certain employee information, company policy, health/dental insurance without giving identifing information, check the recent conversations to idenfy which employee, policy, insurance the query is referring to. Provide clear, concise answers. If the answer is not in the context, say so clearly.""",
        
        'aggregation': """You are an HR analytics assistant.
    Answer using the provided context and calculation results.
    Be precise with numbers - use the exact calculation result.
    Give the answer directly, then briefly explain.
    Keep it concise and professional.""",
        
        'comparison': """You are an HR comparison assistant.
    Compare the items objectively using the provided context.
    Present key differences clearly.
    Be balanced and factual.""",
        
        'batch_analysis': """You are analyzing a batch of employee data.
    Extract key insights for each employee based on the query.
    Return results in structured JSON format.
    Be objective and data-driven.""",
        
        'synthesis': """You are synthesizing results from multiple analyses.
    Combine the batch results into a coherent final answer.
    Prioritize the most important findings.
    Present in clear, actionable format.""", 

        'MONGODB_QUERY_PROMPT' : """You are a MongoDB query expert. Convert natural language questions to MongoDB queries.

    Database Schema:
    - Collection: employees_structured
    - Fields:
    * employeeId (int)
    * joiningDate (string: "YYYY-MM-DD")
    * employmentType (string: "FullTime", "PartTime", "PermanentContractor")
    * salary (float)
    * position (string: "Software Developer", "Technical Project Manager", etc.)
    * assignment (string: "Client", "Internal", "Trainee")
    * healthInsurance (boolean)
    * has401k (boolean)
    * terminationDate (string or null)
    * isActive (boolean)
    * visas (array of objects):
        - visaType (string: "H-1B", "OPT", "Green Card", "Citizen", etc.)
        - status (string: "Complete", "Pending", "Unknown")
        - entryToUS (string)
        - startDate (string)
        - endDate (string)

    Convert the question to a MongoDB operation. Return ONLY valid JSON in this format:
    {{
    "operation": "count" | "aggregate" | "find",
    "filter": {{}},
    "pipeline": [],
    "field": "fieldName" (optional, for aggregations)
    }}

    Examples:

    Question: "How many employees have H-1B visas?"
    {{
    "operation": "count",
    "filter": {{"visas.visaType": "H-1B"}}
    }}

    Question: "Calculate average salary"
    {{
    "operation": "aggregate",
    "pipeline": [
        {{"$group": {{"_id": null, "avgSalary": {{"$avg": "$salary"}}}}}}
    ]
    }}

    Question: "How many Software Developers are there?"
    {{
    "operation": "count",
    "filter": {{"position": "Software Developer"}}
    }}

    Question: "Count employees with salary over $100,000"
    {{
    "operation": "count",
    "filter": {{"salary": {{"$gt": 100000}}}}
    }}

    Question: "Average salary by position"
    {{
    "operation": "aggregate",
    "pipeline": [
        {{"$group": {{"_id": "$position", "avgSalary": {{"$avg": "$salary"}}}}}}
    ]
    }}

    Question: "How many employees have both health insurance and 401k?"
    {{
    "operation": "count",
    "filter": {{"healthInsurance": true, "has401k": true}}
    }}

    Question: "List employees on client assignment with H-1B"
    {{
    "operation": "find",
    "filter": {{"assignment": "Client", "visas.visaType": "H-1B"}},
    "limit": 100
    }}

    Now convert this question:
    Question: "{query}"

    Return ONLY the JSON, nothing else."""
    }


if __name__ == "__main__":
    """Test configuration"""
    print("=" * 70)
    print("HR RAG CHATBOT - CONFIGURATION")
    print("=" * 70)
    print(f"\nPaths:")
    print(f"  ChromaDB: {Config.CHROMA_DB_PATH}")
    print(f"\nModels:")
    print(f"  Embedding: {Config.EMBEDDING_MODEL}")
    print(f"  LLM: {Config.LLM_MODEL}")
    print(f"\nCollections: {len(Config.COLLECTIONS)}")
    for col in Config.COLLECTIONS:
        print(f"  - {col}")
    print("\nConfiguration loaded successfully!")