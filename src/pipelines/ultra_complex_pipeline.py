"""
Ultra-Complex Pipeline - Deep Analysis with Batch Processing
=============================================================

Handles prediction/analysis queries using MongoDB + Batch LLM:

Flow:
1. Get ALL relevant employees from MongoDB (structured data)
2. Split into batches of 10
3. Process batches in parallel (each batch analyzed by LLM)
4. Synthesize final answer
5. Return comprehensive analysis

Speed: 1-3 minutes (with user confirmation)

Usage:
    from src.pipelines.ultra_complex_pipeline import UltraComplexPipeline
    
    pipeline = UltraComplexPipeline()
    result = pipeline.process("Predict raise candidates")
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from retrieval.context_builder import ContextBuilder
from retrieval.llm_interface import OllamaLLM
from pymongo import MongoClient
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltraComplexPipeline:
    
    def __init__(self):
        """Initialize ultra-complex pipeline"""
        logger.info("Initializing Ultra-Complex Pipeline...")
        
        # Connect to MongoDB
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.MONGODB_DB_NAME]
        self.collection = self.db['employees_structured']
        
        self.context_builder = ContextBuilder()
        self.llm = OllamaLLM()
        
        logger.info("✅ Ultra-Complex Pipeline ready!")
    
    def process(self, query: str) -> Dict[str, Any]:
        """        
        Args:
            query: User's question
        
        Returns:
            {
                'answer': str,
                'batch_results': List[str],
                'total_analyzed': int,
                'processing_time': float,
                'pipeline': str
            }
        """
        import time
        start_time = time.time()
        
        logger.info(f"[Ultra-Complex Pipeline] Processing: {query}")
        
        # Step 1: Fetch ALL relevant employees from MongoDB
        logger.info("[1/4] Fetching employees from MongoDB...")
        employees = self._fetch_employees()
        
        if not employees:
            logger.warning("No employees found")
            return self._no_results_response(query)
        
        logger.info(f"Fetched {len(employees)} employees")
        
        # Step 2: Split into batches
        logger.info("[2/4] Creating batches...")
        batches = self._create_batches(employees)
        logger.info(f"Created {len(batches)} batches")
        
        # Step 3: Process batches in parallel
        print(f"\n🔄 Processing {len(batches)} batches in parallel...")
        batch_results = self._process_batches_parallel(query, batches)
        
        # Step 4: Synthesize final answer
        print(f"\n🔄 Synthesizing final answer...")
        final_answer = self._synthesize_results(query, batch_results, len(employees))
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Processing complete in {processing_time:.1f}s")
        
        return {
            'answer': final_answer,
            'batch_results': batch_results,
            'total_analyzed': len(employees),
            'processing_time': processing_time,
            'pipeline': 'ultra_complex'
        }
    
    def _fetch_employees(self) -> List[Dict[str, Any]]:
        """
        Fetch relevant employees from MongoDB
        
        For ultra-complex queries, we usually need ALL active employees,
        but can add filters based on query.
        
        Args:
            query: Original query
        
        Returns:
            List of employee documents
        """        
        # Build filter based on query
        mongo_filter = {'isActive': True}  # Default: only active employees
        
        # Fetch from MongoDB
        employees = list(self.collection.find(mongo_filter))
        
        # Convert ObjectId to string
        for emp in employees:
            if '_id' in emp:
                emp['_id'] = str(emp['_id'])
        
        return employees
    
    def _create_batches(self, employees: List[Dict]) -> List[List[Dict]]:
        """Split employees into batches"""
        batch_size = Config.BATCH_SIZE
        batches = []
        
        for i in range(0, len(employees), batch_size):
            batch = employees[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _process_batches_parallel(
        self,
        query: str,
        batches: List[List[Dict]]
    ) -> List[str]:
        """Process batches in parallel"""
        batch_results = []
        max_workers = Config.MAX_PARALLEL_WORKERS
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(
                    self._process_single_batch,
                    query,
                    batch,
                    batch_num + 1,
                    len(batches)
                ): batch_num
                for batch_num, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    result = future.result()
                    batch_results.append((batch_num, result))
                    print(f"✅ Batch {batch_num + 1}/{len(batches)} complete")
                except Exception as e:
                    logger.error(f"Batch {batch_num + 1} failed: {e}")
                    batch_results.append((batch_num, f"Error: {str(e)}"))
        
        # Sort by batch number
        batch_results.sort(key=lambda x: x[0])
        return [result for _, result in batch_results]
    
    def _process_single_batch(
        self,
        query: str,
        batch: List[Dict],
        batch_num: int,
        total_batches: int
    ) -> str:
        """Process a single batch"""
        logger.info(f"Processing batch {batch_num}/{total_batches}...")
        
        # Format batch data for LLM
        batch_context = self._format_batch_for_llm(query, batch, batch_num, total_batches)
        
        # Build prompt
        prompt = self.context_builder.build_prompt(
            query=query,
            context=batch_context,
            system_prompt=Config.SYSTEM_PROMPTS['batch_analysis'].format(
                focus_area=self._extract_focus_area(query)
            )
        )
        
        # Generate analysis
        result = self.llm.generate(prompt, temperature=0.3)
        
        return result['text']
    
    def _format_batch_for_llm(
        self,
        query: str,
        batch: List[Dict],
        batch_num: int,
        total_batches: int
    ) -> str:
        """Format batch of employees for LLM analysis"""
        context_parts = []
        
        context_parts.append(f"Batch {batch_num}/{total_batches} - Analyzing {len(batch)} employees")
        context_parts.append("")
        context_parts.append(f"Query: {query}")
        context_parts.append("")
        context_parts.append("Employee Data:")
        context_parts.append("-" * 70)
        
        for i, emp in enumerate(batch, 1):
            context_parts.append(f"\nEmployee {i}:")
            context_parts.append(f"  ID: {emp.get('employeeId')}")
            context_parts.append(f"  Position: {emp.get('position')}")
            context_parts.append(f"  Salary: ${emp.get('salary'):,.2f}" if emp.get('salary') else "  Salary: N/A")
            context_parts.append(f"  Joining Date: {emp.get('joiningDate')}")
            context_parts.append(f"  Employment Type: {emp.get('employmentType')}")
            context_parts.append(f"  Assignment: {emp.get('assignment')}")
            context_parts.append(f"  Health Insurance: {emp.get('healthInsurance')}")
            context_parts.append(f"  401k: {emp.get('has401k')}")
            
            if emp.get('visas'):
                visa_info = [f"{v['visaType']} ({v['status']})" for v in emp['visas']]
                context_parts.append(f"  Visas: {', '.join(visa_info)}")
        
        context_parts.append("")
        context_parts.append("-" * 70)
        context_parts.append("Analyze each employee and provide insights relevant to the query.")
        
        return "\n".join(context_parts)
    
    def _synthesize_results(
        self,
        query: str,
        batch_results: List[str],
        total_employees: int
    ) -> str:
        """Synthesize batch results into final answer"""
        context = self.context_builder.build_synthesis_context(
            query=query,
            batch_results=batch_results,
            total_employees=total_employees
        )
        
        prompt = self.context_builder.build_prompt(
            query=query,
            context=context,
            system_prompt=Config.SYSTEM_PROMPTS['synthesis']
        )
        
        result = self.llm.generate(prompt, temperature=0.2)
        
        return result['text']
    
    def _extract_focus_area(self, query: str) -> str:
        """Extract what to focus on from query"""
        query_lower = query.lower()
        
        if 'raise' in query_lower or 'promotion' in query_lower:
            return "salary, tenure, performance, last raise date"
        elif 'flight risk' in query_lower or 'retention' in query_lower:
            return "tenure, satisfaction indicators, market comparison"
        elif 'training' in query_lower or 'development' in query_lower:
            return "skills, experience, role progression"
        else:
            return "all relevant factors"
    
    def _no_results_response(self, query: str) -> Dict[str, Any]:
        """Handle no results"""
        return {
            'answer': f"I couldn't find employees to analyze for: '{query}'",
            'batch_results': [],
            'total_analyzed': 0,
            'processing_time': 0,
            'pipeline': 'ultra_complex'
        }
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


def main():
    """Test ultra-complex pipeline"""
    print("=" * 70)
    print("ULTRA-COMPLEX PIPELINE - TESTING")
    print("=" * 70)
    
    pipeline = UltraComplexPipeline()
    
    # Test query
    query = "Predict which employees are candidates for raises based on salary and tenure"
    
    print(f"\n❓ Query: {query}")
    print(f"\n⚠️  This will take ~60-90 seconds (parallel processing)")
    
    confirm = input("\nContinue? (yes/no): ").strip().lower()
    
    if confirm in ['yes', 'y']:
        result = pipeline.process(query)
        
        print(f"\n{'=' * 70}")
        print("RESULTS")
        print("=" * 70)
        print(f"\n✅ Analyzed: {result['total_analyzed']} employees")
        print(f"⏱️  Time: {result['processing_time']:.1f} seconds")
        print(f"\n📝 Final Answer:")
        print(result['answer'])
    else:
        print("\n❌ Cancelled")
    
    pipeline.close()
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()