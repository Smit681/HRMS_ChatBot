"""
Ultra-Complex Pipeline - Deep Analysis with Batch Processing
=============================================================

Handles prediction/analysis queries using MongoDB + Batch LLM:

Flow:
1. Get ALL relevant employees from MongoDB (structured data)
2. Split into batches of 10
3. Process batches in parallel (each batch analyzed by LLM)
4. Synthesize final answer with STREAMING
5. Return comprehensive analysis

Speed: 1-3 minutes (with user confirmation)

Usage:
    from src.pipelines.ultra_complex_pipeline import UltraComplexPipeline
    
    pipeline = UltraComplexPipeline()
    async for chunk in pipeline.process_stream("Predict raise candidates"):
        print(chunk)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from retrieval.context_builder import ContextBuilder
from retrieval.llm_interface import OllamaLLM
from pymongo import MongoClient
from typing import Dict, Any, AsyncIterator, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import asyncio
import time

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
    
    async def process_stream(self, query: str) -> AsyncIterator[dict]:
        """
        Process ultra-complex query with streaming synthesis
        
        Args:
            query: User's question
        
        Yields:
            dict: Progress updates, token chunks, and metadata
        """
        start_time = time.time()
        
        logger.info(f"[Ultra-Complex Pipeline] Processing with streaming: {query}")
        
        # Initial warning message
        yield {
            'type': 'status',
            'message': '⚠️ Ultra-complex query detected. This will take 2-3 minutes to analyze all employees...'
        }
        await asyncio.sleep(0)
        
        # Step 1: Fetch employees from MongoDB
        logger.info("[1/4] Fetching employees from MongoDB...")
        yield {'type': 'status', 'message': 'Fetching employees from MongoDB...'}
        
        loop = asyncio.get_event_loop()
        employees = await loop.run_in_executor(None, self._fetch_employees)
        
        if not employees:
            logger.warning("No employees found")
            yield {'type': 'error', 'message': 'No employees found in database'}
            return
        
        logger.info(f"✓ Fetched {len(employees)} employees")
        yield {'type': 'status', 'message': f'✓ Fetched {len(employees)} employees from database'}
        await asyncio.sleep(0)
        
        # Step 2: Create batches
        logger.info("[2/4] Creating batches...")
        yield {'type': 'status', 'message': 'Creating batches of 10 employees each...'}
        
        batches = self._create_batches(employees)
        logger.info(f"✓ Created {len(batches)} batches")
        yield {'type': 'status', 'message': f'✓ Created {len(batches)} batches for parallel processing'}
        await asyncio.sleep(0)
        
        # Step 3: Process all batches (this takes the longest time)
        logger.info(f"[3/4] Processing {len(batches)} batches in parallel...")
        yield {
            'type': 'status',
            'message': f'Processing {len(batches)} batches in parallel (this will take 2-3 minutes)...'
        }
        await asyncio.sleep(0)
        
        # Run batch processing in thread pool to avoid blocking event loop
        batch_results = await loop.run_in_executor(
            None,
            self._process_batches_parallel,
            query,
            batches
        )
        
        processing_time_so_far = time.time() - start_time
        logger.info(f"✓ All batches processed in {processing_time_so_far:.1f}s")
        yield {
            'type': 'status',
            'message': f'✓ All {len(batches)} batches processed successfully in {processing_time_so_far:.1f}s'
        }
        await asyncio.sleep(0)
        
        # Step 4: Synthesize final answer with STREAMING
        logger.info("[4/4] Synthesizing final answer with streaming...")
        yield {'type': 'status', 'message': 'Synthesizing final comprehensive answer...'}
        await asyncio.sleep(0)
        
        # Build synthesis context
        context = self.context_builder.build_synthesis_context(
            query=query,
            batch_results=batch_results,
            total_employees=len(employees)
        )
        
        prompt = self.context_builder.build_prompt(
            query=query,
            context=context,
            system_prompt=Config.SYSTEM_PROMPTS['synthesis']
        )
        
        # Stream the final answer token by token
        logger.info("Streaming final answer...")
        async for token in self.llm.generate_stream(prompt, temperature=0.2):
            yield {'type': 'token', 'content': token}
            await asyncio.sleep(0)
        
        # Send final metadata
        total_processing_time = time.time() - start_time
        logger.info(f"✅ Ultra-complex query complete in {total_processing_time:.1f}s")
        
        yield {
            'type': 'metadata',
            'total_analyzed': len(employees),
            'batch_results': len(batch_results),
            'processing_time': total_processing_time,
            'pipeline': 'ultra_complex'
        }
    
    def _fetch_employees(self) -> List[Dict[str, Any]]:
        """
        Fetch relevant employees from MongoDB
        
        For ultra-complex queries, we usually need ALL active employees,
        but can add filters based on query.
        
        Returns:
            List of employee documents
        """
        logger.info("Querying MongoDB for active employees...")
        
        # Build filter based on query
        mongo_filter = {'isActive': True}  # Default: only active employees
        
        # Fetch from MongoDB
        employees = list(self.collection.find(mongo_filter))
        
        # Convert ObjectId to string
        for emp in employees:
            if '_id' in emp:
                emp['_id'] = str(emp['_id'])
        
        logger.info(f"Retrieved {len(employees)} active employees")
        return employees
    
    def _create_batches(self, employees: List[Dict]) -> List[List[Dict]]:
        """
        Split employees into batches of 10
        
        Args:
            employees: List of employee documents
        
        Returns:
            List of batches, where each batch contains up to 10 employees
        """
        batch_size = Config.BATCH_SIZE
        batches = []
        
        for i in range(0, len(employees), batch_size):
            batch = employees[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Split {len(employees)} employees into {len(batches)} batches of {batch_size}")
        return batches
    
    def _process_batches_parallel(
        self,
        query: str,
        batches: List[List[Dict]]
    ) -> List[str]:
        """
        Process batches in parallel using ThreadPoolExecutor
        
        Args:
            query: User's question
            batches: List of employee batches
        
        Returns:
            List of batch analysis results (one per batch)
        """
        batch_results = []
        max_workers = Config.MAX_PARALLEL_WORKERS
        
        logger.info(f"Starting parallel processing with {max_workers} workers...")
        
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
                    logger.info(f"✓ Batch {batch_num + 1}/{len(batches)} complete")
                except Exception as e:
                    logger.error(f"✗ Batch {batch_num + 1} failed: {e}")
                    batch_results.append((batch_num, f"Error processing batch: {str(e)}"))
        
        # Sort by batch number to maintain order
        batch_results.sort(key=lambda x: x[0])
        sorted_results = [result for _, result in batch_results]
        
        logger.info(f"✓ All {len(batches)} batches processed successfully")
        return sorted_results
    
    def _process_single_batch(
        self,
        query: str,
        batch: List[Dict],
        batch_num: int,
        total_batches: int
    ) -> str:
        """
        Process a single batch of employees
        
        Args:
            query: User's question
            batch: List of employee documents (up to 10)
            batch_num: Current batch number (1-indexed)
            total_batches: Total number of batches
        
        Returns:
            LLM analysis result for this batch
        """
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} employees)...")
        
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
        
        # Generate analysis (synchronous - we're already in a thread)
        result = self.llm.generate(prompt, temperature=0.3)
        
        logger.info(f"✓ Batch {batch_num}/{total_batches} analysis complete")
        return result['text']
    
    def _format_batch_for_llm(
        self,
        query: str,
        batch: List[Dict],
        batch_num: int,
        total_batches: int
    ) -> str:
        """
        Format batch of employees for LLM analysis
        
        Args:
            query: User's question
            batch: List of employee documents
            batch_num: Current batch number
            total_batches: Total number of batches
        
        Returns:
            Formatted context string for LLM
        """
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
    
    def _extract_focus_area(self, query: str) -> str:
        """
        Extract what to focus on from query
        
        Args:
            query: User's question
        
        Returns:
            Focus area for analysis
        """
        query_lower = query.lower()
        
        if 'raise' in query_lower or 'promotion' in query_lower:
            return "salary, tenure, performance, last raise date"
        elif 'flight risk' in query_lower or 'retention' in query_lower:
            return "tenure, satisfaction indicators, market comparison"
        elif 'training' in query_lower or 'development' in query_lower:
            return "skills, experience, role progression"
        else:
            return "all relevant factors"
    
    def close(self):
        """Close MongoDB connection"""
        logger.info("Closing MongoDB connection...")
        self.client.close()
        logger.info("✓ MongoDB connection closed")


async def main():
    """Test ultra-complex pipeline with streaming"""
    print("=" * 70)
    print("ULTRA-COMPLEX PIPELINE - STREAMING TEST")
    print("=" * 70)
    
    pipeline = UltraComplexPipeline()
    
    # Test query
    query = "Predict which employees are candidates for raises based on salary and tenure"
    
    print(f"\n❓ Query: {query}")
    print(f"\n⚠️  This will take ~60-120 seconds with streaming output\n")
    
    print("=" * 70)
    print("STREAMING OUTPUT:")
    print("=" * 70)
    
    full_answer = ""
    
    try:
        async for chunk in pipeline.process_stream(query):
            chunk_type = chunk.get('type')
            
            if chunk_type == 'status':
                print(f"\n📊 {chunk['message']}")
            
            elif chunk_type == 'token':
                # Collect tokens for final answer
                token = chunk['content']
                full_answer += token
                print(token, end='', flush=True)
            
            elif chunk_type == 'metadata':
                print(f"\n\n{'=' * 70}")
                print("METADATA:")
                print(f"  • Total Analyzed: {chunk['total_analyzed']} employees")
                print(f"  • Batch Results: {chunk['batch_results']} batches")
                print(f"  • Processing Time: {chunk['processing_time']:.1f} seconds")
                print(f"  • Pipeline: {chunk['pipeline']}")
                print("=" * 70)
            
            elif chunk_type == 'error':
                print(f"\n❌ Error: {chunk['message']}")
    
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.close()
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())