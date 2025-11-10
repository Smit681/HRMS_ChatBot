"""
QUERY CLASSIFIER AGENT - Understanding User Intent

Classifies queries to route them to appropriate handlers.

QUERY TYPES:
1. SIMPLE_LOOKUP - "What is X?" - Direct retrieval
2. AGGREGATION - "How many X?" - Count/sum/average
3. COMPARISON - "Compare X and Y" - Multi-document analysis
4. FILTER - "Show X with Y" - Filtered search
5. CONVERSATIONAL - "Tell me more" - Follow-up
6. ANALYTICAL - "Which is best?" - Reasoning required

WHY CLASSIFICATION?
- Different queries need different approaches
- Aggregations need calculation agent
- Comparisons need multiple retrievals
- Simple lookups can skip complex processing
"""

import re
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classifies user queries to determine processing strategy
    
    METHODS:
    1. Rule-based: Pattern matching (fast, simple)
    2. Entity extraction: Find IDs, dates, numbers
    3. Intent detection: What does user want?
    """
    
    def __init__(self):
        """Initialize query classifier with patterns"""
        logger.info("Initializing Query Classifier...")
        
        # Classification patterns
        self.patterns = {
            'aggregation': [
                r'\bhow many\b', r'\bcount\b', r'\btotal\b',
                r'\baverage\b', r'\bmean\b', r'\bsum\b',
                r'\bnumber of\b', r'\bpercentage\b'
            ],
            'comparison': [
                r'\bcompare\b', r'\bdifference\b', r'\bversus\b',
                r'\bvs\b', r'\bbetter\b', r'\bbest\b',
                r'\bworst\b', r'\bcheaper\b'
            ],
            'filter': [
                r'\bwith\b', r'\bwhere\b', r'\bwho have\b',
                r'\bonly\b', r'\bexclude\b', r'\bexcept\b',
                r'\bfilter\b', r'\bsalary >\b', r'\bsalary <\b'
            ],
            'prediction': [
                r'\bwill\b', r'\bforecast\b', r'\bpredict\b',
                r'\btrend\b', r'\bexpected\b', r'\bfuture\b'
            ],
            'calculation': [
                r'\bcalculate\b', r'\bcompute\b', r'\badd\b',
                r'\bsubtract\b', r'\bmultiply\b', r'\bdivide\b'
            ]
        }
        
        logger.info("✅ Query Classifier ready!")
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify a query and extract entities
        
        Args:
            query: User's question
        
        Returns:
            {
                'query_type': str,        # Primary query type
                'sub_types': List[str],   # Secondary types
                'entities': dict,         # Extracted entities
                'complexity': str,        # 'simple' or 'complex'
                'suggested_strategy': str # Retrieval strategy
            }
        """
        logger.info(f"Classifying query: {query}")
        
        query_lower = query.lower()
        
        # Detect query types
        detected_types = []
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_types.append(query_type)
                    break
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine primary type
        if not detected_types:
            # Simple lookup by default
            primary_type = 'simple_lookup'
        else:
            primary_type = detected_types[0]
        
        # Determine complexity
        complexity = self._determine_complexity(
            primary_type, 
            detected_types, 
            entities
        )
        
        # Suggest retrieval strategy
        strategy = self._suggest_strategy(
            primary_type,
            detected_types,
            entities
        )
        
        result = {
            'query_type': primary_type,
            'sub_types': detected_types[1:] if len(detected_types) > 1 else [],
            'entities': entities,
            'complexity': complexity,
            'suggested_strategy': strategy
        }
        
        logger.info(f"Classification: {primary_type} ({complexity})")
        return result
    
    def _extract_entities(self, query: str) -> Dict[str, List[Any]]:
        """
        Extract entities from query
        
        ENTITIES:
        - employee_ids: [1503, 1504, ...]
        - dates: ['2025-01-01', ...]
        - numbers: [1000, 2500, ...]
        - visa_types: ['H-1B', 'OPT', ...]
        - plan_names: ['PPO 1000', 'PPO 2500', ...]
        """
        entities = {
            'employee_ids': [],
            'dates': [],
            'numbers': [],
            'visa_types': [],
            'plan_names': []
        }
        
        # Extract employee IDs (4-digit numbers)
        employee_ids = re.findall(r'\b\d{4}\b', query)
        entities['employee_ids'] = [int(eid) for eid in employee_ids]
        
        # Extract dates (various formats)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2025-01-01
            r'\d{2}/\d{2}/\d{4}',  # 01/01/2025
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, query, re.IGNORECASE)
            entities['dates'].extend(dates)
        
        # Extract numbers (amounts, quantities)
        numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?', query)
        entities['numbers'] = [n.replace('$', '').replace(',', '') for n in numbers]
        
        # Extract visa types
        visa_patterns = [
            r'H-1B', r'H1B', r'OPT', r'CPT', r'Green Card',
            r'Citizen', r'TN Visa'
        ]
        for pattern in visa_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                entities['visa_types'].append(pattern)
        
        # Extract plan names
        plan_patterns = [
            r'PPO \d+', r'PPO-\d+', r'Plan \d+',
            r'Delta Dental', r'Delta Vision'
        ]
        for pattern in plan_patterns:
            plans = re.findall(pattern, query, re.IGNORECASE)
            entities['plan_names'].extend(plans)
        
        return entities
    
    def _determine_complexity(
        self,
        primary_type: str,
        all_types: List[str],
        entities: Dict[str, List[Any]]
    ) -> str:
        """
        Determine if query is simple or complex
        
        SIMPLE:
        - Single entity lookup
        - No calculations needed
        - Direct retrieval sufficient
        
        COMPLEX:
        - Multiple entities
        - Requires calculations
        - Needs reasoning
        """
        # Multiple detected types = complex
        if len(all_types) > 1:
            return 'complex'
        
        # Aggregation/comparison/filter = complex
        if primary_type in ['aggregation', 'comparison', 'filter', 'prediction']:
            return 'complex'
        
        # Multiple entities = complex
        total_entities = sum(len(v) for v in entities.values())
        if total_entities > 2:
            return 'complex'
        
        return 'simple'
    
    def _suggest_strategy(
        self,
        primary_type: str,
        all_types: List[str],
        entities: Dict[str, List[Any]]
    ) -> str:
        """
        Suggest retrieval strategy
        
        STRATEGIES:
        - direct: Single document by ID
        - standard: Top-k semantic search
        - filtered: Metadata filtering
        - comprehensive: Multiple collections
        - multi_source: Cross-collection comparison
        """
        # Direct lookup if employee ID present
        if entities['employee_ids']:
            return 'direct'
        
        # Multi-source for comparisons
        if primary_type == 'comparison':
            return 'multi_source'
        
        # Filtered for filter queries
        if primary_type == 'filter':
            return 'filtered'
        
        # Comprehensive for aggregations
        if primary_type == 'aggregation':
            return 'comprehensive'
        
        # Standard for everything else
        return 'standard'


def main():
    """
    Test query classifier
    """
    print("="*70)
    print("QUERY CLASSIFIER - TESTING")
    print("="*70)
    
    # Initialize classifier
    classifier = QueryClassifier()
    
    # Test queries
    test_queries = [
        "What is employee 1503's salary?",  # Simple lookup
        "How many employees have H-1B visas?",  # Aggregation
        "Compare PPO 1000 and PPO 2500 plans",  # Comparison
        "Show employees with salary > $100k",  # Filter
        "Tell me more about visa benefits",  # Conversational
        "Which insurance plan is best for families?",  # Analytical
        "Calculate the average tenure of employees hired in 2024"  # Complex
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("-"*70)
        
        result = classifier.classify(query)
        
        print(f"Type: {result['query_type']}")
        print(f"Complexity: {result['complexity']}")
        print(f"Strategy: {result['suggested_strategy']}")
        
        if result['entities']['employee_ids']:
            print(f"Employee IDs: {result['entities']['employee_ids']}")
        if result['entities']['visa_types']:
            print(f"Visa Types: {result['entities']['visa_types']}")
        if result['entities']['plan_names']:
            print(f"Plans: {result['entities']['plan_names']}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()