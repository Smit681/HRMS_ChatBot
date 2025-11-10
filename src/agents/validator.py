"""
VALIDATOR AGENT - Response Quality Assurance

Validates LLM responses before returning to user.

VALIDATION CHECKS:
1. CITATIONS - Has proper source references
2. COMPLETENESS - Fully answers the question
3. NUMERICAL ACCURACY - Numbers match calculations
4. HALLUCINATION DETECTION - No invented facts
5. CONFIDENCE SCORING - Rate answer reliability

WHY VALIDATION?
- LLMs sometimes hallucinate (make up facts)
- Numbers might be incorrect
- Important to catch errors before user sees them
- Builds trust in the system
"""

import re
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class Validator:
    """
    Validates LLM-generated responses for quality and accuracy
    
    VALIDATION PIPELINE:
    1. Check for source citations
    2. Verify completeness
    3. Validate numbers against calculations
    4. Detect hallucinations
    5. Assign confidence score
    """
    
    def __init__(self):
        """Initialize validator"""
        logger.info("Initializing Validator...")
        logger.info("✅ Validator ready!")
    
    def validate(
        self,
        response: str,
        query: str,
        sources: List[Dict[str, Any]],
        calculations: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate a response
        
        Args:
            response: LLM-generated answer
            query: Original user question
            sources: Retrieved source documents
            calculations: Calculator results (if any)
        
        Returns:
            {
                'is_valid': bool,          # Overall validity
                'confidence': float,       # Confidence score (0-1)
                'issues': List[str],       # Problems found
                'warnings': List[str],     # Minor concerns
                'checks': dict            # Individual check results
            }
        """
        logger.info("Validating response...")
        
        checks = {}
        issues = []
        warnings = []
        
        # Check 1: Citations
        citation_result = self._check_citations(response, sources)
        checks['has_citations'] = citation_result['pass']
        if not citation_result['pass']:
            warnings.append(citation_result['message'])
        
        # Check 2: Completeness
        completeness_result = self._check_completeness(response, query)
        checks['is_complete'] = completeness_result['pass']
        if not completeness_result['pass']:
            issues.append(completeness_result['message'])
        
        # Check 3: Numerical accuracy
        if calculations:
            number_result = self._check_numbers(response, calculations)
            checks['numbers_accurate'] = number_result['pass']
            if not number_result['pass']:
                issues.append(number_result['message'])
        else:
            checks['numbers_accurate'] = True
        
        # Check 4: Hallucination detection
        hallucination_result = self._detect_hallucinations(response, sources)
        checks['no_hallucinations'] = hallucination_result['pass']
        if not hallucination_result['pass']:
            issues.append(hallucination_result['message'])
        
        # Check 5: Calculate confidence
        confidence = self._calculate_confidence(checks, sources, response)
        checks['confidence'] = confidence
        
        # Determine overall validity
        is_valid = (
            len(issues) == 0 and
            confidence >= 0.5
        )
        
        result = {
            'is_valid': is_valid,
            'confidence': confidence,
            'issues': issues,
            'warnings': warnings,
            'checks': checks
        }
        
        if is_valid:
            logger.info(f"✅ Response valid (confidence: {confidence:.2f})")
        else:
            logger.warning(f"⚠️  Response has issues: {issues}")
        
        return result
    
    def _check_citations(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if response references sources
        
        GOOD: "According to the medical plan document..."
        BAD: No mention of where info came from
        """
        if not sources:
            return {
                'pass': True,
                'message': 'No sources available to cite'
            }
        
        # Look for citation indicators
        citation_patterns = [
            r'according to',
            r'based on',
            r'from the',
            r'the document states',
            r'as shown in',
            r'source:',
            r'\[source\]'
        ]
        
        has_citation = any(
            re.search(pattern, response.lower())
            for pattern in citation_patterns
        )
        
        if has_citation:
            return {
                'pass': True,
                'message': 'Response includes source references'
            }
        else:
            return {
                'pass': False,
                'message': 'Response should cite sources'
            }
    
    def _check_completeness(
        self,
        response: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Check if response fully answers the question
        
        COMPLETE: Addresses all parts of question
        INCOMPLETE: Misses key aspects
        """
        # Very basic check: response should be substantial
        if len(response.strip()) < 20:
            return {
                'pass': False,
                'message': 'Response is too short'
            }
        
        # Check for cop-out phrases
        cop_outs = [
            "i don't know",
            "i cannot answer",
            "no information available",
            "not sure",
            "unable to determine"
        ]
        
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in cop_outs):
            # These are okay if legitimately no info
            return {
                'pass': True,
                'message': 'Response indicates lack of information (acceptable)'
            }
        
        # Extract key question words
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which']
        query_lower = query.lower()
        
        # Simple heuristic: if question asks for specifics, response should have them
        if 'how many' in query_lower:
            # Should contain a number
            has_number = bool(re.search(r'\d+', response))
            if not has_number:
                return {
                    'pass': False,
                    'message': 'Query asks "how many" but response lacks numbers'
                }
        
        return {
            'pass': True,
            'message': 'Response appears complete'
        }
    
    def _check_numbers(
        self,
        response: str,
        calculations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify numbers in response match calculations
        
        ACCURATE: Numbers match calculator results
        INACCURATE: Numbers don't match (hallucinated)
        """
        calc_result = calculations.get('result')
        if calc_result is None:
            return {
                'pass': True,
                'message': 'No calculation result to verify'
            }
        
        # Extract numbers from response
        response_numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', response)
        response_numbers = [n.replace(',', '') for n in response_numbers]
        
        # Check if calc_result appears in response
        calc_str = str(calc_result)
        if calc_str in response or any(calc_str in n for n in response_numbers):
            return {
                'pass': True,
                'message': 'Numbers match calculations'
            }
        
        # Check with some tolerance for rounding
        try:
            calc_num = float(calc_result)
            for resp_num in response_numbers:
                try:
                    if abs(float(resp_num) - calc_num) < 1:
                        return {
                            'pass': True,
                            'message': 'Numbers approximately match'
                        }
                except ValueError:
                    continue
        except (ValueError, TypeError):
            pass
        
        return {
            'pass': False,
            'message': f'Response numbers don\'t match calculation: {calc_result}'
        }
    
    def _detect_hallucinations(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect if response contains invented information
        
        HALLUCINATION SIGNS:
        - Specific claims not in sources
        - Invented employee IDs/names
        - Made-up policy details
        """
        # Extract employee IDs from response
        response_emp_ids = set(re.findall(r'\b\d{4}\b', response))
        
        # Extract employee IDs from sources
        source_text = ' '.join(doc.get('text', '') for doc in sources)
        source_emp_ids = set(re.findall(r'\b\d{4}\b', source_text))
        
        # Check for employee IDs in response that aren't in sources
        hallucinated_ids = response_emp_ids - source_emp_ids
        
        if hallucinated_ids:
            return {
                'pass': False,
                'message': f'Response mentions employee IDs not in sources: {hallucinated_ids}'
            }
        
        # Check for suspiciously specific claims
        suspicious_patterns = [
            r'exactly \d+ employees',  # Unless calculated
            r'precisely \$\d+',        # Unless in source
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, response.lower()):
                # This is a weak signal, just a warning
                pass
        
        return {
            'pass': True,
            'message': 'No obvious hallucinations detected'
        }
    
    def _calculate_confidence(
        self,
        checks: Dict[str, bool],
        sources: List[Dict[str, Any]],
        response: str
    ) -> float:
        """
        Calculate overall confidence score
        
        FACTORS:
        - Number of sources (more = better)
        - Source relevance scores
        - Validation check results
        - Response length/detail
        """
        confidence = 0.5  # Start at neutral
        
        # Factor 1: Validation checks
        if checks.get('is_complete'):
            confidence += 0.15
        if checks.get('numbers_accurate'):
            confidence += 0.10
        if checks.get('no_hallucinations'):
            confidence += 0.15
        
        # Factor 2: Source quality
        if sources:
            avg_source_score = sum(s.get('score', 0) for s in sources) / len(sources)
            confidence += avg_source_score * 0.10
            
            # More sources = higher confidence
            if len(sources) >= 3:
                confidence += 0.05
        else:
            confidence -= 0.20  # No sources is bad
        
        # Factor 3: Response substance
        if len(response) > 100:
            confidence += 0.05
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return round(confidence, 2)


def main():
    """
    Test validator
    """
    print("="*70)
    print("VALIDATOR - TESTING")
    print("="*70)
    
    # Initialize validator
    validator = Validator()
    
    # Mock sources
    mock_sources = [
        {
            'text': 'Employee 1503 is a Technical Project Manager earning $135,000.',
            'score': 0.95,
            'collection': 'employee_visa'
        }
    ]
    
    # Test 1: Good response
    print("\n--- Test 1: Valid Response ---")
    good_response = "According to the employee records, employee 1503 works as a Technical Project Manager with an annual salary of $135,000."
    
    result = validator.validate(
        response=good_response,
        query="What is employee 1503's salary?",
        sources=mock_sources
    )
    
    print(f"Valid: {result['is_valid']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Issues: {result['issues']}")
    print(f"Warnings: {result['warnings']}")
    
    # Test 2: Response with hallucination
    print("\n--- Test 2: Hallucinated Response ---")
    bad_response = "Employee 9999 works as a Software Engineer earning $200,000."
    
    result = validator.validate(
        response=bad_response,
        query="What is employee 9999's salary?",
        sources=mock_sources
    )
    
    print(f"Valid: {result['is_valid']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Issues: {result['issues']}")
    
    # Test 3: Incomplete response
    print("\n--- Test 3: Incomplete Response ---")
    short_response = "Not sure."
    
    result = validator.validate(
        response=short_response,
        query="What is the dental coverage?",
        sources=mock_sources
    )
    
    print(f"Valid: {result['is_valid']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Issues: {result['issues']}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()