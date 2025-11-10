"""
CALCULATOR AGENT - Precise Numerical Operations

Performs exact calculations that LLMs might get wrong.

OPERATIONS:
1. COUNT - Count entities
2. SUM - Total amounts
3. AVERAGE - Mean values
4. MIN/MAX - Find extremes
5. DATE_DIFF - Time between dates
6. PERCENTAGE - Ratios and percentages

WHY SEPARATE CALCULATOR?
- LLMs are bad at math (hallucinate numbers)
- Need exact counts and sums
- Date arithmetic is error-prone
- Verification requires precision
"""

from typing import Dict, Any, List, Union
from datetime import datetime, timedelta
from dateutil import parser
import re
import logging

logger = logging.getLogger(__name__)


class Calculator:
    """
    Performs precise calculations on retrieved data
    
    CRITICAL: Always use this for numerical operations
    instead of trusting LLM arithmetic.
    
    SUPPORTED OPERATIONS:
    - count: Count items
    - sum: Total numeric values
    - average: Mean of values
    - min/max: Find extremes
    - date_diff: Days/months between dates
    - percentage: Calculate ratios
    """
    
    def __init__(self):
        """Initialize calculator"""
        logger.info("Initializing Calculator...")
        logger.info("✅ Calculator ready!")
    
    def calculate(
        self,
        operation: str,
        data: Union[List[Dict[str, Any]], List[Any]],
        field: str = None
    ) -> Dict[str, Any]:
        """
        Perform calculation on data
        
        Args:
            operation: Type of calculation
            data: List of documents or values
            field: Field to extract (if data is list of dicts)
        
        Returns:
            {
                'result': Any,           # Calculation result
                'operation': str,        # Operation performed
                'count': int,           # Number of items processed
                'details': dict         # Additional info
            }
        """
        logger.info(f"Calculating: {operation}")
        
        # Map operation to method
        operations = {
            'count': self._count,
            'sum': self._sum,
            'average': self._average,
            'min': self._min,
            'max': self._max,
            'date_diff': self._date_diff,
            'percentage': self._percentage
        }
        
        if operation not in operations:
            logger.error(f"Unknown operation: {operation}")
            return {
                'result': None,
                'operation': operation,
                'count': 0,
                'error': f"Unknown operation: {operation}"
            }
        
        # Execute operation
        try:
            method = operations[operation]
            result = method(data, field)
            logger.info(f"Result: {result['result']}")
            return result
        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            return {
                'result': None,
                'operation': operation,
                'count': len(data) if data else 0,
                'error': str(e)
            }
    
    def _count(
        self,
        data: List[Any],
        field: str = None
    ) -> Dict[str, Any]:
        """
        Count items
        
        EXAMPLES:
        - How many employees?
        - Count H-1B visa holders
        """
        count = len(data)
        
        return {
            'result': count,
            'operation': 'count',
            'count': count,
            'details': {
                'total_items': count
            }
        }
    
    def _sum(
        self,
        data: List[Any],
        field: str
    ) -> Dict[str, Any]:
        """
        Sum numeric values
        
        EXAMPLES:
        - Total salaries
        - Sum of premiums
        """
        values = self._extract_numeric_values(data, field)
        total = sum(values)
        
        return {
            'result': total,
            'operation': 'sum',
            'count': len(values),
            'details': {
                'total': total,
                'values_summed': len(values),
                'min': min(values) if values else 0,
                'max': max(values) if values else 0
            }
        }
    
    def _average(
        self,
        data: List[Any],
        field: str
    ) -> Dict[str, Any]:
        """
        Calculate average
        
        EXAMPLES:
        - Average salary
        - Mean tenure
        """
        values = self._extract_numeric_values(data, field)
        
        if not values:
            return {
                'result': 0,
                'operation': 'average',
                'count': 0,
                'details': {'error': 'No numeric values found'}
            }
        
        avg = sum(values) / len(values)
        
        return {
            'result': round(avg, 2),
            'operation': 'average',
            'count': len(values),
            'details': {
                'average': round(avg, 2),
                'sum': sum(values),
                'count': len(values),
                'min': min(values),
                'max': max(values)
            }
        }
    
    def _min(
        self,
        data: List[Any],
        field: str
    ) -> Dict[str, Any]:
        """
        Find minimum value
        
        EXAMPLES:
        - Lowest salary
        - Earliest date
        """
        values = self._extract_numeric_values(data, field)
        
        if not values:
            return {
                'result': None,
                'operation': 'min',
                'count': 0,
                'details': {'error': 'No values found'}
            }
        
        min_val = min(values)
        
        return {
            'result': min_val,
            'operation': 'min',
            'count': len(values),
            'details': {
                'minimum': min_val,
                'total_values': len(values)
            }
        }
    
    def _max(
        self,
        data: List[Any],
        field: str
    ) -> Dict[str, Any]:
        """
        Find maximum value
        
        EXAMPLES:
        - Highest salary
        - Latest date
        """
        values = self._extract_numeric_values(data, field)
        
        if not values:
            return {
                'result': None,
                'operation': 'max',
                'count': 0,
                'details': {'error': 'No values found'}
            }
        
        max_val = max(values)
        
        return {
            'result': max_val,
            'operation': 'max',
            'count': len(values),
            'details': {
                'maximum': max_val,
                'total_values': len(values)
            }
        }
    
    def _date_diff(
        self,
        data: List[Any],
        field: str
    ) -> Dict[str, Any]:
        """
        Calculate days between dates
        
        EXAMPLES:
        - Visa expiration in X days
        - Tenure in months
        """
        # Extract dates
        dates = []
        for item in data:
            if isinstance(item, dict):
                date_str = item.get(field)
            else:
                date_str = item
            
            if date_str:
                try:
                    date_obj = parser.parse(str(date_str))
                    dates.append(date_obj)
                except:
                    pass
        
        if len(dates) < 2:
            return {
                'result': None,
                'operation': 'date_diff',
                'count': len(dates),
                'details': {'error': 'Need at least 2 dates'}
            }
        
        # Calculate difference between first and last
        dates.sort()
        diff_days = (dates[-1] - dates[0]).days
        diff_months = diff_days // 30
        
        return {
            'result': diff_days,
            'operation': 'date_diff',
            'count': len(dates),
            'details': {
                'days': diff_days,
                'months': diff_months,
                'start_date': dates[0].isoformat(),
                'end_date': dates[-1].isoformat()
            }
        }
    
    def _percentage(
        self,
        data: List[Any],
        field: str = None
    ) -> Dict[str, Any]:
        """
        Calculate percentage
        
        EXAMPLES:
        - What % of employees have H-1B?
        - Percentage increase
        """
        if isinstance(data, list) and len(data) == 2:
            # Two numbers: calculate percentage
            part = float(data[0])
            whole = float(data[1])
            
            if whole == 0:
                return {
                    'result': 0,
                    'operation': 'percentage',
                    'count': 2,
                    'details': {'error': 'Division by zero'}
                }
            
            pct = (part / whole) * 100
            
            return {
                'result': round(pct, 2),
                'operation': 'percentage',
                'count': 2,
                'details': {
                    'percentage': round(pct, 2),
                    'part': part,
                    'whole': whole
                }
            }
        
        return {
            'result': None,
            'operation': 'percentage',
            'count': len(data) if data else 0,
            'details': {'error': 'Need exactly 2 numbers for percentage'}
        }
    
    def _extract_numeric_values(
        self,
        data: List[Any],
        field: str
    ) -> List[float]:
        """
        Extract numeric values from data
        
        Handles:
        - List of numbers
        - List of dicts with numeric field
        - Strings with numbers
        """
        values = []
        
        for item in data:
            if isinstance(item, dict):
                # Extract field from dict
                value = item.get(field)
            else:
                # Use item directly
                value = item
            
            # Convert to number
            if value is not None:
                try:
                    # Remove currency symbols and commas
                    if isinstance(value, str):
                        value = re.sub(r'[$,]', '', value)
                    num = float(value)
                    values.append(num)
                except (ValueError, TypeError):
                    pass
        
        return values


def main():
    """
    Test calculator
    """
    print("="*70)
    print("CALCULATOR - TESTING")
    print("="*70)
    
    # Initialize calculator
    calc = Calculator()
    
    # Test data
    employee_data = [
        {'employeeid': 1503, 'salary': 135000},
        {'employeeid': 1504, 'salary': 155000},
        {'employeeid': 1505, 'salary': 175000},
        {'employeeid': 1506, 'salary': 95000},
    ]
    
    # Test 1: Count
    print("\n--- Test 1: COUNT ---")
    result = calc.calculate('count', employee_data)
    print(f"Count: {result['result']}")
    
    # Test 2: Sum
    print("\n--- Test 2: SUM ---")
    result = calc.calculate('sum', employee_data, 'salary')
    print(f"Total salary: ${result['result']:,}")
    print(f"Details: {result['details']}")
    
    # Test 3: Average
    print("\n--- Test 3: AVERAGE ---")
    result = calc.calculate('average', employee_data, 'salary')
    print(f"Average salary: ${result['result']:,}")
    print(f"Min: ${result['details']['min']:,}, Max: ${result['details']['max']:,}")
    
    # Test 4: Min/Max
    print("\n--- Test 4: MIN/MAX ---")
    result_min = calc.calculate('min', employee_data, 'salary')
    result_max = calc.calculate('max', employee_data, 'salary')
    print(f"Lowest salary: ${result_min['result']:,}")
    print(f"Highest salary: ${result_max['result']:,}")
    
    # Test 5: Percentage
    print("\n--- Test 5: PERCENTAGE ---")
    result = calc.calculate('percentage', [30, 100])
    print(f"30 out of 100 = {result['result']}%")
    
    # Test 6: Date difference
    print("\n--- Test 6: DATE DIFF ---")
    dates = ['2024-01-01', '2025-01-01']
    result = calc.calculate('date_diff', dates, None)
    print(f"Days between: {result['result']}")
    print(f"Months: {result['details']['months']}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")


if __name__ == "__main__":
    main()