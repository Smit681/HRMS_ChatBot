"""
Entity Tracker - Track Entities Across Conversation
===================================================

Tracks mentioned entities (employee IDs) across conversation
to resolve pronoun references in follow-up queries.

Simple in-memory storage per user (no session management needed).
"""

import re
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityTracker:
    """
    Track entities mentioned in conversation per user
    
    Stores:
    - Last employee ID mentioned
    - Timestamp (for auto-expiry after 10 minutes)
    """
    
    def __init__(self):
        """Initialize entity tracker with in-memory storage"""
        # Structure: {user_email: {'employee_id': str, 'timestamp': datetime}}
        self._user_entities = {}
        
        # Auto-expire after 10 minutes of inactivity
        self.EXPIRY_MINUTES = 10
        
        logger.info("✅ Entity Tracker initialized")
    
    def extract_employee_id(self, text: str) -> Optional[str]:
        """
        Extract employee ID from text
        
        Patterns:
        - "employee 1520"
        - "employee with id 1520"
        - "emp 1520"
        - "ID 1520"
        - "1520" (4-digit number in context)
        
        Args:
            text: Query or response text
        
        Returns:
            Employee ID as string or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Pattern 1: "employee [with id] XXXX"
        match = re.search(r'employee\s+(?:with\s+)?(?:id\s+)?(\d{4})', text_lower)
        if match:
            emp_id = match.group(1)
            logger.debug(f"Extracted employee ID (pattern 1): {emp_id}")
            return emp_id
        
        # Pattern 2: "emp XXXX"
        match = re.search(r'emp\s+(\d{4})', text_lower)
        if match:
            emp_id = match.group(1)
            logger.debug(f"Extracted employee ID (pattern 2): {emp_id}")
            return emp_id
        
        # Pattern 3: "id XXXX" or "ID: XXXX"
        match = re.search(r'id\s*:?\s*(\d{4})', text_lower)
        if match:
            emp_id = match.group(1)
            logger.debug(f"Extracted employee ID (pattern 3): {emp_id}")
            return emp_id
        
        # Pattern 4: Standalone 4-digit number in employee context
        if any(keyword in text_lower for keyword in ['employee', 'worker', 'staff', 'person', 'joined', 'salary', 'visa']):
            match = re.search(r'\b(\d{4})\b', text)
            if match:
                emp_id = match.group(1)
                logger.debug(f"Extracted employee ID (pattern 4): {emp_id}")
                return emp_id
        
        logger.debug(f"No employee ID found in: {text[:50]}...")
        return None
    
    def has_pronoun_reference(self, text: str) -> bool:
        """
        Check if query contains pronoun references
        
        Pronouns:
        - their, his, her, them, they
        - that employee, this employee
        - the employee, the person
        - same employee
        
        Args:
            text: Query text
        
        Returns:
            True if contains pronoun reference
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Direct pronouns (possessive and object)
        pronouns = ['their', 'his', 'her', 'them', 'they', 'theirs']
        for pronoun in pronouns:
            # Use word boundary to avoid partial matches
            if re.search(rf'\b{pronoun}\b', text_lower):
                logger.debug(f"Found pronoun: {pronoun}")
                return True
        
        # Demonstrative references
        references = [
            'that employee',
            'this employee',
            'the employee',
            'that person',
            'this person',
            'the person',
            'same employee',
            'same person',
            'same worker'
        ]
        for ref in references:
            if ref in text_lower:
                logger.debug(f"Found reference: {ref}")
                return True
        
        return False
    
    def update_entity(self, user_email: str, query: str, response: str = None):
        """
        Update tracked entity for user
        
        Extracts employee ID from query or response and stores it
        
        Args:
            user_email: User's email
            query: User's query
            response: Bot's response (optional)
        """
        print("\n\n\n\n\n\n\n\nUpdating entity")
        # Try to extract from query first
        employee_id = self.extract_employee_id(query)
        
        # If not in query, try response
        if not employee_id and response:
            employee_id = self.extract_employee_id(response)
        
        # If found, update storage
        if employee_id:
            self._user_entities[user_email] = {
                'employee_id': employee_id,
                'timestamp': datetime.now()
            }
            logger.info(f"📌 Tracked entity for {user_email}: employee_id={employee_id}")
        else:
            logger.debug(f"No entity found in query/response for {user_email}")
    
    def get_last_employee_id(self, user_email: str) -> Optional[str]:
        """
        Get last mentioned employee ID for user
        
        Args:
            user_email: User's email
        
        Returns:
            Employee ID or None if expired/not found
        """
        if user_email not in self._user_entities:
            logger.debug(f"No tracked entity for {user_email}")
            return None
        
        entity_data = self._user_entities[user_email]
        
        # Check if expired (10 minutes)
        time_diff = datetime.now() - entity_data['timestamp']
        if time_diff > timedelta(minutes=self.EXPIRY_MINUTES):
            logger.info(f"⏰ Entity expired for {user_email} (>10 min)")
            del self._user_entities[user_email]
            return None
        
        employee_id = entity_data['employee_id']
        logger.info(f"🔍 Retrieved tracked entity for {user_email}: employee_id={employee_id}")
        return employee_id
    
    def resolve_query(self, user_email: str, query: str) -> str:
        """
        Resolve pronoun references in query
        
        Args:
            user_email: User's email
            query: Original query with potential pronouns
        
        Returns:
            Resolved query with explicit employee ID
        """
        logger.info(f"🔍 Resolving query for {user_email}: '{query}'")
        
        # Step 1: Check if query has pronoun reference
        has_pronoun = self.has_pronoun_reference(query)
        logger.info(f"   Has pronoun reference: {has_pronoun}")
        
        if not has_pronoun:
            logger.info(f"   No pronoun detected, returning original query")
            return query
        
        # Step 2: Get last employee ID
        employee_id = self.get_last_employee_id(user_email)
        logger.info(f"   Last tracked employee_id: {employee_id}")
        
        if not employee_id:
            logger.warning(f"   ⚠️  Pronoun detected but no tracked employee ID")
            return query
        
        # Step 3: Replace pronouns
        resolved_query = self._replace_pronouns(query, employee_id)
        
        logger.info(f"   ✅ Resolved: '{query}' → '{resolved_query}'")
        return resolved_query
    
    def _replace_pronouns(self, query: str, employee_id: str) -> str:
        """
        Replace pronouns with explicit employee reference
        
        Args:
            query: Original query
            employee_id: Employee ID to insert
        
        Returns:
            Query with pronouns replaced
        """
        resolved = query
        
        logger.debug(f"Replacing pronouns in: '{query}' with employee_id={employee_id}")
        
        # Apply replacements in order (most specific first)
        
        # Demonstrative references (2-word phrases)
        resolved = re.sub(r'\bthat\s+employee\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bthis\s+employee\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bthe\s+employee\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bthat\s+person\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bthis\s+person\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bthe\s+person\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bsame\s+employee\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bsame\s+person\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bsame\s+worker\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        
        # Possessive pronouns
        resolved = re.sub(r'\btheir\b', f"employee {employee_id}'s", resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bhis\b', f"employee {employee_id}'s", resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bher\b', f"employee {employee_id}'s", resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\btheirs\b', f"employee {employee_id}'s", resolved, flags=re.IGNORECASE)
        
        # Object pronouns
        resolved = re.sub(r'\bthem\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        resolved = re.sub(r'\bthey\b', f'employee {employee_id}', resolved, flags=re.IGNORECASE)
        
        logger.debug(f"After replacement: '{resolved}'")
        
        return resolved
    
    def clear_user_entities(self, user_email: str):
        """Clear tracked entities for user"""
        if user_email in self._user_entities:
            del self._user_entities[user_email]
            logger.info(f"🗑️  Cleared entities for {user_email}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        return {
            'total_tracked_users': len(self._user_entities),
            'users': list(self._user_entities.keys()),
            'expiry_minutes': self.EXPIRY_MINUTES
        }


# Singleton instance
_entity_tracker_instance = None

def get_entity_tracker() -> EntityTracker:
    """Get singleton entity tracker instance"""
    global _entity_tracker_instance
    
    if _entity_tracker_instance is None:
        _entity_tracker_instance = EntityTracker()
    
    return _entity_tracker_instance


def main():
    """Test entity tracker"""
    print("=" * 70)
    print("ENTITY TRACKER - COMPREHENSIVE TESTING")
    print("=" * 70)
    
    tracker = EntityTracker()
    test_user = "test@example.com"
    
    # Test 1: Extract employee ID
    print("\n--- Test 1: Extract Employee ID ---")
    test_cases = [
        "When did employee 1520 join?",
        "What is employee with id 1503's salary?",
        "Tell me about emp 1504",
        "Employee ID: 1528 visa status",
        "Employee 1520 joined on 2022-04-11."
    ]
    
    for query in test_cases:
        emp_id = tracker.extract_employee_id(query)
        print(f"Text: {query}")
        print(f"  → Employee ID: {emp_id}\n")
    
    # Test 2: Detect pronouns
    print("\n--- Test 2: Detect Pronoun References ---")
    test_cases = [
        "What is their visa status?",
        "Tell me about his salary",
        "What about that employee?",
        "How many years has the person been here?",
        "What is the salary?" # Should be False (no pronoun)
    ]
    
    for query in test_cases:
        has_pronoun = tracker.has_pronoun_reference(query)
        print(f"Query: {query}")
        print(f"  → Has pronoun: {has_pronoun}\n")
    
    # Test 3: Pronoun replacement (isolated)
    print("\n--- Test 3: Pronoun Replacement (Direct) ---")
    test_cases = [
        "What is their visa status?",
        "Tell me about his salary",
        "What about that employee?",
        "How long has the person been here?",
    ]
    
    for query in test_cases:
        resolved = tracker._replace_pronouns(query, "1520")
        print(f"Original:  {query}")
        print(f"Resolved:  {resolved}\n")
    
    # Test 4: Full conversation flow
    print("\n--- Test 4: Full Conversation Flow ---")
    
    # Turn 1: Ask about employee 1520
    query1 = "When did employee 1520 join the company?"
    response1 = "Employee 1520 joined on 2022-04-11."
    
    print(f"Turn 1:")
    print(f"  User: {query1}")
    print(f"  Bot: {response1}")
    
    tracker.update_entity(test_user, query1, response1)
    tracked_id = tracker.get_last_employee_id(test_user)
    print(f"  → Tracked: {tracked_id}\n")
    
    # Turn 2: Follow-up with pronoun
    query2 = "What is their current visa status?"
    
    print(f"Turn 2:")
    print(f"  User: {query2}")
    
    resolved = tracker.resolve_query(test_user, query2)
    print(f"  → Resolved: {resolved}\n")
    
    # Turn 3: Another pronoun variant
    query3 = "Tell me about his salary"
    
    print(f"Turn 3:")
    print(f"  User: {query3}")
    
    resolved = tracker.resolve_query(test_user, query3)
    print(f"  → Resolved: {resolved}\n")
    
    # Test 5: Stats
    print("\n--- Test 5: Stats ---")
    stats = tracker.get_stats()
    print(f"Tracked users: {stats['total_tracked_users']}")
    print(f"Users: {stats['users']}")
    print(f"Expiry time: {stats['expiry_minutes']} minutes")
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()