"""
Restructure MongoDB Data
=========================

Converts summary strings into structured documents for querying.

U
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import re
from pymongo import MongoClient
from config import Config


def parse_employee_summary(summary: str, employee_id: int) -> dict:
    """
    Parse summary string into structured fields
    
    Input: "Employee joined on 2024-08-05, Salary of $135000.0..."
    Output: {joiningDate: "2024-08-05", salary: 135000, ...}
    """
    
    # Extract joining date
    joining_match = re.search(r'joined on (\d{4}-\d{2}-\d{2})', summary)
    joining_date = joining_match.group(1) if joining_match else None
    
    # Extract employment type
    emp_type_match = re.search(r'working as a (\w+)', summary)
    employment_type = emp_type_match.group(1) if emp_type_match else None
    
    # Extract salary (fixed pattern)
    salary_match = re.search(r'Salary of \$?([\d,]+(?:\.\d+)?)', summary)
    if salary_match:
        salary_str = salary_match.group(1).replace(',', '')
        # Remove trailing period if present
        salary_str = salary_str.rstrip('.')
        try:
            salary = float(salary_str)
        except ValueError:
            salary = None
    else:
        salary = None
    
    # Extract position
    position_match = re.search(r'Current Position: ([^,]+)', summary)
    position = position_match.group(1).strip() if position_match else None
    
    # Extract assignment
    assignment_match = re.search(r'Assignment: (\w+)', summary)
    assignment = assignment_match.group(1) if assignment_match else None
    
    # Extract health insurance
    health_match = re.search(r'Health Insurance: (True|False)', summary)
    health_insurance = health_match.group(1) == 'True' if health_match else None
    
    # Extract 401k
    k401_match = re.search(r'401k: (True|False)', summary)
    has_401k = k401_match.group(1) == 'True' if k401_match else None
    
    # Extract termination date
    term_match = re.search(r'Termination Date: (\d{4}-\d{2}-\d{2})', summary)
    termination_date = term_match.group(1) if term_match else None
    
    # Extract visa information (can have multiple visas)
    visa_pattern = r'Visa type: ([^(]+)\(([^)]+)\): Entry to US: ([^,]*), Start Dates: ([^,]*), End Dates: ([^.]*)\.'
    visa_matches = re.findall(visa_pattern, summary)
    
    visas = []
    for visa_match in visa_matches:
        visa_type = visa_match[0].strip()
        visa_status = visa_match[1].strip()
        entry_date = visa_match[2].strip() if visa_match[2].strip() not in ['None', 'NaT', ''] else None
        start_date = visa_match[3].strip() if visa_match[3].strip() not in ['None', 'NaT', ''] else None
        end_date = visa_match[4].strip() if visa_match[4].strip() not in ['None', 'NaT', ''] else None
        
        visas.append({
            'visaType': visa_type,
            'status': visa_status if visa_status not in ['nan', 'None'] else 'Unknown',
            'entryToUS': entry_date,
            'startDate': start_date,
            'endDate': end_date
        })
    
    # Build structured document
    structured = {
        'employeeId': employee_id,
        'joiningDate': joining_date,
        'employmentType': employment_type,
        'salary': salary,
        'position': position,
        'assignment': assignment,
        'healthInsurance': health_insurance,
        'has401k': has_401k,
        'terminationDate': termination_date,
        'isActive': termination_date is None,
        'visas': visas
    }
    
    return structured


def restructure_database():
    """Main restructuring function"""
    
    print("=" * 70)
    print("RESTRUCTURING MONGODB DATABASE")
    print("=" * 70)
    
    # Connect to MongoDB
    client = MongoClient(Config.MONGODB_URI)
    db = client[Config.MONGODB_DB_NAME]
    
    # Load raw data
    data_file = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'HRWIKI.Employee and Visa sponsorship information.json'
    
    with open(data_file, 'r') as f:
        raw_data = json.load(f)
    
    print(f"\n📁 Loaded {len(raw_data)} employee records")
    
    # Create new structured collection
    structured_collection = db['employees_structured']
    
    # Drop if exists
    structured_collection.drop()
    print(f"🗑️  Dropped existing 'employees_structured' collection")
    
    # Parse and insert
    structured_docs = []
    
    for item in raw_data:
        employee_id = item.get('employeeid')
        summary = item.get('summary', '')
        
        # Parse summary
        structured = parse_employee_summary(summary, employee_id)
        structured_docs.append(structured)
    
    # Insert all
    structured_collection.insert_many(structured_docs)
    
    print(f"✅ Inserted {len(structured_docs)} structured documents")
    
    # Create indexes for faster queries
    print(f"\n📊 Creating indexes...")
    structured_collection.create_index('employeeId')
    structured_collection.create_index('position')
    structured_collection.create_index('salary')
    structured_collection.create_index('visas.visaType')
    structured_collection.create_index('isActive')
    
    print(f"✅ Indexes created")
    
    # Show sample
    print(f"\n📋 Sample structured document:")
    print("-" * 70)
    sample = structured_collection.find_one()
    print(json.dumps(sample, indent=2, default=str))
    
    print("\n" + "=" * 70)
    print("✅ RESTRUCTURING COMPLETE!")
    print("=" * 70)
    print(f"\nNew collection: employees_structured")
    print(f"Total documents: {structured_collection.count_documents({})}")
    
    # Test queries
    print(f"\n🧪 Testing queries:")
    print("-" * 70)
    
    # Count H-1B employees
    h1b_count = structured_collection.count_documents({'visas.visaType': 'H-1B'})
    print(f"Employees with H-1B visa: {h1b_count}")
    
    # Average salary
    pipeline = [{'$group': {'_id': None, 'avgSalary': {'$avg': '$salary'}}}]
    avg_result = list(structured_collection.aggregate(pipeline))
    if avg_result:
        print(f"Average salary: ${avg_result[0]['avgSalary']:,.2f}")
    
    # Software Developers
    dev_count = structured_collection.count_documents({'position': 'Software Developer'})
    print(f"Software Developers: {dev_count}")
    
    # Employees with health insurance and 401k
    benefits_count = structured_collection.count_documents({
        'healthInsurance': True,
        'has401k': True
    })
    print(f"Employees with health insurance AND 401k: {benefits_count}")
    
    print("=" * 70)


if __name__ == "__main__":
    restructure_database()