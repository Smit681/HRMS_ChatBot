import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

def load_json_file(filepath):
    """Load JSON file and return data"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def assess_employee_data(data):
    """Assess data quality for employee records"""
    issues = []
    stats = defaultdict(int)
    
    for idx, employee in enumerate(data):
        emp_id = employee.get('employeeid', f'Unknown_{idx}')
        summary = employee.get('summary', '')
        
        # Check for missing joining dates
        if 'NaT' in summary or 'joined on NaT' in summary:
            issues.append({
                'employee_id': emp_id,
                'issue_type': 'Missing joining date',
                'field': 'joiningDate',
                'severity': 'High'
            })
            stats['missing_joining_date'] += 1
        
        # Check for missing salaries
        if 'nan' in summary.lower() or 'salary of $nan' in summary.lower():
            issues.append({
                'employee_id': emp_id,
                'issue_type': 'Missing salary',
                'field': 'salary',
                'severity': 'Medium'
            })
            stats['missing_salary'] += 1
        
        # Check for terminated employees
        if 'Termination Date: NaT' not in summary and 'Termination Date:' in summary:
            stats['terminated_employees'] += 1
        
        # Check for visa information
        if 'Visa type:' not in summary:
            issues.append({
                'employee_id': emp_id,
                'issue_type': 'Missing visa information',
                'field': 'visa',
                'severity': 'High'
            })
            stats['missing_visa'] += 1
        
        # Check for inconsistent dates (Entry to US vs visa dates)
        if 'Entry to US: None' in summary and 'Start Dates:' in summary:
            # Has visa dates but no entry date - potential inconsistency
            stats['inconsistent_entry_dates'] += 1
    
    stats['total_employees'] = len(data)
    return issues, stats

def assess_insurance_data(data, plan_type):
    """Assess data quality for insurance plans"""
    issues = []
    stats = defaultdict(int)
    
    for idx, record in enumerate(data):
        content = record.get('content', '')
        
        # Check for missing critical information
        if not content or len(content) < 100:
            issues.append({
                'record_id': idx,
                'issue_type': f'Incomplete {plan_type} plan data',
                'field': 'content',
                'severity': 'High'
            })
            stats['incomplete_records'] += 1
        
        # Check for pricing information
        if '$' not in content and plan_type != 'employment_agreement':
            issues.append({
                'record_id': idx,
                'issue_type': f'Missing pricing in {plan_type}',
                'field': 'pricing',
                'severity': 'Medium'
            })
            stats['missing_pricing'] += 1
    
    stats['total_records'] = len(data)
    return issues, stats

def generate_report(all_issues, all_stats):
    """Generate comprehensive quality report"""
    report = []
    
    report.append("=" * 70)
    report.append("DATA QUALITY ASSESSMENT REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY STATISTICS:")
    report.append("-" * 70)
    for category, stats in all_stats.items():
        report.append(f"\n{category.upper()}:")
        for key, value in stats.items():
            report.append(f"  • {key.replace('_', ' ').title()}: {value}")
    
    report.append("\n" + "=" * 70)
    report.append("ISSUES FOUND:")
    report.append("-" * 70)
    
    # Group issues by severity
    high_severity = [i for i in all_issues if i.get('severity') == 'High']
    medium_severity = [i for i in all_issues if i.get('severity') == 'Medium']
    
    report.append(f"\nHIGH SEVERITY ISSUES: {len(high_severity)}")
    for issue in high_severity[:10]:  # Show first 10
        report.append(f"  • {issue['issue_type']} - ID: {issue.get('employee_id', issue.get('record_id'))}")
    
    report.append(f"\nMEDIUM SEVERITY ISSUES: {len(medium_severity)}")
    for issue in medium_severity[:10]:  # Show first 10
        report.append(f"  • {issue['issue_type']} - ID: {issue.get('employee_id', issue.get('record_id'))}")
    
    report.append("\n" + "=" * 70)
    report.append("RECOMMENDATIONS:")
    report.append("-" * 70)
    report.append("1. Missing joining dates: Exclude from tenure calculations")
    report.append("2. Missing salaries: Mark as 'Compensation not disclosed'")
    report.append("3. Missing visa data: Contact HR to update records")
    report.append("4. Inconsistent dates: Manual review required")
    report.append("=" * 70)
    
    return "\n".join(report)

def main():
    """Run data quality assessment"""
    raw_data_path = Path("data/raw")
    
    all_issues = []
    all_stats = {}
    
    # Assess employee data
    print("📊 Assessing employee data...")
    employee_data = load_json_file(raw_data_path / "HRWIKI.Employee and Visa sponsorship information.json")
    emp_issues, emp_stats = assess_employee_data(employee_data)
    all_issues.extend(emp_issues)
    all_stats['employees'] = emp_stats
    
    # Assess insurance plans
    print("📊 Assessing medical plans...")
    medical_1000 = load_json_file(raw_data_path / "HRWIKI.1000 PLAN SBC - ITLIZE GLOBAL.json")
    med_issues_1000, med_stats_1000 = assess_insurance_data(medical_1000, "medical_1000")
    all_issues.extend(med_issues_1000)
    all_stats['medical_1000_plan'] = med_stats_1000
    
    print("📊 Assessing dental plans...")
    dental = load_json_file(raw_data_path / "HRWIKI.Delta Dental Benefit Summary.json")
    dental_issues, dental_stats = assess_insurance_data(dental, "dental")
    all_issues.extend(dental_issues)
    all_stats['dental_plan'] = dental_stats
    
    print("📊 Assessing vision plans...")
    vision = load_json_file(raw_data_path / "HRWIKI.Delta Vision Benefit Summary.json")
    vision_issues, vision_stats = assess_insurance_data(vision, "vision")
    all_issues.extend(vision_issues)
    all_stats['vision_plan'] = vision_stats
    
    print("📊 Assessing employment agreements...")
    employment = load_json_file(raw_data_path / "HRWIKI.EmploymentAgreement.json")
    emp_agr_issues, emp_agr_stats = assess_insurance_data(employment, "employment_agreement")
    all_issues.extend(emp_agr_issues)
    all_stats['employment_agreement'] = emp_agr_stats
    
    # Generate report
    report = generate_report(all_issues, all_stats)
    print("\n" + report)
    
    # Save report to file
    report_path = Path("data/processed/data_quality_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save issues as JSON for programmatic access
    issues_path = Path("data/processed/data_quality_issues.json")
    with open(issues_path, 'w') as f:
        json.dump({
            'issues': all_issues,
            'statistics': all_stats
        }, f, indent=2)
    
    print(f"\n✓ Report saved to: {report_path}")
    print(f"✓ Issues saved to: {issues_path}")
    print(f"\n📋 Total issues found: {len(all_issues)}")

if __name__ == "__main__":
    main()