#!/usr/bin/env python3
"""
Test to compare the effect of empty MetadataRegistry() vs SampleMetadataRegistry()
on SQLLineageAnalyzer output.
"""

import json
import os
from src.analyzer.core.analyzer import SQLLineageAnalyzer
from src.analyzer.metadata.registry import MetadataRegistry
from src.analyzer.metadata.sample_registry import SampleMetadataRegistry

def test_metadata_registry_comparison():
    """Compare analyzer output with empty vs sample metadata registry."""
    
    # Complex derived table query for testing
    query = """
    SELECT 
        d1.department_id,
        d1.avg_salary,
        d2.dept_name,
        d2.total_employees
    FROM (
        SELECT 
            department_id, 
            AVG(salary) AS avg_salary,
            COUNT(*) AS emp_count
        FROM employees 
        GROUP BY department_id
        HAVING AVG(salary) > 50000
    ) AS d1
    JOIN (
        SELECT 
            dept_id,
            dept_name,
            COUNT(DISTINCT manager_id) AS total_employees
        FROM departments 
        WHERE active = true
        GROUP BY dept_id, dept_name
    ) AS d2 ON d1.department_id = d2.dept_id
    WHERE d1.avg_salary > 60000
    """
    
    # Test 1: With empty MetadataRegistry
    print("=== Testing with Empty MetadataRegistry ===")
    analyzer_empty = SQLLineageAnalyzer(dialect="trino")
    # Keep the default empty registry
    
    try:
        result_empty = analyzer_empty.get_lineage_chain(query, "downstream", depth=0)
        print(f"✓ Empty registry analysis completed")
        print(f"  - Number of entities: {len(result_empty.get('chains', {}))}")
        
        # Save empty result
        with open('/Users/adhirpotdar/Work/git-repos/sql-lineage/output_empty_registry.json', 'w') as f:
            json.dump(result_empty, f, indent=2)
        print(f"  - Output saved to output_empty_registry.json")
        
    except Exception as e:
        print(f"✗ Empty registry analysis failed: {e}")
        result_empty = {}
    
    # Test 2: With SampleMetadataRegistry
    print("\n=== Testing with SampleMetadataRegistry ===")
    analyzer_sample = SQLLineageAnalyzer(dialect="trino")
    # Replace with sample registry
    analyzer_sample.metadata_registry = SampleMetadataRegistry()
    
    try:
        result_sample = analyzer_sample.get_lineage_chain(query, "downstream", depth=0)
        print(f"✓ Sample registry analysis completed")
        print(f"  - Number of entities: {len(result_sample.get('chains', {}))}")
        
        # Save sample result
        with open('/Users/adhirpotdar/Work/git-repos/sql-lineage/output_sample_registry.json', 'w') as f:
            json.dump(result_sample, f, indent=2)
        print(f"  - Output saved to output_sample_registry.json")
        
    except Exception as e:
        print(f"✗ Sample registry analysis failed: {e}")
        result_sample = {}
    
    # Compare results
    print("\n=== Comparison Analysis ===")
    
    if result_empty and result_sample:
        compare_results(result_empty, result_sample)
    else:
        print("Cannot compare - one or both analyses failed")

def compare_results(empty_result, sample_result):
    """Compare the two results and highlight differences."""
    
    # Create detailed comparison report
    comparison_report = []
    comparison_report.append("# Metadata Registry Comparison Report")
    comparison_report.append("=" * 50)
    comparison_report.append("")
    
    # Compare top-level structure
    empty_chains = empty_result.get('chains', {})
    sample_chains = sample_result.get('chains', {})
    
    comparison_report.append("## 📊 Structural Comparison:")
    comparison_report.append(f"- Empty registry entities: {list(empty_chains.keys())}")
    comparison_report.append(f"- Sample registry entities: {list(sample_chains.keys())}")
    comparison_report.append("")
    
    print("📊 Structural Comparison:")
    print(f"  - Empty registry entities: {list(empty_chains.keys())}")
    print(f"  - Sample registry entities: {list(sample_chains.keys())}")
    
    # Compare metadata for each entity
    for entity_name in empty_chains.keys():
        if entity_name in sample_chains:
            comparison_report.append(f"## 🔍 Entity: {entity_name}")
            
            empty_metadata = empty_chains[entity_name].get('metadata', {})
            sample_metadata = sample_chains[entity_name].get('metadata', {})
            
            comparison_report.append(f"### Metadata Keys:")
            comparison_report.append(f"- Empty registry: {list(empty_metadata.keys())}")
            comparison_report.append(f"- Sample registry: {list(sample_metadata.keys())}")
            
            print(f"\n🔍 Entity: {entity_name}")
            print(f"  Empty registry metadata keys: {list(empty_metadata.keys())}")
            print(f"  Sample registry metadata keys: {list(sample_metadata.keys())}")
            
            # Compare specific metadata fields
            for key in set(empty_metadata.keys()) | set(sample_metadata.keys()):
                empty_val = empty_metadata.get(key, "❌ MISSING")
                sample_val = sample_metadata.get(key, "❌ MISSING")
                
                if empty_val != sample_val:
                    comparison_report.append(f"### 🔄 Difference in '{key}':")
                    comparison_report.append(f"- Empty registry: {empty_val}")
                    comparison_report.append(f"- Sample registry: {sample_val}")
                    comparison_report.append("")
            
            # Compare table_columns if present
            if 'table_columns' in empty_metadata and 'table_columns' in sample_metadata:
                empty_cols = empty_metadata['table_columns']
                sample_cols = sample_metadata['table_columns']
                
                comparison_report.append(f"### Column Information:")
                comparison_report.append(f"- Column count - Empty: {len(empty_cols)}, Sample: {len(sample_cols)}")
                
                print(f"  Column count - Empty: {len(empty_cols)}, Sample: {len(sample_cols)}")
                
                # Show first few columns for comparison
                if empty_cols and sample_cols:
                    comparison_report.append(f"- First empty column: {empty_cols[0]}")
                    comparison_report.append(f"- First sample column: {sample_cols[0]}")
                    comparison_report.append("")
                    
                    print(f"  First empty column: {empty_cols[0]}")
                    print(f"  First sample column: {sample_cols[0]}")
    
    # Compare analysis metadata
    empty_analysis = empty_result.get('summary', {})
    sample_analysis = sample_result.get('summary', {})
    
    comparison_report.append("## 📈 Analysis Summary:")
    comparison_report.append(f"- Empty registry: {empty_analysis}")
    comparison_report.append(f"- Sample registry: {sample_analysis}")
    comparison_report.append("")
    
    print(f"\n📈 Analysis Summary:")
    print(f"  Empty registry: {empty_analysis}")
    print(f"  Sample registry: {sample_analysis}")
    
    # Save detailed comparison report
    report_content = "\n".join(comparison_report)
    with open('/Users/adhirpotdar/Work/git-repos/sql-lineage/metadata_comparison_report.txt', 'w') as f:
        f.write(report_content)
    
    print(f"\n📝 Detailed comparison report saved to: metadata_comparison_report.txt")
    return report_content

if __name__ == "__main__":
    test_metadata_registry_comparison()