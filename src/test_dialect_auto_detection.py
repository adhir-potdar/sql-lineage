#!/usr/bin/env python3
"""Test dialect auto-detection functionality."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# from analyzer import SQLLineageAnalyzer  # Commented out - now using lineage_chain_builder
from analyzer.core.analyzers.lineage_chain_builder import LineageChainBuilder
from analyzer import SQLLineageAnalyzer  # Still needed for creating the analyzer instance


def test_top_clause_auto_detection():
    """Test auto-detection of TOP clause from Trino to TSQL."""
    print("🔍 Testing TOP clause auto-detection")
    print("=" * 50)
    
    # Query with TOP clause that should trigger auto-detection
    sql_with_top = """
    WITH select_step1 as (
        SELECT
          "orders"."order_id" AS "order_id",
            CAST(YEAR("order_date") AS varchar) AS "year of order"
           FROM
          "customer_demo"."orders"
    )
    SELECT TOP 100 * FROM select_step1
    """
    
    # Start with Trino dialect (should fail and auto-correct to TSQL)
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # NEW: Use LineageChainBuilder instead of direct analyzer
    lineage_builder = LineageChainBuilder(dialect="trino", main_analyzer=analyzer)
    
    try:
        print(f"🚀 Analyzing with initial dialect: trino using LineageChainBuilder")
        # result = analyzer.analyze(sql_with_top)  # Commented out - now using lineage_chain_builder
        chain_result = lineage_builder.get_lineage_chain(sql_with_top, "downstream", 0)
        
        print(f"✅ Analysis successful!")
        print(f"   • Final dialect used: {lineage_builder.dialect}")
        print(f"   • Chain result keys: {list(chain_result.keys())}")
        # print(f"   • Tables found: {len(result.table_lineage.upstream.get('QUERY_RESULT', []))}")
        # print(f"   • Columns found: {len(result.column_lineage.upstream)}")
        
        # Check if dialect was auto-corrected (should be 'tsql' now instead of 'trino')
        if lineage_builder.dialect == 'tsql' and analyzer.dialect == 'tsql':
            print(f"\n🔄 Auto-correction occurred:")
            print(f"   • trino → {lineage_builder.dialect}")
            print(f"   • Both lineage_builder and analyzer dialects updated")
            return True
        else:
            print(f"❌ Expected auto-correction to tsql, but got:")
            print(f"   • lineage_builder.dialect: {lineage_builder.dialect}")
            print(f"   • analyzer.dialect: {analyzer.dialect}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_genuine_syntax_error():
    """Test that genuine syntax errors don't trigger auto-detection."""
    print("\n🔍 Testing genuine syntax error (should NOT auto-detect)")
    print("=" * 50)
    
    # Query with genuine syntax error (missing FROM)
    sql_with_error = "SELECT col1, col2 WHERE col1 = 'value'"
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    lineage_builder = LineageChainBuilder(dialect="trino", main_analyzer=analyzer)
    
    try:
        print(f"🚀 Analyzing query with syntax error using LineageChainBuilder")
        # result = analyzer.analyze(sql_with_error)  # Commented out - now using lineage_chain_builder
        chain_result = lineage_builder.get_lineage_chain(sql_with_error, "downstream", 0)
        print(f"❌ Expected failure but analysis succeeded")
        return False
        
    except Exception as e:
        # Check that no auto-correction was attempted
        correction_info = analyzer.get_dialect_correction_info()
        if correction_info:
            print(f"❌ Unexpected auto-correction occurred: {correction_info}")
            return False
        else:
            print(f"✅ Correctly identified as genuine syntax error (no auto-detection)")
            print(f"   • Error: {str(e)[:100]}...")
            return True


def test_limit_clause_auto_detection():
    """Test auto-detection of LIMIT clause patterns.""" 
    print("\n🔍 Testing LIMIT clause (should work with current dialect)")
    print("=" * 50)
    
    # Query with LIMIT clause - should work with Trino
    sql_with_limit = "SELECT * FROM customer_demo.orders LIMIT 50"
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    lineage_builder = LineageChainBuilder(dialect="trino", main_analyzer=analyzer)
    
    try:
        print(f"🚀 Analyzing LIMIT query with Trino dialect using LineageChainBuilder")
        # result = analyzer.analyze(sql_with_limit)  # Commented out - now using lineage_chain_builder
        chain_result = lineage_builder.get_lineage_chain(sql_with_limit, "downstream", 0)
        
        print(f"✅ Analysis successful (no auto-correction needed)")
        print(f"   • Dialect: {lineage_builder.dialect}")
        
        # Should NOT have auto-correction info
        correction_info = analyzer.get_dialect_correction_info()
        if correction_info:
            print(f"❌ Unexpected auto-correction: {correction_info}")
            return False
        else:
            print(f"   • No auto-correction (as expected)")
            return True
            
    except Exception as e:
        print(f"❌ Unexpected failure: {str(e)}")
        return False


def main():
    """Run all auto-detection tests."""
    print("🚀 Dialect Auto-Detection Tests")
    print("=" * 60)
    
    tests = [
        ("TOP clause auto-detection", test_top_clause_auto_detection),
        ("Genuine syntax error", test_genuine_syntax_error),
        ("LIMIT clause (no correction needed)", test_limit_clause_auto_detection),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    print(f"\n📊 Test Results:")
    print("=" * 40)
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🏆 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All dialect auto-detection tests passed!")
        return 0
    else:
        print("💥 Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())