#!/usr/bin/env python3
"""
Contract Analyzer Improvement Testing Script
This script tests and validates the improvements made to the contract analysis system.
"""

import os
import sys
import time
import json
import re
from typing import List, Dict, Any
import unittest

# Add the current directory to Python path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import functions from our enhanced app.py
    from app import (
        extract_monetary_values,
        filter_potentially_risky_sentences,
        clean_ocr_text,
        post_process_ocr_text,
        extract_text_with_easyocr,
        analyze_contract
    )
    print("✓ Successfully imported enhanced functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the same directory as app.py")
    sys.exit(1)

class ContractAnalyzerTests(unittest.TestCase):
    """Test suite for contract analyzer improvements."""
    
    def setUp(self):
        """Set up test data for each test."""
        self.test_monetary_texts = [
            "Late payment fee of $500 will be charged",
            "Monthly rent is $2,500.00",
            "Interest rate of 5% per annum",
            "Penalty ranges from $1,000 to $5,000",
            "Fee: USD 10,000",
            "5% late fee applies",
            "Minimum payment $100/month",
            "Cost is 1,500 euros",
            "Security deposit: $2,500",
            "Administrative fee of twenty-five dollars ($25)",
            "Liquidated damages of $10k",
            "Premium of 2.5% annually"
        ]
        
        self.test_contract_sentences = [
            "This Agreement shall terminate upon 30 days written notice.",
            "Late payment penalty of $500 applies after 5 business days.",
            "The Company shall indemnify and hold harmless the Client.",
            "All intellectual property rights transfer upon final payment.",
            "Confidential information must not be disclosed for 2 years.",
            "Standard boilerplate language for governing law applies.",
            "This section contains definitions of terms used herein.",
            "Force majeure events excuse performance delays.",
            "Arbitration is required for dispute resolution.",
            "Liquidated damages of $10,000 for each breach.",
            "The parties agree to execute this agreement in counterparts.",
            "Headings are for convenience only and not binding.",
            "Client is responsible for all taxes and fees.",
            "Termination triggers immediate payment of outstanding amounts.",
            "Non-compete restrictions apply for 12 months after termination."
        ]
        
        self.test_ocr_text = """
        RENT AL    AGREE MENT
        
        Month ly   rent:   $ 2 , 5 0 0 . 0 0
        Late   fee:   5 %   per   month
        Security   depos it:   $ 5 , 0 0 0
        
        TERMIN ATION   CLAUSE
        Either   party   may   termin ate   with   3 0   days   notice.
        """

class TestMonetaryExtraction(ContractAnalyzerTests):
    """Test the enhanced monetary value extraction."""
    
    def test_monetary_extraction_coverage(self):
        """Test that we can extract monetary values from various formats."""
        print("\n🔍 Testing Monetary Value Extraction...")
        
        results = {}
        for text in self.test_monetary_texts:
            extracted = extract_monetary_values(text)
            results[text] = extracted
            print(f"  '{text}' -> {extracted}")
        
        # Verify we extracted something from each test case
        failed_extractions = [text for text, values in results.items() if not values]
        
        if failed_extractions:
            print(f"\n❌ Failed to extract values from: {len(failed_extractions)} cases")
            for text in failed_extractions:
                print(f"  - {text}")
        else:
            print(f"\n✅ Successfully extracted values from all {len(self.test_monetary_texts)} test cases")
        
        # Test specific patterns
        self.assertGreater(len(extract_monetary_values("Late fee of $500")), 0)
        self.assertGreater(len(extract_monetary_values("5% interest rate")), 0)
        self.assertGreater(len(extract_monetary_values("Range $1,000-$5,000")), 0)
        
        return len(failed_extractions) == 0

class TestClauseFiltering(ContractAnalyzerTests):
    """Test the smart clause filtering system."""
    
    def test_clause_filtering_effectiveness(self):
        """Test that filtering reduces noise while keeping important clauses."""
        print("\n🎯 Testing Clause Filtering Effectiveness...")
        
        original_count = len(self.test_contract_sentences)
        filtered_sentences = filter_potentially_risky_sentences(self.test_contract_sentences)
        filtered_count = len(filtered_sentences)
        
        print(f"  Original sentences: {original_count}")
        print(f"  Filtered sentences: {filtered_count}")
        print(f"  Reduction: {((original_count - filtered_count) / original_count * 100):.1f}%")
        
        # Check that high-priority clauses are kept
        high_priority_kept = []
        for sentence in filtered_sentences:
            if any(term in sentence.lower() for term in ['penalty', 'termination', 'indemnify', 'liquidated']):
                high_priority_kept.append(sentence)
        
        print(f"  High-priority clauses kept: {len(high_priority_kept)}")
        for clause in high_priority_kept:
            print(f"    - {clause}")
        
        # Check that boilerplate is filtered out
        boilerplate_removed = []
        boilerplate_terms = ['headings', 'counterparts', 'definitions']
        for sentence in self.test_contract_sentences:
            if any(term in sentence.lower() for term in boilerplate_terms):
                if sentence not in filtered_sentences:
                    boilerplate_removed.append(sentence)
        
        print(f"  Boilerplate clauses removed: {len(boilerplate_removed)}")
        for clause in boilerplate_removed:
            print(f"    - {clause}")
        
        # Verify filtering is working
        self.assertLess(filtered_count, original_count, "Filtering should reduce sentence count")
        self.assertGreater(len(high_priority_kept), 0, "Should keep high-priority clauses")
        
        return filtered_count < original_count and len(high_priority_kept) > 0

class TestOCRImprovements(ContractAnalyzerTests):
    """Test the enhanced OCR text processing."""
    
    def test_ocr_text_cleaning(self):
        """Test OCR text cleaning and post-processing."""
        print("\n📄 Testing OCR Text Cleaning...")
        
        # Test individual text cleaning
        test_cases = [
            ("RENT AL AGREE MENT", "RENTAL AGREEMENT"),
            ("$ 2 , 5 0 0 . 0 0", "$2,500.00"),
            ("5 %", "5%"),
            ("termin ation", "termination"),
            ("liab ility", "liability")
        ]
        
        print("  Testing post-processing fixes:")
        for dirty, expected in test_cases:
            cleaned = post_process_ocr_text(dirty)
            success = expected.lower() in cleaned.lower()
            status = "✅" if success else "❌"
            print(f"    {status} '{dirty}' -> '{cleaned}' (expected '{expected}')")
        
        # Test full text processing
        processed = post_process_ocr_text(self.test_ocr_text)
        print(f"\n  Original OCR text length: {len(self.test_ocr_text)}")
        print(f"  Processed text length: {len(processed)}")
        print(f"  Sample processed text: {processed[:100]}...")
        
        # Check for improvements
        improvements = [
            "$2,500.00" in processed,  # Fixed monetary amount
            "TERMINATION" in processed.upper(),  # Fixed split word
            "30 days" in processed  # Fixed number spacing
        ]
        
        success_count = sum(improvements)
        print(f"  Improvements detected: {success_count}/3")
        
        return success_count >= 2

def benchmark_performance():
    """Benchmark the performance of enhanced functions."""
    print("\n⚡ Performance Benchmarking...")
    
    # Create larger test dataset
    large_text_dataset = []
    base_sentences = [
        "The monthly payment of $1,500 is due on the first of each month.",
        "Late fees of 5% will be applied after 10 days.",
        "Termination requires 30 days written notice.",
        "Intellectual property rights remain with the original owner.",
        "Confidentiality obligations continue for 2 years post-termination."
    ]
    
    # Replicate to create larger dataset
    for i in range(100):
        large_text_dataset.extend(base_sentences)
    
    print(f"  Created dataset with {len(large_text_dataset)} sentences")
    
    # Benchmark filtering
    start_time = time.time()
    filtered = filter_potentially_risky_sentences(large_text_dataset)
    filtering_time = time.time() - start_time
    
    print(f"  Filtering {len(large_text_dataset)} sentences took: {filtering_time:.3f} seconds")
    print(f"  Filtered to {len(filtered)} sentences ({len(filtered)/len(large_text_dataset)*100:.1f}%)")
    
    # Benchmark monetary extraction
    test_text = " ".join(large_text_dataset)
    start_time = time.time()
    monetary_values = extract_monetary_values(test_text)
    extraction_time = time.time() - start_time
    
    print(f"  Monetary extraction from {len(test_text)} chars took: {extraction_time:.3f} seconds")
    print(f"  Found {len(monetary_values)} monetary values")
    
    return {
        'filtering_time': filtering_time,
        'extraction_time': extraction_time,
        'filtered_percentage': len(filtered)/len(large_text_dataset)*100
    }

def run_integration_test():
    """Test the complete analysis pipeline with sample data."""
    print("\n🔗 Integration Testing...")
    
    sample_contract_summary = """
    This rental agreement establishes a monthly rent of $2,500 due on the 1st of each month.
    Late payments incur a 5% monthly penalty fee. Security deposit of $5,000 is required.
    Either party may terminate with 30 days written notice. Tenant is responsible for utilities.
    Property damage exceeding normal wear may result in charges up to $10,000.
    Subletting is prohibited without written consent. Pet deposit of $500 applies per pet.
    """
    
    print("  Testing complete contract analysis pipeline...")
    try:
        # Test the analyze_contract function with our sample
        start_time = time.time()
        clauses = analyze_contract(sample_contract_summary)
        analysis_time = time.time() - start_time
        
        print(f"  Analysis completed in {analysis_time:.2f} seconds")
        print(f"  Found {len(clauses)} risk clauses")
        
        # Analyze the results
        high_risk = [c for c in clauses if c.get('risk') == 'High']
        medium_risk = [c for c in clauses if c.get('risk') == 'Medium']
        low_risk = [c for c in clauses if c.get('risk') == 'Low']
        
        print(f"    High risk: {len(high_risk)}")
        print(f"    Medium risk: {len(medium_risk)}")
        print(f"    Low risk: {len(low_risk)}")
        
        # Check if monetary values were extracted
        clauses_with_amounts = [c for c in clauses if c.get('amount') and c['amount'] != 'Not specified']
        print(f"    Clauses with specific amounts: {len(clauses_with_amounts)}")
        
        for clause in clauses_with_amounts[:3]:  # Show first 3
            print(f"      - {clause.get('type', 'Unknown')}: {clause.get('amount')}")
        
        return len(clauses) > 0 and len(clauses_with_amounts) > 0
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Main testing function."""
    print("🧪 Contract Analyzer Improvement Validation")
    print("=" * 60)
    
    # Track test results
    test_results = {}
    
    # Run individual component tests
    print("\n📋 Running Component Tests...")
    
    # Monetary extraction test
    test_monetary = TestMonetaryExtraction()
    test_monetary.setUp()
    try:
        test_results['monetary_extraction'] = test_monetary.test_monetary_extraction_coverage()
    except Exception as e:
        print(f"❌ Monetary extraction test failed: {e}")
        test_results['monetary_extraction'] = False
    
    # Clause filtering test
    test_filtering = TestClauseFiltering()
    test_filtering.setUp()
    try:
        test_results['clause_filtering'] = test_filtering.test_clause_filtering_effectiveness()
    except Exception as e:
        print(f"❌ Clause filtering test failed: {e}")
        test_results['clause_filtering'] = False
    
    # OCR improvements test
    test_ocr = TestOCRImprovements()
    test_ocr.setUp()
    try:
        test_results['ocr_improvements'] = test_ocr.test_ocr_text_cleaning()
    except Exception as e:
        print(f"❌ OCR improvements test failed: {e}")
        test_results['ocr_improvements'] = False
    
    # Performance benchmark
    try:
        benchmark_results = benchmark_performance()
        test_results['performance'] = benchmark_results
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        test_results['performance'] = None
    
    # Integration test
    try:
        test_results['integration'] = run_integration_test()
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        test_results['integration'] = False
    
    # Summary
    print("\n📊 Test Results Summary")
    print("-" * 40)
    
    passed_tests = 0
    total_tests = 0
    
    for test_name, result in test_results.items():
        if test_name == 'performance':
            continue  # Skip performance in pass/fail count
        total_tests += 1
        if result:
            passed_tests += 1
            print(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
        else:
            print(f"❌ {test_name.replace('_', ' ').title()}: FAILED")
    
    # Performance summary
    if test_results.get('performance'):
        perf = test_results['performance']
        print(f"⚡ Performance: {perf['filtered_percentage']:.1f}% reduction in clauses")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All improvements validated successfully!")
        print("The enhanced contract analyzer is working as expected.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
        print("Some improvements may need debugging.")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if test_results.get('monetary_extraction'):
        print("✓ Monetary extraction improvements are working")
    else:
        print("❗ Review monetary extraction patterns")
    
    if test_results.get('clause_filtering'):
        print("✓ Smart clause filtering is reducing noise")
    else:
        print("❗ Clause filtering may need adjustment")
    
    if test_results.get('ocr_improvements'):
        print("✓ OCR text cleaning is working")
    else:
        print("❗ OCR improvements may need refinement")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 