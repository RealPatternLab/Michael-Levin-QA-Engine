#!/usr/bin/env python3
"""
Test runner for CI/CD pipeline.
Runs all tests to verify the system is working correctly.
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_script):
    """Run a test script and return success status."""
    print(f"\n🧪 Running: {test_script}")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, test_script
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {test_script} passed")
            return True
        else:
            print(f"❌ {test_script} failed (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Error running {test_script}: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running CI/CD Test Suite")
    print("=" * 50)
    
    # Define test scripts in order of dependency
    test_scripts = [
        "scripts/test/test_api_key.py",
        "scripts/test/test_llm_extraction.py"
    ]
    
    # Run tests
    results = []
    for test_script in test_scripts:
        success = run_test(test_script)
        results.append((test_script, success))
    
    # Summary
    print("\n📊 Test Results")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_script, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_script}")
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("🔧 Some tests failed. Please fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 