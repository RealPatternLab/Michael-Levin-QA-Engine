#!/usr/bin/env python3
"""
CI/CD-friendly script to test AI model API key availability.
"""

import os
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import OPENAI_API_KEY, CLAUDE_API_KEY, DEFAULT_MODEL
from ai_models import get_model, get_available_models

def test_api_key():
    """Test if AI model API keys are available and valid."""
    
    print("🔑 Testing AI Model API Keys")
    print("=" * 40)
    
    # Test OpenAI API key
    print("\n🤖 Testing OpenAI API Key")
    print("-" * 30)
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-api-key-here':
        print("❌ OpenAI API key not found or not configured")
        print()
        print("📋 To fix this:")
        print("1. Get your API key from: https://platform.openai.com/api-keys")
        print("2. Edit the .env file in the project root")
        print("3. Replace 'your-api-key-here' with your actual API key")
        print("4. Your API key should start with 'sk-'")
        print()
        print("💡 Example .env file:")
        print("OPENAI_API_KEY=sk-1234567890abcdef...")
        print()
        openai_valid = False
    else:
        if not OPENAI_API_KEY.startswith('sk-'):
            print("❌ API key format looks incorrect")
            print("   API key should start with 'sk-'")
            print(f"   Found: {OPENAI_API_KEY[:10]}...")
            openai_valid = False
        else:
            print(f"✅ API key found: {OPENAI_API_KEY[:10]}...")
            
            # Test the API key with the new interface
            try:
                model = get_model("openai")
                response = model.call("Say 'Hello'", max_tokens=5)
                print("✅ OpenAI API key is valid and working!")
                print(f"   Test response: {response}")
                openai_valid = True
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if "authentication" in error_msg or "invalid" in error_msg:
                    print("❌ API key is invalid or expired")
                    print("   Please check your API key at: https://platform.openai.com/api-keys")
                elif "rate limit" in error_msg:
                    print("❌ Rate limit exceeded")
                    print("   You may need to upgrade your OpenAI plan")
                else:
                    print(f"❌ API test failed: {e}")
                    print("   Please check your internet connection and try again")
                openai_valid = False
    
    # Test Claude API key (if configured)
    print("\n🤖 Testing Claude API Key")
    print("-" * 30)
    
    if not CLAUDE_API_KEY:
        print("⚠️  Claude API key not configured (optional)")
        print("   To use Claude, add CLAUDE_API_KEY to your .env file")
        claude_valid = False
    else:
        print(f"✅ Claude API key found: {CLAUDE_API_KEY[:10]}...")
        print("⚠️  Claude integration not yet implemented")
        claude_valid = False
    
    # Summary
    print("\n📊 API Key Test Results")
    print("=" * 40)
    print(f"OpenAI: {'✅ Valid' if openai_valid else '❌ Invalid/Not configured'}")
    print(f"Claude: {'✅ Valid' if claude_valid else '❌ Invalid/Not configured'}")
    
    available_models = get_available_models()
    print(f"\n🎯 Available models: {', '.join(available_models)}")
    print(f"🎯 Default model: {DEFAULT_MODEL}")
    
    return openai_valid or claude_valid

def main():
    """Main function to test API keys."""
    success = test_api_key()
    
    if success:
        print("\n🎉 At least one AI model is ready for use!")
        print("   You can now run: python scripts/test/test_llm_extraction.py")
    else:
        print("\n🔧 Please configure at least one AI model before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main() 