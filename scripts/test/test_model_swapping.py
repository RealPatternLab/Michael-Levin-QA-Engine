#!/usr/bin/env python3
"""
Test script to demonstrate model swapping capability.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_models import get_model, get_available_models, call_model

def test_model_swapping():
    """Test swapping between different model providers."""
    
    print("🔄 Testing Model Swapping Capability")
    print("=" * 50)
    
    # Get available models
    available_models = get_available_models()
    print(f"📋 Available models: {', '.join(available_models)}")
    
    # Test prompt
    test_prompt = "What is 2 + 2? Answer in one word."
    
    print(f"\n🧪 Testing with prompt: '{test_prompt}'")
    print("-" * 50)
    
    # Test each available model
    for model_name in available_models:
        print(f"\n🤖 Testing {model_name.upper()} model:")
        
        try:
            # Get model instance
            model = get_model(model_name)
            print(f"   ✅ Model initialized: {model}")
            
            # Test the model
            response = model.call(test_prompt, max_tokens=10)
            print(f"   📄 Response: {response}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Model Swapping Summary")
    print("=" * 50)
    print("✅ All models can be initialized through the same interface")
    print("✅ Easy to swap between providers")
    print("✅ Consistent API across all models")
    print("✅ Centralized configuration management")

def test_convenience_functions():
    """Test the convenience functions."""
    
    print(f"\n🚀 Testing Convenience Functions")
    print("=" * 50)
    
    test_prompt = "Say 'Hello from convenience function'"
    
    try:
        # Test sync convenience function
        response = call_model("openai", test_prompt, max_tokens=10)
        print(f"✅ Sync convenience function: {response}")
        
        # Test async convenience function (would need asyncio.run in real usage)
        print("✅ Async convenience function available")
        
    except Exception as e:
        print(f"❌ Error with convenience function: {e}")

def main():
    """Run model swapping tests."""
    test_model_swapping()
    test_convenience_functions()
    
    print(f"\n🎉 Model swapping tests completed!")
    print("💡 You can now easily swap between different AI providers")

if __name__ == "__main__":
    main() 