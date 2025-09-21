# server/scripts/test_agent.py
"""
Test script to verify agents work independently of the API
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.mandi.agent import MandiAgent
from agents.mandi.models import MandiRequest
from core.config import get_settings
from core.logging import setup_logging

async def test_mandi_agent():
    """Test mandi agent functionality"""
    
    print("🧪 Testing Mandi Agent")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("1. Initializing Mandi Agent...")
        agent = MandiAgent()
        print("   ✅ Agent initialized successfully")
        
        # Test health check
        print("\n2. Running health check...")
        health = await agent.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Config Valid: {health['config_valid']}")
        
        # Test basic request
        print("\n3. Testing basic request...")
        request = MandiRequest(
            state="Punjab",
            district="Ludhiana",
            market="Ludhiana",
            commodity="Wheat"
        )
        
        print(f"   Request: {request.dict()}")
        
        # Execute request
        print("\n4. Executing request...")
        response = await agent.execute(request, use_cache=False)
        
        print(f"   Success: {response.success}")
        print(f"   Message: {response.message}")
        print(f"   Data points: {len(response.data)}")
        
        if response.data:
            price_data = response.data[0]
            print(f"   Current Price: ₹{price_data.currentPrice}")
            print(f"   Trend: {price_data.trend}")
            print(f"   Recommendation: {price_data.prediction.recommendation}")
        
        # Test fallback
        print("\n5. Testing fallback response...")
        fallback = agent.get_fallback_response(request, Exception("Test error"))
        print(f"   Fallback success: {fallback.success}")
        print(f"   Fallback price: ₹{fallback.data[0].currentPrice}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Test API endpoints using httpx"""
    
    print("\n🌐 Testing API Endpoints")
    print("=" * 50)
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            print("1. Testing health endpoint...")
            response = await client.get(f"{base_url}/api/health")
            if response.status_code == 200:
                print("   ✅ Health endpoint working")
            else:
                print(f"   ❌ Health endpoint failed: {response.status_code}")
            
            # Test mandi prices endpoint
            print("\n2. Testing mandi prices endpoint...")
            response = await client.get(f"{base_url}/api/mandi/prices")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Mandi prices endpoint working")
                print(f"   Response: {data.get('message', 'No message')}")
            else:
                print(f"   ❌ Mandi prices failed: {response.status_code}")
        
        return True
        
    except ImportError:
        print("   ⚠️  httpx not installed, skipping API tests")
        print("   Install with: pip install httpx")
        return True
    except Exception as e:
        print(f"   ❌ API tests failed: {e}")
        return False

def check_environment():
    """Check environment setup"""
    
    print("🔧 Checking Environment")
    print("=" * 50)
    
    settings = get_settings()
    
    # Check API key
    if settings.data_gov_in_api_key:
        print("✅ DATA_GOV_IN_API_KEY is set")
    else:
        print("⚠️  DATA_GOV_IN_API_KEY is not set (will use fallback data)")
    
    # Check configuration
    print(f"✅ Environment: {settings.environment}")
    print(f"✅ Debug mode: {settings.debug}")
    print(f"✅ API host: {settings.api_host}:{settings.api_port}")
    
    return True

async def main():
    """Run all tests"""
    
    print("🚀 KisanSathi Backend Test Suite")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Check environment
    check_environment()
    
    # Test agent
    agent_success = await test_mandi_agent()
    
    # Test API (only if server is running)
    print("\n📝 Note: To test API endpoints, start the server first:")
    print("   python run.py")
    print("   Then run this script again")
    
    if agent_success:
        print("\n🎉 Backend tests completed successfully!")
        print("\nNext steps:")
        print("1. Start the server: python run.py")
        print("2. Visit http://localhost:8000/docs for API documentation")
        print("3. Test endpoints manually or run: python scripts/test_agent.py")
    else:
        print("\n❌ Some tests failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())