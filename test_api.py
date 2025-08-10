#!/usr/bin/env python3
"""
Simple test script for the TDS Data Analyst Agent API.
"""

import requests
import os
import json

def test_server_health():
    """Test if the server is running."""
    try:
        response = requests.get('http://127.0.0.1:8000/health')
        print(f"✅ Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint."""
    try:
        response = requests.get('http://127.0.0.1:8000/')
        print(f"✅ Root Endpoint: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_file_upload():
    """Test the file upload endpoint."""
    try:
        # Test with Wikipedia question
        wiki_file_path = 'test_questions/wikipedia_question.txt'
        if not os.path.exists(wiki_file_path):
            print(f"❌ File not found: {wiki_file_path}")
            return False
        
        with open(wiki_file_path, 'rb') as f:
            files = {'question': f}
            print(f"📤 Uploading {wiki_file_path}...")
            response = requests.post('http://127.0.0.1:8000/api/', files=files)
        
        print(f"✅ File Upload: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("📊 Analysis Results:")
            print(f"  - Summary: {result.get('summary', 'N/A')}")
            print(f"  - Status: {result.get('status', 'N/A')}")
            if 'data' in result:
                print(f"  - Data keys: {list(result['data'].keys()) if isinstance(result['data'], dict) else 'N/A'}")
            if 'visualizations' in result:
                vis_count = len(result['visualizations']) if result['visualizations'] else 0
                print(f"  - Visualizations: {vis_count}")
        else:
            print(f"❌ Error response: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ File upload test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing TDS Data Analyst Agent API")
    print("=" * 50)
    
    # Test server health
    if not test_server_health():
        print("❌ Server is not running. Start it with: python -m uvicorn main:app --reload")
        return
    
    print("\n" + "-" * 30)
    
    # Test root endpoint
    test_root_endpoint()
    
    print("\n" + "-" * 30)
    
    # Test file upload (this will require OpenAI API key)
    print("🧪 Testing File Upload (requires OPENAI_API_KEY)...")
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set - file upload test will fail")
    
    test_file_upload()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")

if __name__ == "__main__":
    main()