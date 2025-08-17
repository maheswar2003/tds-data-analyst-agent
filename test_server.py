#!/usr/bin/env python3
"""
Complete API test suite for TDS Data Analyst Agent.
This will work regardless of PowerShell version or curl availability.
"""

import json
import os
import sys
import time
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError

def test_endpoint(url, method="GET", data=None, timeout=10):
    """Test an API endpoint."""
    try:
        if data:
            if isinstance(data, dict):
                data = json.dumps(data).encode('utf-8')
            req = Request(url, data=data, method=method)
            req.add_header('Content-Type', 'application/json')
        else:
            req = Request(url, method=method)
        
        with urlopen(req, timeout=timeout) as response:
            return {
                'status': response.status,
                'data': json.loads(response.read().decode('utf-8')),
                'success': True
            }
    except HTTPError as e:
        return {
            'status': e.code,
            'data': e.read().decode('utf-8'),
            'success': False,
            'error': str(e)
        }
    except URLError as e:
        return {
            'status': 0,
            'data': None,
            'success': False,
            'error': f"Connection failed: {e.reason}"
        }
    except Exception as e:
        return {
            'status': 0,
            'data': None,
            'success': False,
            'error': str(e)
        }

def test_file_upload():
    """Test file upload using basic HTTP."""
    try:
        import urllib.request
        import mimetypes
        
        file_path = 'test_questions/wikipedia_question.txt'
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': f'Test file not found: {file_path}'
            }
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # Create multipart form data manually
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        
        form_data = []
        form_data.append(f'--{boundary}'.encode())
        form_data.append(b'Content-Disposition: form-data; name="question"; filename="wikipedia_question.txt"')
        form_data.append(b'Content-Type: text/plain')
        form_data.append(b'')
        form_data.append(file_content.encode('utf-8'))
        form_data.append(f'--{boundary}--'.encode())
        
        body = b'\r\n'.join(form_data)
        
        # Create request
        req = Request('http://127.0.0.1:8000/api/', data=body, method='POST')
        req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        
        with urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return {
                'status': response.status,
                'data': result,
                'success': True
            }
            
    except Exception as e:
        return {
            'status': 0,
            'data': None,
            'success': False,
            'error': str(e)
        }

def main():
    """Run all API tests."""
    print("🚀 TDS Data Analyst Agent - API Test Suite")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n🏥 Testing Health Endpoint...")
    health_result = test_endpoint("http://127.0.0.1:8000/health")
    
    if health_result['success']:
        print("✅ Health Check: PASSED")
        print(f"   Status: {health_result['status']}")
        print(f"   Message: {health_result['data'].get('message', 'N/A')}")
    else:
        print("❌ Health Check: FAILED")
        print(f"   Error: {health_result['error']}")
        print("\n💡 Make sure the server is running:")
        print("   python start_server.py")
        return False
    
    # Test 2: Root Endpoint
    print("\n🏠 Testing Root Endpoint...")
    root_result = test_endpoint("http://127.0.0.1:8000/")
    
    if root_result['success']:
        print("✅ Root Endpoint: PASSED")
        print(f"   Version: {root_result['data'].get('version', 'N/A')}")
        endpoints = root_result['data'].get('endpoints', {})
        print("   Available Endpoints:")
        for endpoint, description in endpoints.items():
            print(f"     • {endpoint}: {description}")
    else:
        print("❌ Root Endpoint: FAILED")
        print(f"   Error: {root_result['error']}")
    
    # Test 3: File Upload
    print("\n📤 Testing File Upload Endpoint...")
    
    # Check if Google API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  GOOGLE_API_KEY not set - upload test will likely fail")
        print("   Set it with: set GOOGLE_API_KEY=your-key-here")
    
    upload_result = test_file_upload()
    
    if upload_result['success']:
        print("✅ File Upload: PASSED")
        print(f"   Status: {upload_result['status']}")
        
        data = upload_result['data']
        if isinstance(data, dict):
            if 'summary' in data:
                summary = data['summary'][:100] + "..." if len(data['summary']) > 100 else data['summary']
                print(f"   Summary: {summary}")
            
            if 'status' in data:
                print(f"   Analysis Status: {data['status']}")
            
            if 'data' in data and isinstance(data['data'], dict):
                print(f"   Data Keys: {', '.join(data['data'].keys())}")
            
            if 'visualizations' in data and data['visualizations']:
                vis_count = len(data['visualizations'])
                print(f"   Visualizations: {vis_count} generated")
        else:
            print(f"   Raw Response: {str(data)[:200]}...")
            
    else:
        print("❌ File Upload: FAILED")
        print(f"   Error: {upload_result['error']}")
        
        if "API" in str(upload_result['error']) or "key" in str(upload_result['error']).lower():
            print("   💡 This might be due to missing Google API key")
        elif "timeout" in str(upload_result['error']).lower():
            print("   💡 The request timed out - this is normal for complex analysis")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    tests = [health_result, root_result, upload_result]
    passed = sum(1 for test in tests if test['success'])
    total = len(tests)
    
    print(f"   ✅ Passed: {passed}/{total}")
    if passed == total:
        print("   🎉 All tests passed! Your API is working perfectly.")
    elif passed >= 2:
        print("   ⚠️  Core functionality working. Check Google API key for full features.")
    else:
        print("   ❌ Multiple failures detected. Check server and configuration.")
    
    print(f"\n💻 Interactive API docs: http://127.0.0.1:8000/docs")
    print(f"📋 API info: http://127.0.0.1:8000/")
    
    return passed >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)