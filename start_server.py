#!/usr/bin/env python3
"""
Server startup script for TDS Data Analyst Agent.
"""

import uvicorn
import sys
import os

if __name__ == "__main__":
    print("🚀 Starting TDS Data Analyst Agent Server...")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("📖 API Documentation: http://127.0.0.1:8000/docs")
    print("❌ Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "main:app", 
            host="127.0.0.1", 
            port=8000, 
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n✅ Server stopped gracefully")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)