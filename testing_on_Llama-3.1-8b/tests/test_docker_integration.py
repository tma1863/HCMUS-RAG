#!/usr/bin/env python3
"""
Test Docker Integration for HippoRAG System
Tests the complete pipeline: Ollama + HippoRAG + Qdrant
"""

import json
import requests
import time

def test_ollama_connection():
    """Test Ollama API connection"""
    print("Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()['models']
            print(f"Ollama connected. Available models: {len(models)}")
            for model in models:
                print(f"   - {model['name']} ({model['details']['parameter_size']}, {model['size']//1024//1024//1024:.1f}GB)")
            return True
        else:
            print(f"Ollama connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Ollama connection error: {e}")
        return False

def test_hipporag_health():
    """Test HippoRAG health endpoint"""
    print("\nTesting HippoRAG health...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"HippoRAG health: {health_data['status']}")
            print(f"   Services: {health_data['services']}")
            return True
        else:
            print(f"HippoRAG health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"HippoRAG health error: {e}")
        return False

def test_simple_generation():
    """Test simple text generation through Ollama"""
    print("\nTesting simple text generation...")
    try:
        payload = {
            "model": "llama3:8b",
            "prompt": "What is machine learning in simple terms?",
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Generation successful")
            print(f"   Response: {result['response'][:100]}...")
            return True
        else:
            print(f"Generation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Generation error: {e}")
        return False

def test_qdrant_connection():
    """Test Qdrant vector database connection"""
    print("\nTesting Qdrant connection...")
    try:
        response = requests.get("http://localhost:6333/")
        if response.status_code == 200:
            print("Qdrant connected")
            return True
        else:
            print(f"Qdrant connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Qdrant connection error: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Starting Docker Integration Tests for HippoRAG System")
    print("=" * 60)
    
    tests = [
        ("Ollama API", test_ollama_connection),
        ("HippoRAG Health", test_hipporag_health),
        ("Qdrant Vector DB", test_qdrant_connection),
        ("Text Generation", test_simple_generation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All systems operational! Docker HippoRAG is ready for use.")
    else:
        print("Some tests failed. Please check the services.")
    
    return passed == total

if __name__ == "__main__":
    main() 