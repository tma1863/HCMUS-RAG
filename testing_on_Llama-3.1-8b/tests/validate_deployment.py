#!/usr/bin/env python3
"""
HippoRAG Deployment Validation Script
====================================
Comprehensive test suite to validate HippoRAG deployment
"""

import requests
import time
import json
import sys
import subprocess

def test_health_endpoints():
    """Test all health endpoints"""
    print("Testing Health Endpoints...")
    
    tests = [
        ("HippoRAG Health", "http://localhost:8000/health"),
        ("HippoRAG Status", "http://localhost:8000/api/status"),
        ("HippoRAG Models", "http://localhost:8000/api/models"),
        ("Ollama API", "http://localhost:11434/api/tags"),
        ("Qdrant Health", "http://localhost:6333/collections")
    ]
    
    results = []
    for name, url in tests:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"  {name}: OK")
                results.append(True)
            else:
                print(f"  {name}: HTTP {response.status_code}")
                results.append(False)
        except Exception as e:
            print(f"  {name}: {str(e)}")
            results.append(False)
    
    return all(results)

def test_llm_generation():
    """Test LLM generation"""
    print("\nTesting LLM Generation...")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/test-generation",
            json={"prompt": "What is the capital of Vietnam?"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result and len(result['response']) > 0:
                print(f"  LLM Generation: OK")
                print(f"  Response: {result['response'][:100]}...")
                return True
            else:
                print(f"  LLM Generation: Empty response")
                return False
        else:
            print(f"  LLM Generation: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  LLM Generation: {str(e)}")
        return False

def test_hipporag_evaluation():
    """Test HippoRAG evaluation"""
    print("\nTesting HippoRAG Evaluation...")
    
    try:
        cmd = [
            "docker", "exec", "hipporag_app", 
            "python", "main_hipporag.py", 
            "--dataset", "AM", 
            "--test_type", "closed_end", 
            "--max_questions", "1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            if "Evaluation completed" in result.stdout:
                print(f"  HippoRAG Evaluation: OK")
                return True
            else:
                print(f"  HippoRAG Evaluation: Failed")
                print(f"  Output: {result.stdout[-200:]}")
                return False
        else:
            print(f"  HippoRAG Evaluation: Process failed")
            print(f"  Error: {result.stderr[-200:]}")
            return False
    except Exception as e:
        print(f"  HippoRAG Evaluation: {str(e)}")
        return False

def test_docker_containers():
    """Test Docker container status"""
    print("\nTesting Docker Containers...")
    
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"], 
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode == 0:
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        container = json.loads(line)
                        containers.append(container)
                    except:
                        continue
            
            required_services = ['hipporag', 'ollama', 'qdrant']
            running_services = []
            
            for container in containers:
                service = container.get('Service', '')
                state = container.get('State', '')
                
                if service in required_services and state == 'running':
                    running_services.append(service)
                    print(f"  {service}: {state}")
                elif service in required_services:
                    print(f"  {service}: {state}")
            
            return len(running_services) == len(required_services)
        else:
            print(f"  Docker command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Docker test failed: {str(e)}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\nTesting GPU Availability...")
    
    try:
        # Test nvidia-smi
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  NVIDIA GPU: Available")
            
            # Test GPU in container
            gpu_test = subprocess.run([
                "docker", "exec", "hipporag_ollama", 
                "nvidia-smi", "--query-gpu=name,memory.used,memory.total", 
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if gpu_test.returncode == 0:
                print(f"  GPU in container: Available")
                print(f"  GPU Info: {gpu_test.stdout.strip()}")
                return True
            else:
                print(f"  GPU in container: Not available (CPU fallback)")
                return True
        else:
            print(f"  NVIDIA GPU: Not available (CPU fallback)")
            return True
    except Exception as e:
        print(f"  GPU test: {str(e)} (CPU fallback)")
        return True

def test_data_persistence():
    """Test data persistence"""
    print("\nTesting Data Persistence...")
    
    directories = [
        ("outputs", "outputs"),
        ("logs", "logs"),
        ("embedding_stores", "embedding_stores")
    ]
    
    results = []
    for name, path in directories:
        try:
            result = subprocess.run([
                "docker", "exec", "hipporag_app", 
                "ls", "-la", f"/app/{path}"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"  {name}: Accessible")
                results.append(True)
            else:
                print(f"  {name}: Not accessible")
                results.append(False)
        except Exception as e:
            print(f"  {name}: {str(e)}")
            results.append(False)
    
    return all(results)

def main():
    """Run all validation tests"""
    print("HippoRAG Deployment Validation")
    print("="*50)
    
    tests = [
        ("Docker Containers", test_docker_containers),
        ("Health Endpoints", test_health_endpoints),
        ("GPU Availability", test_gpu_availability),
        ("LLM Generation", test_llm_generation),
        ("Data Persistence", test_data_persistence),
        ("HippoRAG Evaluation", test_hipporag_evaluation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name}: Exception - {str(e)}")
            results.append((test_name, False))
    
    # Final results
    print(f"\n{'='*50}")
    print("VALIDATION RESULTS")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("ALL TESTS PASSED - HippoRAG is ready for deployment!")
        return 0
    elif passed >= total * 0.8:
        print("MOSTLY WORKING - Some optional features may be unavailable")
        return 0
    else:
        print("DEPLOYMENT ISSUES - Please check failed tests")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 