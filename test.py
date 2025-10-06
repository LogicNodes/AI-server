#!/usr/bin/env python3
"""
Production server test client
Tests real LLM inference with comprehensive scenarios
"""

import requests
import json
import time
import threading
from datetime import datetime

class ProductionTester:
    def __init__(self, base_url="http://localhost:8080", api_key="hardq_dev_key_001"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def test_server_status(self):
        """Test server health and status"""
        print("=== Server Status Test ===")
        
        try:
            # Test root endpoint
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                status = response.json()
                print("OK Server running:")
                print(f"  Model loaded: {status['model']['loaded']}")
                print(f"  GPU memory: {status['gpu']['used_gb']}GB / {status['gpu']['total_gb']}GB")
                print(f"  Active requests: {status['requests']['active']}")
                print(f"  Total processed: {status['requests']['total_processed']}")
                print(f"  Uptime: {status['uptime_seconds']} seconds")
                return True
            else:
                print(f"ERROR Server status failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR Could not connect: {e}")
            return False
    
    def test_health_check(self):
        """Test health endpoint"""
        print("\n=== Health Check Test ===")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health = response.json()
                print(f"Health: {health['status']}")
                print(f"Model loaded: {health['model_loaded']}")
                print(f"Message: {health['message']}")
                return health['status'] == "healthy"
            else:
                print(f"ERROR Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR Health check error: {e}")
            return False
    
    def test_authentication(self):
        """Test API key authentication"""
        print("\n=== Authentication Test ===")
        
        # Test without API key
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code == 401:
                print("OK Authentication required (no API key)")
            else:
                print(f"ERROR Expected 401, got {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR Auth test failed: {e}")
            return False
        
        # Test with wrong API key
        try:
            wrong_headers = {"Authorization": "Bearer wrong_key", "Content-Type": "application/json"}
            response = requests.get(f"{self.base_url}/v1/models", headers=wrong_headers)
            if response.status_code == 401:
                print("OK Invalid API key rejected")
            else:
                print(f"ERROR Expected 401, got {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR Wrong key test failed: {e}")
            return False
        
        # Test with correct API key
        try:
            response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
            if response.status_code == 200:
                print("OK Valid API key accepted")
                return True
            else:
                print(f"ERROR Valid key rejected: {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR Valid key test failed: {e}")
            return False
    
    def test_models_endpoint(self):
        """Test models listing"""
        print("\n=== Models Endpoint Test ===")
        
        try:
            response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
            if response.status_code == 200:
                models = response.json()
                print("Available models:")
                for model in models['data']:
                    print(f"  - {model['id']} (created: {model['created']})")
                return len(models['data']) > 0
            else:
                print(f"ERROR Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR Models test failed: {e}")
            return False
    
    def test_simple_chat(self):
        """Test basic chat completion"""
        print("\n=== Simple Chat Test ===")
        
        # First get the model name
        try:
            models_response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
            if models_response.status_code != 200:
                print("ERROR Could not get model list")
                return False
            
            models = models_response.json()
            if not models['data']:
                print("ERROR No models available")
                return False
            
            model_name = models['data'][0]['id']
            print(f"Testing with model: {model_name}")
            
        except Exception as e:
            print(f"ERROR Getting model name: {e}")
            return False
        
        # Test simple completion
        chat_request = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Hello! Please tell me a short joke."}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        try:
            print("Sending chat request...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=chat_request,
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                usage = result['usage']
                
                print(f"OK Response received ({response_time:.1f}s):")
                print(f"  Content: {content[:100]}{'...' if len(content) > 100 else ''}")
                print(f"  Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total")
                
                return True
            else:
                print(f"ERROR Chat failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"ERROR Chat test failed: {e}")
            return False
    
    def test_concurrent_requests(self, num_requests=3):
        """Test concurrent request handling"""
        print(f"\n=== Concurrent Requests Test ({num_requests} requests) ===")
        
        # Get model name
        try:
            models_response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
            models = models_response.json()
            model_name = models['data'][0]['id']
        except Exception as e:
            print(f"ERROR Getting model: {e}")
            return False
        
        results = []
        threads = []
        
        def make_request(request_id):
            chat_request = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": f"Request {request_id}: What is {request_id} + {request_id}?"}
                ],
                "max_tokens": 50,
                "temperature": 0.3
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=chat_request,
                    timeout=30
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        'id': request_id,
                        'status': 'success',
                        'time': end_time - start_time,
                        'content': result['choices'][0]['message']['content'],
                        'tokens': result['usage']['total_tokens']
                    })
                else:
                    results.append({
                        'id': request_id,
                        'status': 'failed',
                        'time': end_time - start_time,
                        'error': response.text
                    })
            except Exception as e:
                results.append({
                    'id': request_id,
                    'status': 'error',
                    'time': 0,
                    'error': str(e)
                })
        
        # Start all requests
        print("Starting concurrent requests...")
        for i in range(num_requests):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Analyze results
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] != 'success']
        
        print(f"Results: {len(successful)}/{num_requests} successful")
        
        for result in results:
            status_symbol = "OK" if result['status'] == 'success' else "ERROR"
            print(f"  Request {result['id']}: {status_symbol} ({result['time']:.1f}s)")
            if result['status'] == 'success':
                print(f"    Content: {result['content'][:50]}...")
                print(f"    Tokens: {result['tokens']}")
            else:
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        return len(successful) > 0
    
    def test_parameter_validation(self):
        """Test request parameter validation"""
        print("\n=== Parameter Validation Test ===")
        
        # Get model name
        try:
            models_response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
            models = models_response.json()
            model_name = models['data'][0]['id']
        except Exception:
            print("ERROR Could not get model name")
            return False
        
        tests = [
            {
                "name": "Empty messages",
                "request": {"model": model_name, "messages": []},
                "expect_fail": True
            },
            {
                "name": "Invalid model",
                "request": {"model": "nonexistent-model", "messages": [{"role": "user", "content": "test"}]},
                "expect_fail": True
            },
            {
                "name": "Excessive max_tokens",
                "request": {"model": model_name, "messages": [{"role": "user", "content": "test"}], "max_tokens": 10000},
                "expect_fail": True
            },
            {
                "name": "Valid request",
                "request": {"model": model_name, "messages": [{"role": "user", "content": "test"}], "max_tokens": 10},
                "expect_fail": False
            }
        ]
        
        all_passed = True
        
        for test in tests:
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=test["request"],
                    timeout=20
                )
                
                failed = response.status_code >= 400
                expected_result = test["expect_fail"]
                
                if failed == expected_result:
                    print(f"  OK {test['name']}")
                else:
                    print(f"  ERROR {test['name']}: expected {'failure' if expected_result else 'success'}, got {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"  ERROR {test['name']}: {e}")
                all_passed = False
        
        return all_passed
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("=" * 60)
        print("PRODUCTION LLM SERVER - COMPREHENSIVE TEST")
        print("=" * 60)
        
        tests = [
            ("Server Status", self.test_server_status),
            ("Health Check", self.test_health_check),
            ("Authentication", self.test_authentication),
            ("Models Endpoint", self.test_models_endpoint),
            ("Simple Chat", self.test_simple_chat),
            ("Concurrent Requests", self.test_concurrent_requests),
            ("Parameter Validation", self.test_parameter_validation)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print()
            try:
                result = test_func()
                if result:
                    passed += 1
                    print(f"✓ {test_name}: PASSED")
                else:
                    print(f"✗ {test_name}: FAILED")
            except Exception as e:
                print(f"✗ {test_name}: ERROR - {e}")
        
        print()
        print("=" * 60)
        print(f"TEST RESULTS: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - Server is ready for production!")
        elif passed >= total - 1:
            print("✅ Most tests passed - Server is mostly functional")
        else:
            print("⚠️ Some tests failed - Check server configuration")
        
        print("=" * 60)
        
        return passed == total

def main():
    tester = ProductionTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()