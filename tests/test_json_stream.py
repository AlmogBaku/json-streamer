"""
Comprehensive tests for json-streamer
Includes correctness tests, edge cases, and realistic LLM streaming benchmarks
"""
import json
import time
import unittest
from json_streamer import loads, ParseState


# ============================================================================
# Helper Functions
# ============================================================================

def simulate_network_chunks(data: str, avg_size: int = 50, variance: int = 20):
    """
    Simulate realistic network streaming with variable chunk sizes
    Mimics actual LLM API behavior (chunks aren't uniform)
    """
    import random
    pos = 0
    while pos < len(data):
        # Vary chunk size like real network would
        chunk_size = max(1, avg_size + random.randint(-variance, variance))
        chunk = data[pos:pos + chunk_size]
        yield chunk
        pos += chunk_size


def stream_fixed_chunks(data: str, chunk_size: int):
    """Stream data in fixed-size chunks"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


# ============================================================================
# Basic Correctness Tests
# ============================================================================

class TestBasicStreaming(unittest.TestCase):
    """Original test - backwards compatibility"""

    def test_basic_streaming(self):
        """Test basic streaming with small chunks"""
        json_string = '{"field1": "value1", "field2": 42}'

        def stream():
            for i in range(0, len(json_string), 4):
                yield json_string[i:i + 4]

        results = list(loads(stream()))

        expected_states = [
            (ParseState.PARTIAL, {'field1': ''}),
            (ParseState.PARTIAL, {'field1': 'valu'}),
            (ParseState.PARTIAL, {'field1': 'value1', 'field2': 4}),
            (ParseState.COMPLETE, {'field1': 'value1', 'field2': 42})
        ]

        for i, (state, parsed_dict) in enumerate(results):
            expected_state, expected_dict = expected_states[i]
            self.assertEqual(state, expected_state)
            self.assertEqual(parsed_dict, expected_dict)


# ============================================================================
# Edge Cases - String Escaping & Special Characters
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases that broke the original implementation"""

    def test_escaped_quotes_in_string(self):
        """Test handling of escaped quotes: {\"key\": \"value with \\\" quote\"}"""
        json_str = '{"key": "value with \\" quote"}'

        results = list(loads(stream_fixed_chunks(json_str, 15)))
        self.assertTrue(len(results) > 0)

        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertEqual(final_obj['key'], 'value with " quote')

    def test_escaped_backslash_before_quote(self):
        """Test \\\\\" sequence - escaped backslash, then regular quote"""
        json_str = '{"key": "value\\\\"}'

        results = list(loads(stream_fixed_chunks(json_str, 5)))
        self.assertTrue(len(results) > 0)

        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertEqual(final_obj['key'], 'value\\')

    def test_brackets_inside_string_value(self):
        """Test that brackets inside strings don't affect parsing"""
        json_str = '{"msg": "has } and ] chars", "id": 1}'

        results = list(loads(stream_fixed_chunks(json_str, 6)))
        self.assertTrue(len(results) > 0)

        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertEqual(final_obj['msg'], 'has } and ] chars')
        self.assertEqual(final_obj['id'], 1)

    def test_partial_string_completion(self):
        """Test that partial strings are closed correctly"""
        def stream():
            yield '{"key": "val'

        results = list(loads(stream()))
        self.assertTrue(len(results) > 0)

        state, obj = results[0]
        self.assertEqual(state, ParseState.PARTIAL)
        self.assertIn('key', obj)

    def test_number_with_decimal_point(self):
        """Test partial number like 3.14"""
        def stream():
            yield '{"value": 3.'
            yield '14}'

        results = list(loads(stream()))
        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertEqual(final_obj['value'], 3.14)

    def test_nested_objects_with_strings(self):
        """Test nested objects with special characters in strings"""
        json_str = '{"outer": {"inner": "has { and }"}, "done": true}'

        results = list(loads(stream_fixed_chunks(json_str, 8)))
        self.assertTrue(len(results) > 0)

        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertEqual(final_obj['outer']['inner'], 'has { and }')

    def test_array_with_partial_elements(self):
        """Test array streaming"""
        def stream():
            yield '[1, 2'
            yield ', 3]'

        results = list(loads(stream()))
        self.assertTrue(len(results) > 0)

        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertEqual(final_obj, [1, 2, 3])

    def test_unicode_escape_sequences(self):
        """Test \\uXXXX escape sequences"""
        json_str = '{"emoji": "\\u2764"}'  # Heart emoji

        results = list(loads(stream_fixed_chunks(json_str, 7)))
        self.assertTrue(len(results) > 0)

        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertIn('emoji', final_obj)


# ============================================================================
# Realistic LLM Streaming Scenarios
# ============================================================================

class TestLLMStreamingScenarios(unittest.TestCase):
    """Test patterns from real LLM streaming APIs"""

    def test_openai_chat_completion_pattern(self):
        """
        Simulate OpenAI streaming response pattern
        Typical: Small chunks with content deltas
        """
        # Typical OpenAI streaming response structure
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "delta": {
                    "content": "This is a streaming response from GPT-4 with a reasonably long message that would be sent in chunks over the network."
                },
                "index": 0,
                "finish_reason": None
            }]
        }

        json_str = json.dumps(response)

        # OpenAI typically sends ~50-100 byte chunks with network jitter
        results = list(loads(simulate_network_chunks(json_str, avg_size=60, variance=30)))

        self.assertTrue(len(results) > 0)
        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertEqual(final_obj['id'], 'chatcmpl-123')
        self.assertIn('content', final_obj['choices'][0]['delta'])

    def test_function_calling_response(self):
        """
        Test OpenAI function calling response (can be large)
        These have nested objects and arrays
        """
        response = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1699896916,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        json_str = json.dumps(response)
        results = list(loads(simulate_network_chunks(json_str, avg_size=80, variance=40)))

        self.assertTrue(len(results) > 0)
        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertIn('tool_calls', final_obj['choices'][0]['delta'])

    def test_very_small_chunks_rapid_fire(self):
        """
        Test extremely small chunks (1-5 bytes)
        Stress test for character-by-character streaming
        """
        response = {"status": "processing", "data": {"items": [1, 2, 3, 4, 5]}}
        json_str = json.dumps(response)

        # Extremely small chunks
        results = list(loads(stream_fixed_chunks(json_str, chunk_size=3)))

        self.assertTrue(len(results) > 0)
        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertEqual(final_obj['status'], 'processing')

    def test_large_content_field(self):
        """
        Test large content field (like LLM generating long text)
        This is where regex optimization really shines
        """
        # Simulate a long LLM response
        long_text = "This is a very long generated response. " * 100  # 4KB of text
        response = {
            "id": "completion-xyz",
            "content": long_text,
            "tokens": 500,
            "model": "gpt-4"
        }

        json_str = json.dumps(response)

        # Stream with realistic chunk sizes
        results = list(loads(simulate_network_chunks(json_str, avg_size=100, variance=50)))

        self.assertTrue(len(results) > 0)
        final_state, final_obj = results[-1]
        self.assertEqual(final_state, ParseState.COMPLETE)
        self.assertEqual(final_obj['content'], long_text)
        self.assertEqual(final_obj['tokens'], 500)


# ============================================================================
# Performance Threshold Tests (Regression Detection)
# ============================================================================

class TestPerformanceThresholds(unittest.TestCase):
    """
    Performance tests with hard thresholds
    These FAIL if the parser gets too slow (regression detection)
    """

    def test_large_json_performance(self):
        """
        50KB JSON should parse in < 30ms
        This catches major performance regressions
        """
        # Generate realistic large JSON (like LLM with long responses)
        data = {f"field_{i}": "x" * 500 for i in range(100)}
        json_str = json.dumps(data)

        self.assertGreater(len(json_str), 50_000, "Test data should be > 50KB")

        start = time.perf_counter()
        results = list(loads(stream_fixed_chunks(json_str, 100)))
        elapsed = time.perf_counter() - start

        # Hard threshold: should be fast with regex optimization
        self.assertLess(elapsed, 0.030, f"Too slow: {elapsed*1000:.1f}ms (max: 30ms)")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[-1][0], ParseState.COMPLETE)

    def test_deeply_nested_performance(self):
        """
        Deep nesting should still be fast
        Tests stack management efficiency
        """
        json_str = '{"a":' * 50 + '{"value":"test"}' + '}' * 50

        start = time.perf_counter()
        results = list(loads(stream_fixed_chunks(json_str, 50)))
        elapsed = time.perf_counter() - start

        # Should handle deep nesting efficiently
        self.assertLess(elapsed, 0.005, f"Too slow: {elapsed*1000:.1f}ms (max: 5ms)")
        self.assertEqual(results[-1][0], ParseState.COMPLETE)

    def test_many_small_fields_performance(self):
        """
        Many fields should parse quickly
        Tests structural scanning efficiency
        """
        data = {f"f{i}": i for i in range(1000)}
        json_str = json.dumps(data)

        start = time.perf_counter()
        results = list(loads(stream_fixed_chunks(json_str, 100)))
        elapsed = time.perf_counter() - start

        # Lots of structural chars - should still be fast
        self.assertLess(elapsed, 0.015, f"Too slow: {elapsed*1000:.1f}ms (max: 15ms)")
        self.assertEqual(results[-1][0], ParseState.COMPLETE)
        self.assertEqual(len(results[-1][1]), 1000)

    def test_realistic_llm_streaming_performance(self):
        """
        Realistic LLM streaming pattern performance test
        Simulates actual OpenAI API usage
        """
        # Typical GPT-4 response structure
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1699896916,
            "model": "gpt-4-turbo",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Here is a detailed explanation of the algorithm. " * 50
                },
                "finish_reason": None
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110}
        }

        json_str = json.dumps(response)

        start = time.perf_counter()
        # Simulate realistic network chunks (50-150 bytes with variance)
        results = list(loads(simulate_network_chunks(json_str, avg_size=100, variance=50)))
        elapsed = time.perf_counter() - start

        # Should be very fast for typical responses
        self.assertLess(elapsed, 0.010, f"Too slow: {elapsed*1000:.1f}ms (max: 10ms)")
        self.assertEqual(results[-1][0], ParseState.COMPLETE)


if __name__ == '__main__':
    unittest.main()
