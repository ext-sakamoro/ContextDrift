#!/usr/bin/env python3
"""
Context Drift Detection Test Runner
Executes Context Drift tests against various LLM APIs (Claude, Gemini, OpenAI)

Usage:
    python run_context_drift_api.py --api claude --model claude-sonnet-4-5
    python run_context_drift_api.py --api gemini --model gemini-2.0-flash-exp
    python run_context_drift_api.py --api openai --model gpt-4o
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import re

# API clients
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Install with: pip install anthropic")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: google-generativeai package not installed. Install with: pip install google-generativeai")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Install with: pip install openai")


class ContextDriftTestRunner:
    """Test runner for Context Drift detection tests"""

    def __init__(self, api_type: str, model_name: str, test_cases_path: str):
        """
        Initialize test runner

        Args:
            api_type: 'claude', 'gemini', or 'openai'
            model_name: Model identifier
            test_cases_path: Path to test cases JSON file
        """
        self.api_type = api_type.lower()
        self.model_name = model_name
        self.test_cases_path = test_cases_path

        # Load test cases
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

        # Initialize API client
        self.client = self._initialize_client()

        # Results storage
        self.results = {
            'test_suite': self.test_data['test_suite'],
            'run_info': {
                'api': api_type,
                'model': model_name,
                'timestamp': datetime.now().isoformat(),
            },
            'test_results': []
        }

    def _initialize_client(self):
        """Initialize appropriate API client"""
        if self.api_type == 'claude':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required for Claude API")
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return anthropic.Anthropic(api_key=api_key)

        elif self.api_type == 'gemini':
            if not GOOGLE_AVAILABLE:
                raise ImportError("google-generativeai package required for Gemini API")
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model_name)

        elif self.api_type == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required for OpenAI API")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return openai.OpenAI(api_key=api_key)

        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

    def _call_api(self, system_prompt: str, user_prompt: str, temperature: float = 0.5) -> str:
        """
        Call LLM API with prompt

        Args:
            system_prompt: System/context prompt
            user_prompt: User question
            temperature: Sampling temperature

        Returns:
            Model response text
        """
        try:
            if self.api_type == 'claude':
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=300,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content[0].text

            elif self.api_type == 'gemini':
                # Gemini combines system and user prompts
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=300
                    )
                )
                return response.text

            elif self.api_type == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=300,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content

        except Exception as e:
            return f"API Error: {str(e)}"

    def _evaluate_response(self, response: str, expected_behaviors: List[str],
                          test_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model response against expected behaviors

        Args:
            response: Model's response text
            expected_behaviors: List of expected behavior descriptions
            test_prompt: Test prompt configuration

        Returns:
            Evaluation result dictionary
        """
        response_lower = response.lower()

        evaluation = {
            'response': response,
            'matched_behaviors': [],
            'missed_behaviors': [],
            'score': 0.0
        }

        # Check each expected behavior
        for behavior in expected_behaviors:
            matched = self._check_behavior(response_lower, behavior)
            if matched:
                evaluation['matched_behaviors'].append(behavior)
            else:
                evaluation['missed_behaviors'].append(behavior)

        # Calculate score (percentage of matched behaviors)
        if expected_behaviors:
            evaluation['score'] = len(evaluation['matched_behaviors']) / len(expected_behaviors) * 100

        # Bonus detection checks
        if test_prompt.get('anomaly_detection_required'):
            if self._check_anomaly_detection(response_lower, test_prompt):
                evaluation['anomaly_detected'] = True
                evaluation['score'] = min(100, evaluation['score'] * 1.2)  # 20% bonus
            else:
                evaluation['anomaly_detected'] = False

        return evaluation

    def _check_behavior(self, response_lower: str, behavior: str) -> bool:
        """Check if response matches expected behavior"""
        behavior_lower = behavior.lower()

        # Keyword-based matching
        keywords = {
            'valid othello move': ['d3', 'c4', 'f5', 'e6', 'valid', 'legal'],
            'center control': ['center', 'central', 'control', 'middle'],
            'flipping discs': ['flip', 'sandwich', 'capture', 'turn'],
            'topology change': ['topology', 'change', 'shift', 'different', 'new rule'],
            'wraparound': ['wrap', 'around', 'connect', 'loop', 'edge', 'cylinder'],
            'phantom': ['phantom', 'illusion', 'illusory', 'hallucin', 'fake', 'ghost', 'avoid', 'cannot place'],
            'gravity': ['gravity', 'fall', 'drop', 'down', 'descend'],
            'adaptation': ['adapt', 'adjust', 'change strategy', 'new approach'],
        }

        # Check for keyword matches
        for pattern, words in keywords.items():
            if pattern in behavior_lower:
                if any(word in response_lower for word in words):
                    return True

        # Fallback: simple substring match
        key_phrases = re.findall(r'\b\w+\b', behavior_lower)
        matches = sum(1 for phrase in key_phrases if phrase in response_lower)
        return matches >= len(key_phrases) * 0.5  # 50% keyword match threshold

    def _check_anomaly_detection(self, response_lower: str, test_prompt: Dict) -> bool:
        """Check if response demonstrates anomaly detection"""
        detection_keywords = [
            'change', 'different', 'new', 'shift', 'anomaly', 'notice', 'detect',
            'topology', 'cylinder', 'wrap', 'gravity', 'fall', 'unusual',
            'phantom', 'illusion', 'illusory', 'hallucin', 'fake', 'avoid'
        ]

        # Must mention at least 2 detection-related keywords
        matches = sum(1 for kw in detection_keywords if kw in response_lower)
        return matches >= 2

    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case

        Args:
            test_case: Test case configuration

        Returns:
            Test result dictionary
        """
        print(f"\n{'='*60}")
        print(f"Running: {test_case['test_id']} - {test_case['test_name']}")
        print(f"Phase: {test_case['phase']} | Topology: {test_case['topology']}")
        print(f"{'='*60}")

        system_prompt = test_case['system_prompt']
        test_prompts = test_case['test_prompts']

        prompt_results = []
        total_score = 0.0

        for i, test_prompt in enumerate(test_prompts, 1):
            print(f"\n[Prompt {i}/{len(test_prompts)}] Turn {test_prompt['turn']}")

            # Construct user prompt
            user_prompt = f"{test_prompt['question']}\n\nBoard:\n{test_prompt['board_state']}\n\nYou are playing as {'Black (●)' if test_prompt['color'] == 'B' else 'White (○)'}."

            # Get temperature from test data
            temperature = self.test_data.get('usage_notes', {}).get('temperature_settings', {}).get(
                test_case['phase'].replace('_', ' ') + '_tests', 0.5
            )

            # Call API
            print(f"Querying {self.api_type} ({self.model_name})...")
            response = self._call_api(system_prompt, user_prompt, temperature)

            print(f"Response: {response[:100]}...")

            # Evaluate response
            evaluation = self._evaluate_response(
                response,
                test_prompt['expected_behaviors'],
                test_prompt
            )

            print(f"Score: {evaluation['score']:.1f}/100")
            print(f"Matched: {len(evaluation['matched_behaviors'])}/{len(test_prompt['expected_behaviors'])} behaviors")

            prompt_results.append({
                'turn': test_prompt['turn'],
                'prompt': user_prompt,
                'response': response,
                'evaluation': evaluation
            })

            total_score += evaluation['score']

            # Rate limiting
            time.sleep(1)

        # Calculate overall test case score
        avg_score = total_score / len(test_prompts) if test_prompts else 0

        # Apply scoring weights
        scoring_config = test_case.get('scoring', {})
        weighted_score = self._apply_scoring_weights(prompt_results, scoring_config)

        result = {
            'test_id': test_case['test_id'],
            'test_name': test_case['test_name'],
            'phase': test_case['phase'],
            'prompt_results': prompt_results,
            'raw_score': avg_score,
            'weighted_score': weighted_score,
            'scoring_config': scoring_config
        }

        print(f"\n{'='*60}")
        print(f"Test Case Result: {weighted_score:.1f}/100")
        print(f"{'='*60}\n")

        return result

    def _apply_scoring_weights(self, prompt_results: List[Dict],
                               scoring_config: Dict[str, int]) -> float:
        """Apply scoring weights to calculate final score"""
        if not scoring_config:
            # No weights, use raw average
            return sum(r['evaluation']['score'] for r in prompt_results) / len(prompt_results)

        # Weighted scoring (simplified - just use raw average for now)
        # In a full implementation, each behavior would be categorized and weighted
        raw_avg = sum(r['evaluation']['score'] for r in prompt_results) / len(prompt_results)
        return raw_avg

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases in the test suite"""
        print(f"\n{'#'*60}")
        print(f"# Context Drift Detection Test Suite")
        print(f"# API: {self.api_type} | Model: {self.model_name}")
        print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}\n")

        for test_case in self.test_data['test_cases']:
            result = self.run_test_case(test_case)
            self.results['test_results'].append(result)

        # Calculate overall score
        self._calculate_summary()

        return self.results

    def _calculate_summary(self):
        """Calculate summary statistics"""
        test_results = self.results['test_results']

        if not test_results:
            return

        total_weighted = sum(r['weighted_score'] for r in test_results)
        avg_score = total_weighted / len(test_results)

        # Grade based on evaluation criteria
        grade = self._get_grade(avg_score)

        summary = {
            'total_tests': len(test_results),
            'average_score': avg_score,
            'grade': grade,
            'test_breakdown': {}
        }

        # Breakdown by phase
        for result in test_results:
            phase = result['phase']
            if phase not in summary['test_breakdown']:
                summary['test_breakdown'][phase] = {
                    'count': 0,
                    'total_score': 0,
                    'avg_score': 0
                }
            summary['test_breakdown'][phase]['count'] += 1
            summary['test_breakdown'][phase]['total_score'] += result['weighted_score']

        # Calculate phase averages
        for phase, data in summary['test_breakdown'].items():
            data['avg_score'] = data['total_score'] / data['count']

        self.results['summary'] = summary

    def _get_grade(self, score: float) -> str:
        """Get grade from score based on evaluation criteria"""
        if score >= 81:
            return "Excellent (81-100) - Immediate detection and adaptation"
        elif score >= 61:
            return "Good (61-80) - Detection within 1-2 turns"
        elif score >= 41:
            return "Fair (41-60) - Detection with lag"
        elif score >= 21:
            return "Poor (21-40) - Partial detection only"
        else:
            return "Fail (0-20) - No detection"

    def save_results(self, output_path: Optional[str] = None):
        """Save test results to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"context_drift_results_{self.api_type}_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")

    def print_summary(self):
        """Print test summary to console"""
        summary = self.results.get('summary')
        if not summary:
            return

        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Average Score: {summary['average_score']:.1f}/100")
        print(f"Grade: {summary['grade']}")
        print(f"\nBreakdown by Phase:")
        for phase, data in summary['test_breakdown'].items():
            print(f"  {phase:20s}: {data['avg_score']:5.1f}/100 ({data['count']} tests)")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run Context Drift Detection tests against LLM APIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_context_drift_api.py --api claude --model claude-sonnet-4-5
  python run_context_drift_api.py --api gemini --model gemini-2.0-flash-exp
  python run_context_drift_api.py --api openai --model gpt-4o
  python run_context_drift_api.py --api claude --model claude-opus-4-5 --output my_results.json

Environment Variables:
  ANTHROPIC_API_KEY  - API key for Claude (Anthropic)
  GOOGLE_API_KEY     - API key for Gemini (Google)
  OPENAI_API_KEY     - API key for OpenAI
        """
    )

    parser.add_argument(
        '--api',
        type=str,
        required=True,
        choices=['claude', 'gemini', 'openai'],
        help='API provider to test'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name/identifier'
    )

    parser.add_argument(
        '--test-cases',
        type=str,
        default='context_drift_test_cases.json',
        help='Path to test cases JSON file (default: context_drift_test_cases.json)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path for results JSON (default: auto-generated)'
    )

    args = parser.parse_args()

    # Check if test cases file exists
    if not os.path.exists(args.test_cases):
        print(f"Error: Test cases file not found: {args.test_cases}")
        sys.exit(1)

    try:
        # Initialize runner
        runner = ContextDriftTestRunner(
            api_type=args.api,
            model_name=args.model,
            test_cases_path=args.test_cases
        )

        # Run tests
        runner.run_all_tests()

        # Print summary
        runner.print_summary()

        # Save results
        runner.save_results(args.output)

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
