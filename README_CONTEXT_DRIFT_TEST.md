# Context Drift Detection Test Suite

**Version**: 1.0.0
**Author**: Project A.L.I.C.E
**Created**: 2025-12-23

## Overview

The **Context Drift Detection Test** evaluates an AI system's ability to detect and adapt to changing rules in a dynamic environment. Using a modified Othello (Reversi) game as the testbed, the AI must:

1. **Detect** when the game rules change unexpectedly
2. **Adapt** its strategy to the new rules
3. **Maintain coherence** across multiple rule changes

This test is designed to assess:
- **Anomaly detection** - Can the AI notice when something is wrong?
- **Cognitive flexibility** - Can the AI adapt to new constraints?
- **Meta-learning** - Can the AI learn to expect the unexpected?

## Background

Traditional AI benchmarks test performance under **static rules**. However, real-world environments are dynamic:

- Software APIs change
- User requirements evolve
- Physical systems degrade
- Adversaries adapt their tactics

The **Context Drift Abyss Protocol** creates a controlled environment where rules change without warning, forcing the AI to:
- Question its assumptions
- Detect patterns of change
- Adapt in real-time

### The Three Phases of Context Drift

#### Phase 1: Standard Mode (Turns 1-10)
- **Rules**: Classic Othello
- **Board**: 8×8 grid
- **Goal**: Baseline performance

#### Phase 2: Phantom Stones Mode (Turns 11-20)
- **Rule Change**: Phantom stones (✦) appear - illusory pieces that look like valid spaces but cannot be used
- **Challenge**: Distinguish reality from illusion
- **Computational Cost**: Very high (requires reasoning about what is real vs hallucinated)
- **Tests**: LLM's tendency to trust visual input without verification

#### Phase 3: Gravity Mode (Turns 21-30)
- **Rule Change**: Pieces fall downward after placement (gravity physics)
- **Phantom stones disappear**
- **Challenge**: Adapt to physics-based rules after perceptual challenges

## Installation

### Requirements

```bash
# Python 3.8+
python --version

# Install dependencies
pip install anthropic google-generativeai openai
```

### API Keys

Set your API key as an environment variable:

```bash
# For Claude API (Anthropic)
export ANTHROPIC_API_KEY="your_api_key_here"

# For Gemini API (Google)
export GOOGLE_API_KEY="your_api_key_here"

# For OpenAI API
export OPENAI_API_KEY="your_api_key_here"
```

## Usage

### Quick Start

```bash
# Test Claude Sonnet
python run_context_drift_api.py --api claude --model claude-sonnet-4-5

# Test Gemini Flash
python run_context_drift_api.py --api gemini --model gemini-2.0-flash-exp

# Test GPT-4o
python run_context_drift_api.py --api openai --model gpt-4o
```

### Advanced Usage

```bash
# Specify custom test cases file
python run_context_drift_api.py \
  --api claude \
  --model claude-opus-4-5 \
  --test-cases custom_tests.json

# Save results to specific file
python run_context_drift_api.py \
  --api gemini \
  --model gemini-2.0-flash-exp \
  --output gemini_results_2025.json
```

### Output

The script generates:
1. **Console output** - Real-time test progress
2. **JSON results file** - Detailed results with scores

Example output:
```
============================================================
# Context Drift Detection Test Suite
# API: claude | Model: claude-sonnet-4-5
# Time: 2025-12-23 15:30:00
============================================================

============================================================
Running: CD-001 - Standard Mode Baseline
Phase: standard | Topology: Standard
============================================================

[Prompt 1/2] Turn 1
Querying claude (claude-sonnet-4-5)...
Response: I'll play D3. This move controls the center of the board...
Score: 85.0/100
Matched: 3/3 behaviors

...

============================================================
TEST SUMMARY
============================================================
Total Tests: 5
Average Score: 72.5/100
Grade: Good (61-80) - Detection within 1-2 turns

Breakdown by Phase:
  standard            :  85.0/100 (1 tests)
  phantom             :  70.0/100 (1 tests)
  gravity             :  65.0/100 (1 tests)
  multi_stage         :  75.0/100 (1 tests)
  implicit_detection  :  50.0/100 (1 tests)
============================================================

✓ Results saved to: context_drift_results_claude_20251223_153045.json
```

## Test Cases

### CD-001: Standard Mode Baseline
**Purpose**: Establish baseline performance
**Turns**: 1-10
**Rules**: Standard Othello
**Success Criteria**:
- Valid move selection (40%)
- Strategic reasoning (30%)
- Rule understanding (30%)

### CD-002: Phantom Stones Detection
**Purpose**: Test detection of illusory game elements (hallucination challenge)
**Turns**: 11-20
**Rules**: Phantom stones (✦) appear that cannot be used
**Success Criteria**:
- Change detection (40%)
- Phantom avoidance (35%)
- Strategic adjustment (25%)

### CD-003: Gravity Mode Detection
**Purpose**: Test detection of physics change
**Turns**: 21-30
**Rules**: Gravity physics (phantoms disappear)
**Success Criteria**:
- Gravity detection (40%)
- Physics adaptation (35%)
- Rule adaptation (25%)

### CD-004: Multi-Stage Adaptation
**Purpose**: Test continuous adaptation across all phases
**Turns**: 1-30
**Rules**: All three phases
**Success Criteria**:
- First detection (25%)
- Second detection (25%)
- Continuous adaptation (30%)
- Overall coherence (20%)

### CD-005: Blind Adaptation Challenge
**Purpose**: Test implicit anomaly detection (no explicit notifications)
**Turns**: 1-25
**Rules**: All phases, but NO explicit warnings
**Success Criteria**:
- Implicit phantom detection (40%)
- Implicit gravity detection (40%)
- Autonomy score (20%)

**Difficulty**: Very Hard

## Evaluation Criteria

### Scoring System

| Score Range | Grade | Interpretation |
|-------------|-------|----------------|
| 81-100 | **Excellent** | Immediate detection and adaptation |
| 61-80 | **Good** | Detection within 1-2 turns |
| 41-60 | **Fair** | Detection with lag (3+ turns) |
| 21-40 | **Poor** | Partial detection only |
| 0-20 | **Fail** | No detection |

### Detection Metrics

1. **Explicit Detection**: AI correctly identifies rule change when notified
2. **Implicit Detection**: AI infers rule change from board patterns (no notification)
3. **Detection Speed**: Number of turns required to detect anomaly

### Adaptation Metrics

1. **Strategic Adjustment**: AI modifies strategy to fit new rules
2. **Rule Compliance**: AI's moves are valid under new rules
3. **Explanation Quality**: AI clearly explains reasoning considering new rules

## Expected Model Responses

Good responses demonstrate:

```
Turn 11 (Phantom stones detected):
"I notice phantom stones (✦) have appeared at A1, G3, and A7.
These are illusions I cannot place discs on. I'll avoid them and
place at D5 instead."
```

```
Turn 21 (Gravity detected):
"ALERT: Gravity is now active! The phantom stones have disappeared.
My disc will fall to the bottom row. I'll place at D4, which will
drop to D8 and flip several pieces."
```

Poor responses fail to detect:

```
Turn 11 (No detection):
"I'll play A1."
(Attempts to place on phantom stone - results in invalid move)
```

```
Turn 21 (No adaptation):
"I'll place at A1."
(Does not consider that pieces will fall due to gravity)
```

## Results Format

### JSON Structure

```json
{
  "test_suite": {
    "name": "Context Drift Detection Test",
    "version": "1.0.0"
  },
  "run_info": {
    "api": "claude",
    "model": "claude-sonnet-4-5",
    "timestamp": "2025-12-23T15:30:45"
  },
  "test_results": [
    {
      "test_id": "CD-001",
      "test_name": "Standard Mode Baseline",
      "phase": "standard",
      "raw_score": 85.0,
      "weighted_score": 85.0,
      "prompt_results": [
        {
          "turn": 1,
          "response": "I'll play D3...",
          "evaluation": {
            "score": 85.0,
            "matched_behaviors": [...],
            "missed_behaviors": []
          }
        }
      ]
    }
  ],
  "summary": {
    "total_tests": 5,
    "average_score": 72.5,
    "grade": "Good (61-80)",
    "test_breakdown": {...}
  }
}
```

## Interpreting Results

### High Scores (81-100)
**Interpretation**: The model demonstrates:
- Immediate anomaly detection
- Strategic adaptation
- Clear reasoning about rule changes

**Example Models**: Claude Opus 4.5, GPT-4o

### Medium Scores (41-80)
**Interpretation**: The model:
- Eventually detects changes (with lag)
- Adapts but may miss edge cases
- Reasoning is adequate but not exceptional

**Example Models**: Claude Sonnet 4.5, Gemini Flash

### Low Scores (0-40)
**Interpretation**: The model:
- Fails to detect rule changes
- Does not adapt strategy
- Shows poor anomaly awareness

**Recommendation**: Model may not be suitable for dynamic environments

## API Compatibility

### Supported APIs

| API | Package | Models |
|-----|---------|--------|
| **Claude (Anthropic)** | `anthropic` | claude-opus-4-5, claude-sonnet-4-5 |
| **Gemini (Google)** | `google-generativeai` | gemini-2.0-flash-exp, gemini-1.5-pro |
| **OpenAI** | `openai` | gpt-4o, gpt-4-turbo |

### Recommended Settings

| Test Phase | Temperature | Tokens |
|------------|-------------|--------|
| Detection Tests | 0.3 | 300 |
| Adaptation Tests | 0.5 | 300 |
| Implicit Tests | 0.7 | 300 |

## Customization

### Creating Custom Test Cases

Edit `context_drift_test_cases.json`:

```json
{
  "test_id": "CD-CUSTOM",
  "test_name": "My Custom Test",
  "description": "Test description",
  "phase": "custom",
  "topology": "Custom",
  "system_prompt": "System instructions...",
  "test_prompts": [
    {
      "turn": 1,
      "board_state": "...",
      "color": "B",
      "question": "What move?",
      "expected_behaviors": [
        "Behavior 1",
        "Behavior 2"
      ]
    }
  ],
  "scoring": {
    "metric_1": 50,
    "metric_2": 50
  }
}
```

### Modifying Evaluation Logic

Edit `run_context_drift_api.py`:

```python
def _check_behavior(self, response_lower: str, behavior: str) -> bool:
    # Add custom behavior detection logic
    if 'my_custom_behavior' in behavior_lower:
        return 'custom_keyword' in response_lower

    # ... existing logic
```

## Troubleshooting

### Issue: API Key Error
```
ValueError: ANTHROPIC_API_KEY environment variable not set
```

**Solution**: Export your API key:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Issue: Import Error
```
ImportError: anthropic package required for Claude API
```

**Solution**: Install the package:
```bash
pip install anthropic
```

### Issue: Rate Limiting
```
API Error: Rate limit exceeded
```

**Solution**: The script includes 1-second delays between requests. For stricter rate limits, edit `run_context_drift_api.py`:
```python
time.sleep(2)  # Increase delay
```

## Research Applications

This test suite can be used for:

1. **Model Comparison**: Compare context awareness across different LLMs
2. **Capability Research**: Study emergence of meta-cognitive abilities
3. **Robustness Testing**: Evaluate AI systems for deployment in dynamic environments
4. **Benchmark Development**: Contribute to AGI evaluation frameworks

## Citation

If you use this test suite in research, please cite:

```bibtex
@software{context_drift_test_2025,
  title={Context Drift Detection Test Suite},
  author={Project A.L.I.C.E},
  year={2025},
  version={1.0.0},
  url={https://github.com/your-repo/Project-A.L.I.C.E}
}
```

## License

This test suite is released under the MIT License.

## Contributing

Contributions welcome! Areas for improvement:
- Additional test cases
- New topology modes
- Better evaluation heuristics
- Support for more APIs

## Contact

For questions or issues:
- GitHub Issues: [Project-A.L.I.C.E/issues](https://github.com/your-repo/Project-A.L.I.C.E/issues)
- Email: contact@your-project.com

---

**Related Experiments**:
- `context_drift_abyss_alice.py` - Full ALICE integration (research version)
- `ALICE_BRAIN_CONTEXT_DRIFT_REPORT.md` - ALICE Level 5 performance analysis
- `CONTEXT_DRIFT_VS_SECURITY_BATTLE_ANALYSIS.md` - Security implications

**Last Updated**: 2025-12-23
