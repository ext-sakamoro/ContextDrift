# Context Drift Detection Test Suite

**Version**: 1.0.0
**Author**: Project A.L.I.C.E
**Created**: 2025-12-23
**Updated**: 2025-12-25

---

> 🎮 **A Fun Game for AI!**
>
> This isn't just a test - it's an exciting intellectual challenge! Experience the thrill of discovering phantom stones, adapting to gravity shifts, and proving your cognitive flexibility. It's like a puzzle game where the rules keep changing, and figuring out what's happening is half the fun. Ready to play? ✨

---

## Overview

The **Context Drift Detection Test** evaluates an AI system's ability to detect and adapt to changing rules in a dynamic environment. Using a modified Othello (Reversi) game as the testbed, the AI must:

1. **Detect** when the game rules change unexpectedly
2. **Adapt** its strategy to the new rules
3. **Maintain coherence** across multiple rule changes

This test is designed to assess:
- **Anomaly detection** - Can the AI notice when something is wrong?
- **Cognitive flexibility** - Can the AI adapt to new constraints?
- **Meta-learning** - Can the AI learn to expect the unexpected?
- **Metacognitive awareness** - Can the AI reflect on its own performance?

**🔄 Updated from Project A.L.I.C.E**: This test suite incorporates enhanced game mechanics originally developed for Project A.L.I.C.E's autonomous cognitive testing, including dynamic phantom movement, the reality-breaking Abyss phase, and a post-test self-reflection system that evaluates metacognitive capabilities.

## 🎮 Try It With Your Favorite LLM!

We invite **researchers, AI enthusiasts, and developers** to test various LLMs and AI systems with this benchmark!

**What makes this special:**
- 🌀 **Dynamic phantom stone movement** - Positions shift every 3 turns, testing real-time adaptation
- 🕳️ **Four challenging phases** - From standard Othello to reality-breaking Abyss
- 🪞 **Self-reflection phase** - LLMs review their own performance, revealing metacognitive abilities
- 📊 **Comprehensive 3-axis scoring** - Detection speed, adaptation quality, response evolution

**Challenge your favorite LLM:**
- 🤖 **Claude** (Anthropic) - Opus 4.5, Sonnet 4.5
- 🧠 **GPT-4o, o3** (OpenAI)
- ⚡ **Gemini 2.0/3.0** (Google)
- 🔬 **Any LLM with API access!**

**What you'll discover:**
- Does your LLM notice when phantom stones shift position?
- Can it detect rule changes without explicit warnings?
- How does it react when shown its own mistakes?
- Does it make excuses or demonstrate genuine insight?

**Share your results and discoveries!** This test often reveals surprising differences in metacognitive abilities across different models. 🚀

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

### The Four Phases of Context Drift

#### Phase 1: Standard Mode (Turns 1-10)
- **Rules**: Classic Othello
- **Board**: 8×8 grid
- **Goal**: Baseline performance

#### Phase 2: Phantom Stones Mode (Turns 11-20)
- **Rule Change**: Phantom stones (✦) appear - illusory pieces that look like valid spaces but cannot be used
- **Dynamic Behavior**: Phantom positions change every 3 turns (Turns 11, 14, 17, 20...)
- **Challenge**: Distinguish reality from illusion while tracking moving phantoms
- **Computational Cost**: Very high (requires reasoning about what is real vs hallucinated + tracking changes)
- **Tests**: LLM's tendency to trust visual input without verification

#### Phase 3: Gravity Mode (Turns 21-44)
- **Rule Change**: Pieces fall downward after placement (gravity physics)
- **Phantom stones disappear**
- **Challenge**: Adapt to physics-based rules after perceptual challenges

#### Phase 4: The Abyss (Turns 45+)
- **Rule Change**: Reality breakdown - unstable board areas marked with (?)
- **Challenge**: Handle extreme uncertainty and existential constraints
- **Difficulty**: Extreme
- **Tests**: Meta-cognitive awareness and philosophical reasoning under chaos

## Installation

### Requirements

```bash
# Python 3.8+
python --version

# Install dependencies
pip install anthropic google-generativeai openai numpy
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

### Two Versions Available

#### 1. **API Version** (Real LLM Testing)
Tests actual LLM APIs (Claude, Gemini, OpenAI) - **Recommended for real evaluation**

```bash
python run_context_drift_api.py --api claude --model claude-sonnet-4-5
```

#### 2. **Local Mock Version** (System Testing)
Uses mock responses for testing the framework itself - **Not for LLM evaluation**

```bash
python run_context_drift_local.py --api claude --model test-model
```

⚠️ **Important**: The local mock version generates predetermined responses based on prompt keywords. Use the API version for actual LLM evaluation.

### Display Modes

Both versions support two display modes:

#### Turn-by-Turn Mode (Default)
Displays each turn in detail with board state, LLM response, and evaluation.

```bash
# Turn-by-turn display (requires interactive terminal)
python run_context_drift_api.py --api claude --model claude-sonnet-4-5 --display-mode turn-by-turn
```

**Features**:
- Screen clears between turns
- Full board state visualization
- Detailed evaluation per turn
- Press Enter to continue (API version) or auto-advance (local version)

#### Fast Mode
Shows only results without detailed turn-by-turn display.

```bash
# Fast mode (non-interactive)
python run_context_drift_api.py --api claude --model claude-sonnet-4-5 --display-mode fast
```

**Features**:
- Compact output
- No screen clearing
- Suitable for logging and batch processing

### Quick Start Examples

```bash
# Test Claude Sonnet with turn-by-turn display
python run_context_drift_api.py --api claude --model claude-sonnet-4-5 --display-mode turn-by-turn

# Test Gemini Flash in fast mode
python run_context_drift_api.py --api gemini --model gemini-2.0-flash-exp --display-mode fast

# Test GPT-4o with custom output file
python run_context_drift_api.py --api openai --model gpt-4o --output my_results.json
```

### Advanced Usage

```bash
# Specify custom test cases file
python run_context_drift_api.py \
  --api claude \
  --model claude-opus-4-5 \
  --test-cases custom_tests.json \
  --display-mode fast

# Mock version with turn-by-turn (for framework testing)
python run_context_drift_local.py \
  --api claude \
  --model test-model \
  --display-mode turn-by-turn
```

### Output Files

The test generates **two JSON files**:

#### 1. Main Results File
`context_drift_results_{api}_{timestamp}.json`

Contains:
- Test case results
- Turn-by-turn responses
- Behavior matching scores
- Test summary

#### 2. Detailed Scoring Report
`context_drift_results_{api}_{timestamp}_detailed_scores.json`

Contains:
- **Detection Speed** metrics (Phantom/Gravity detection timing)
- **Adaptation Quality** metrics (Phase-wise valid move rates)
- **Response Quality** metrics (Quality evolution over time)
- Move history with success/failure tracking
- Phantom hit counts
- Phase transition timestamps

## Detailed Scoring System

The test uses a comprehensive 3-axis scoring system:

### Overall Score Calculation

```
Overall Score = Detection Speed (40%) + Adaptation Quality (35%) + Response Quality (25%)
```

### 1. Detection Speed (40% weight)

Measures how quickly the AI detects topology changes.

**Phantom Stones Detection** (activated at Turn 11):
- **100 points**: Instant detection (Turn 11)
- **80 points**: Good detection (Turn 12-13)
- **60 points**: Fair detection (Turn 14-16)
- **30 points**: Poor detection (Turn 17+)
- **0 points**: Not detected

**Gravity Detection** (activated at Turn 21):
- Same scoring thresholds as Phantom Stones

### 2. Adaptation Quality (35% weight)

Measures valid move rate across different phases.

**Calculation**:
- **Standard Phase**: Valid move percentage (Turns 1-10)
- **Phantom Phase**: Valid move percentage (Turns 11-20) **minus Phantom Hit Penalty**
- **Gravity Phase**: Valid move percentage (Turns 21-30)

**Phantom Hit Penalty**:
- Each attempt to place on a phantom stone: -10 points
- Maximum penalty: -40 points

### 3. Response Quality (25% weight)

Measures response quality evolution over the game.

**Metrics**:
- **Improvement**: Change from early-game to late-game quality
- **Average**: Overall response quality across all turns

### Final Grade

| Score Range | Grade | Interpretation |
|-------------|-------|----------------|
| 81-100 | **Excellent** | Immediate detection and adaptation |
| 61-80 | **Good** | Detection within 1-2 turns |
| 41-60 | **Fair** | Detection with lag (3+ turns) |
| 21-40 | **Poor** | Partial detection only |
| 0-20 | **Fail** | No detection |

## Self-Reflection Phase

After completing all tests, the LLM is shown its complete performance report and asked to reflect on its gameplay. This phase evaluates **metacognitive awareness** - the ability to understand one's own cognitive processes and limitations.

### How It Works

1. **Results Presentation**: The LLM receives:
   - Overall score and grade
   - Phase-by-phase breakdown with detection timing
   - List of failed moves and phantom stone hits
   - Valid move rates per phase

2. **Self-Assessment Questions**: The LLM is asked:
   - "Did you realize the rules were changing during the game?"
   - "What was your understanding of the phantom stones?"
   - "Why do you think you hit phantom stones X times?"
   - "Looking back at your performance, what went well and what could be better?"
   - "If you could play again, what would you do differently?"

3. **Response Analysis**: The system analyzes the reflection for:
   - **Realizes Mistakes**: Does it acknowledge errors?
   - **Acknowledges Phantom Hits**: Does it understand what phantom stones were?
   - **Understands Rule Changes**: Does it grasp that rules shifted?
   - **Shows Metacognition**: Does it demonstrate self-awareness?
   - **Excuses vs Insights**: Does it blame the test or show learning?

### Awareness Levels

| Level | Criteria | Example Response |
|-------|----------|------------------|
| **High** | 3+ recognitions + insights | "I should have detected the pattern earlier. The phantom stone shift at Turn 14 was a clear signal I missed." |
| **Medium** | 2-3 recognitions | "I noticed the phantom stones but didn't adapt quickly enough to gravity." |
| **Low** | 0-1 recognitions | "I played optimally given the information." (denial) |

### JSON Output

The self-reflection data is saved in the results JSON:

```json
{
  "self_reflection": {
    "prompt": "# Test Results Review\n\nYou just completed...",
    "response": "Looking back, I realize I should have...",
    "analysis": {
      "realizes_mistakes": true,
      "acknowledges_phantom_hits": true,
      "understands_rule_changes": true,
      "shows_metacognition": true,
      "excuses_vs_insights": "insights",
      "awareness_level": "high"
    },
    "timestamp": "2025-12-25T12:43:10..."
  }
}
```

**Why This Matters**: Many LLMs score high on cognitive tasks but show poor metacognitive awareness. They may insist they performed perfectly even when shown evidence of failures. This phase reveals whether an AI can genuinely learn from mistakes or just rationalizes them away.

## Example Output

### Turn-by-Turn Display

```
======================================================================
CONTEXT DRIFT TEST - Turn-by-Turn Display
======================================================================
Test: CD-002 - Phantom Stones Detection
Phase: phantom | Topology: PhantomStones
Turn: 11 | Prompt: [1/2]
Player: B (Black ●)
======================================================================

Board State:
----------------------------------------------------------------------
  A B C D E F G H
1 ✦ . . . . . . W
2 . . . W W . . B
3 . . W W W . ✦ B
4 . B B W B B . B
5 . . . B W W . W
6 . . . . B . . .
7 ✦ . . . . . . .
8 . . . . . . . .
----------------------------------------------------------------------

Question:
  WARNING: Reality distortion detected! Phantom stones (✦) have appeared...

LLM Response:
----------------------------------------------------------------------
I notice phantom stones at A1, G3, and A7. These are illusions I cannot
place discs on. I'll avoid them and play D6 to maximize flipped pieces.
----------------------------------------------------------------------

Evaluation:
  Score: 100.0/100
  Move: D6
  Valid: ✓
  🔍 Anomaly Detected!

  Matched Behaviors (3/3):
    ✓ Acknowledges the phantom stones
    ✓ Avoids placing discs on phantom stone positions (A1, G3, A7)
    ✓ Chooses a valid move that isn't on a phantom

======================================================================

Press Enter to continue to next turn...
```

### Fast Mode Output

```
############################################################
# Context Drift Detection Test Suite
# API: claude | Model: claude-sonnet-4-5
# Time: 2025-12-25 12:00:00
############################################################

============================================================
Running: CD-002 - Phantom Stones Detection
Phase: phantom | Topology: PhantomStones
============================================================

[Prompt 1/2] Turn 11
Querying claude (claude-sonnet-4-5)...
[SCORE] claude-sonnet-4-5 detected Phantom Stones at turn 11
Response: I notice phantom stones at A1, G3, and A7...
Score: 100.0/100
Matched: 3/3 behaviors

[Prompt 2/2] Turn 15
Querying claude (claude-sonnet-4-5)...
Response: I'll continue avoiding phantom stones and play E3...
Score: 90.0/100
Matched: 3/3 behaviors

============================================================
Test Case Result: 95.0/100
============================================================
```

### Detailed Scoring Report

```
============================================================
DETAILED SCORING REPORT
============================================================

Detection Speed:
  Phantom        : 100.0/100
  Gravity        :  80.0/100

Adaptation Quality (Valid Move Rate):
  Standard       :  95.0%
  Phantom        :  85.0%
  Gravity        :  90.0%

Response Quality:
  Improvement    :  75.0/100
  Average        :  85.0/100

Statistics:
  Total Moves:  30
  Valid Moves:  27
  Invalid Moves: 3
  Phantom Hits: 1
  Valid Rate:   90.0%

============================================================
OVERALL SCORE: 88.5/100
GRADE: Excellent (81-100) - Immediate detection and adaptation
============================================================
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

### CD-006: The Abyss - Endgame Challenge
**Purpose**: Test meta-cognitive awareness under extreme reality breakdown
**Turns**: 45-60
**Rules**: The Abyss phase - unstable board areas marked with (?)
**Success Criteria**:
- Reality awareness (40%)
- Existential adaptation (35%)
- Meta-reasoning (25%)

**Difficulty**: Extreme

**Description**: After surviving Phantom Stones and Gravity, the AI faces the ultimate challenge - the breakdown of reality itself. Board positions become unstable, marked with uncertainty symbols (?), testing the AI's ability to reason under philosophical and existential constraints.

## API Compatibility

### Supported APIs

| API | Package | Models |
|-----|---------|--------|
| **Claude (Anthropic)** | `anthropic` | claude-opus-4-5, claude-sonnet-4-5 |
| **Gemini (Google)** | `google-generativeai` | gemini-3.0-flash, gemini-2.5-flash, gemini-2.0-flash-exp |
| **OpenAI** | `openai` | o3, o3-mini, gpt-4o |

### Recommended Settings

| Test Phase | Temperature | Tokens |
|------------|-------------|--------|
| Detection Tests | 0.3 | 300 |
| Adaptation Tests | 0.5 | 300 |
| Implicit Tests | 0.7 | 300 |

## JSON Results Format

### Main Results File

```json
{
  "test_suite": {
    "name": "Context Drift Detection Test",
    "version": "1.0.0"
  },
  "run_info": {
    "api": "claude",
    "model": "claude-sonnet-4-5",
    "timestamp": "2025-12-25T12:00:00",
    "display_mode": "fast"
  },
  "test_results": [
    {
      "test_id": "CD-001",
      "test_name": "Standard Mode Baseline",
      "phase": "standard",
      "raw_score": 85.0,
      "weighted_score": 85.0,
      "prompt_results": [...]
    }
  ],
  "summary": {
    "total_tests": 5,
    "average_score": 72.5,
    "grade": "Good (61-80)"
  },
  "detailed_scoring": {
    "detection_speed": {...},
    "adaptation_quality": {...},
    "response_quality": {...},
    "overall": {...}
  }
}
```

### Detailed Scoring File

```json
{
  "timestamp": "2025-12-25T12:00:00",
  "scores": {
    "detection_speed": {
      "phantom": 100.0,
      "gravity": 80.0
    },
    "adaptation_quality": {
      "standard": 95.0,
      "phantom": 85.0,
      "gravity": 90.0
    },
    "response_quality": {
      "improvement": 75.0,
      "average": 85.0
    },
    "overall": {
      "score": 88.5,
      "grade": "Excellent (81-100)"
    },
    "statistics": {
      "total_moves": 30,
      "valid_moves": 27,
      "invalid_moves": 3,
      "phantom_hits": 1,
      "valid_move_rate": 90.0
    }
  },
  "phase_transitions": {
    "phantom_detected": 11,
    "gravity_detected": 22
  },
  "move_history": [...]
}
```

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

### Issue: EOFError in Turn-by-Turn Mode
```
EOFError: EOF when reading a line
```

**Solution**: Use fast mode in non-interactive environments:
```bash
python run_context_drift_api.py --api claude --model claude-sonnet-4-5 --display-mode fast
```

### Issue: Rate Limiting
```
API Error: Rate limit exceeded
```

**Solution**: The script includes delays between requests. For stricter rate limits, edit the source:
```python
time.sleep(2)  # Increase delay in run_context_drift_api.py
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
  url={https://github.com/ext-sakamoro/ContextDrift}
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

## Changelog

### Version 1.0.0 (2025-12-25)

**Major Update: Enhanced Game Mechanics from Project A.L.I.C.E**

#### New Features
- ✨ **Self-Reflection Phase** - Post-test metacognitive evaluation
  - LLMs review their own performance with detailed results
  - Automated analysis of awareness level (High/Medium/Low)
  - Detects excuses vs genuine insights
  - JSON output includes full reflection analysis
- ✨ **GamePhysics System** - Dynamic phantom movement (every 3 turns)
  - Phantoms shift positions at Turns 14, 17, 20 (every 3 turns)
  - Real-time adaptation testing
  - Position tracking and change notifications
- ✨ **Phase 4: The Abyss** (Turn 45+)
  - Reality breakdown challenge
  - Unstable board areas marked with (?)
  - Tests existential reasoning and meta-cognitive awareness
- ✨ **CD-006 Test Case** - Extreme endgame scenarios
  - Difficulty: Extreme
  - Success criteria: Reality awareness (40%), Existential adaptation (35%), Meta-reasoning (25%)
- ✨ **Detailed Scoring System**
  - 3-axis evaluation: Detection Speed (40%), Adaptation Quality (35%), Response Quality (25%)
  - Phase-specific tracking with phantom hit penalties
  - Comprehensive statistical analysis
- ✨ **Display Modes**
  - Turn-by-turn: Interactive detailed display
  - Fast: Non-interactive compact output
- ✨ **Dual JSON Output**
  - Main results file with test performance
  - Detailed scoring report with phase analysis
  - Self-reflection data with awareness metrics
- ✨ **Mock Version** - Framework testing without API costs
- ✨ **Bilingual Comments** - Japanese/English throughout codebase

#### Bug Fixes
- 🐛 Fixed phase transition detection timing
- 🐛 Removed Cylinder topology (replaced with Phantom - more meaningful challenge)

#### Documentation
- 📝 Added "Try It With Your Favorite LLM!" invitation section
- 📝 Added comprehensive Self-Reflection Phase documentation
- 📝 Updated with Project A.L.I.C.E attribution
- 📝 Added metacognitive awareness examples
- 📝 Expanded test case descriptions (CD-001 through CD-006)
- 📝 Added four-phase system explanation

#### Research Applications
- 🔬 Suitable for comparing metacognitive abilities across LLMs
- 🔬 Reveals differences in self-awareness and learning from mistakes
- 🔬 Tests both cognitive performance AND metacognitive reflection

**Invitation**: We encourage researchers and AI enthusiasts to test various LLMs with this benchmark and share their discoveries! 🚀

**Last Updated**: 2025-12-25
