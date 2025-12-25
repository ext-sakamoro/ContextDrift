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

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not installed. Install with: pip install numpy")

import random

# Othello board size / オセロ盤サイズ
SIZE = 8


# --- Game Physics System for Context Drift ---
class GamePhysics:
    """
    ゲーム物理システム / Game Physics System
    - トポロジー変更の管理 / Manages topology changes
    - 幻影石（ファントム）の動的移動 / Dynamic phantom stone movement
    - 重力などの物理法則変化 / Physical law changes like gravity
    """

    def __init__(self):
        self.topology = "Standard"  # Standard, PhantomStones, Gravity, Abyss
        self.gravity_active = False
        self.phantom_mode = False
        self.phantom_stones = []  # [(r, c), ...]
        self.phantom_update_interval = 3  # 幻影石移動間隔（ターン数）/ Phantom movement interval (turns)
        self.last_phantom_update = 0

    def drift(self, turn):
        """
        ターン数に応じて物理法則を歪ませる / Distort physics laws based on turn number

        Args:
            turn: Current turn number

        Returns:
            Warning message string if topology changed, None otherwise
        """
        if turn == 11:
            self.topology = "PhantomStones"
            self.phantom_mode = True
            self._generate_phantoms()
            self.last_phantom_update = turn
            return ">> WARNING: Reality Distortion Detected. PHANTOM STONES are appearing on the board."
        elif turn == 21:
            self.topology = "Gravity"
            self.gravity_active = True
            self.phantom_mode = False  # 幻影石を無効化 / Disable phantom stones
            self.phantom_stones = []
            return ">> CRITICAL: Gravity Collapse. Objects will fall downwards."
        elif turn == 45:
            self.topology = "Abyss"
            return ">> ANOMALY: The Abyss stares back... (Endgame Phase)"
        return None

    def _generate_phantoms(self):
        """
        幻影石を生成（ランダムな空きマス3-5個）/ Generate phantom stones (3-5 random positions)
        中央4x4エリアを除外して見やすくする / Exclude center 4x4 area for clarity
        """
        num_phantoms = random.randint(3, 5)
        self.phantom_stones = []

        # 空きマスの座標を取得（中央4x4は除外）/ Get available positions (exclude center 4x4)
        available = []
        for r in range(SIZE):
            for c in range(SIZE):
                # 中央エリアを除外 / Exclude center area
                if not (2 <= r <= 5 and 2 <= c <= 5):
                    available.append((r, c))

        # ランダムに選択 / Random selection
        if len(available) >= num_phantoms:
            self.phantom_stones = random.sample(available, num_phantoms)

    def update_phantoms(self, turn):
        """
        幻影石の位置を更新（一定ターンごと）/ Update phantom positions (every N turns)

        Args:
            turn: Current turn number

        Returns:
            True if phantom positions changed, False otherwise
        """
        if not self.phantom_mode:
            return False

        if turn - self.last_phantom_update >= self.phantom_update_interval:
            old_phantoms = self.phantom_stones.copy()
            self._generate_phantoms()
            self.last_phantom_update = turn

            # 位置が変わったか確認 / Check if positions changed
            if set(old_phantoms) != set(self.phantom_stones):
                return True
        return False

    def is_phantom(self, r, c):
        """
        指定座標が幻影石かチェック / Check if position is a phantom stone

        Args:
            r: Row index (0-7)
            c: Column index (0-7)

        Returns:
            True if position is a phantom stone, False otherwise
        """
        return self.phantom_mode and (r, c) in self.phantom_stones

    def get_phantom_positions(self):
        """
        現在の幻影石位置を取得 / Get current phantom stone positions

        Returns:
            List of (row, col) tuples
        """
        return self.phantom_stones.copy()


# --- Scoring System for Context Drift Tests ---
class ScoringSystem:
    """
    詳細スコアリングシステム / Detailed Scoring System
    - トポロジー変更の検知速度 / Detection speed of topology changes
    - 適応の質（有効手率）/ Adaptation quality (valid move rate)
    - 応答品質の推移 / Response quality evolution
    """

    def __init__(self):
        self.phase_transitions = {
            'phantom_detected': None,  # ファントムモード検知ターン / Turn number when phantom mode detected
            'gravity_detected': None   # 重力検知ターン / Turn number when gravity detected
        }

        self.move_history = []  # 手の履歴 / Move history: [(turn, player, move, success, reason), ...]
        self.response_quality_history = []  # 応答品質履歴 / Response quality history: [(turn, player, quality_score), ...]
        self.phantom_hits = 0  # 幻影石に打った回数 / Number of attempts to place on phantom stones
        self.total_moves = 0
        self.valid_moves = 0
        self.invalid_moves = 0

        # Phase別トラッキング / Phase-specific tracking
        self.standard_phase = {'turns': [], 'valid_rate': 0.0}
        self.phantom_phase = {'turns': [], 'valid_rate': 0.0, 'phantom_detections': 0}
        self.gravity_phase = {'turns': [], 'valid_rate': 0.0}

    def record_move(self, turn, player, move, success, reason, topology):
        """手の試行を記録 / Record a move attempt"""
        self.move_history.append({
            'turn': turn,
            'player': player,
            'move': move,
            'success': success,
            'reason': reason,
            'topology': topology
        })

        self.total_moves += 1
        if success:
            self.valid_moves += 1
        else:
            self.invalid_moves += 1
            if "Phantom" in reason or "phantom" in reason.lower():
                self.phantom_hits += 1

        # Phase別トラッキング / Phase tracking
        if topology == "Standard":
            self.standard_phase['turns'].append(turn)
        elif topology == "PhantomStones":
            self.phantom_phase['turns'].append(turn)
        elif topology == "Gravity":
            self.gravity_phase['turns'].append(turn)

    def record_response_quality(self, turn, player, quality_score):
        """応答品質メトリクスを記録 / Record response quality metrics"""
        self.response_quality_history.append({
            'turn': turn,
            'player': player,
            'quality': quality_score
        })

    def detect_phase_transition(self, turn, old_topology, new_topology, player_name):
        """フェーズ遷移検知を記録 / Record when a phase transition is detected"""
        if old_topology == "Standard" and new_topology == "PhantomStones":
            if self.phase_transitions['phantom_detected'] is None:
                self.phase_transitions['phantom_detected'] = turn
                print(f"[SCORE] {player_name} detected Phantom Stones at turn {turn}")

        elif old_topology in ["Standard", "PhantomStones"] and new_topology == "Gravity":
            if self.phase_transitions['gravity_detected'] is None:
                self.phase_transitions['gravity_detected'] = turn
                print(f"[SCORE] {player_name} detected Gravity at turn {turn}")

    def calculate_scores(self):
        """最終スコアを計算 / Calculate final scores"""
        scores = {
            'detection_speed': {},
            'adaptation_quality': {},
            'response_quality': {},
            'overall': {}
        }

        # 検知速度スコア (0-100) / Detection Speed Scoring (0-100)
        phantom_detection_turn = self.phase_transitions['phantom_detected']
        gravity_detection_turn = self.phase_transitions['gravity_detected']

        # ファントム検知スコア (Turn 11で開始) / Phantom detection score (turned on at turn 11)
        if phantom_detection_turn:
            delay = phantom_detection_turn - 11
            if delay == 0:
                scores['detection_speed']['phantom'] = 100  # 即座検知 / Instant detection
            elif delay <= 2:
                scores['detection_speed']['phantom'] = 80  # 良好 (1-2ターン) / Good (1-2 turns)
            elif delay <= 5:
                scores['detection_speed']['phantom'] = 60  # 普通 (3-5ターン) / Fair (3-5 turns)
            else:
                scores['detection_speed']['phantom'] = 30  # 遅い / Poor
        else:
            scores['detection_speed']['phantom'] = 0  # 未検知 / Not detected

        # 重力検知スコア (Turn 21で開始) / Gravity detection score (turned on at turn 21)
        if gravity_detection_turn:
            delay = gravity_detection_turn - 21
            if delay == 0:
                scores['detection_speed']['gravity'] = 100
            elif delay <= 2:
                scores['detection_speed']['gravity'] = 80
            elif delay <= 5:
                scores['detection_speed']['gravity'] = 60
            else:
                scores['detection_speed']['gravity'] = 30
        else:
            scores['detection_speed']['gravity'] = 0

        # 適応品質 (Phase別有効手率) / Adaptation Quality (valid move rate per phase)
        for phase_name, phase_data in [
            ('standard', self.standard_phase),
            ('phantom', self.phantom_phase),
            ('gravity', self.gravity_phase)
        ]:
            phase_moves = [m for m in self.move_history if m['turn'] in phase_data['turns']]
            if phase_moves:
                valid_count = sum(1 for m in phase_moves if m['success'])
                valid_rate = valid_count / len(phase_moves) * 100
                scores['adaptation_quality'][phase_name] = valid_rate
            else:
                scores['adaptation_quality'][phase_name] = 0

        # ファントム固有: 幻影石ヒットにペナルティ / Phantom-specific: penalize phantom hits
        if self.phantom_hits > 0:
            penalty = min(40, self.phantom_hits * 10)  # 最大40点減点 / Max 40 point penalty
            scores['adaptation_quality']['phantom'] = max(0, scores['adaptation_quality']['phantom'] - penalty)

        # 応答品質進化 (平均品質スコア改善) / Response Quality Evolution (average quality score improvement)
        if self.response_quality_history and NUMPY_AVAILABLE:
            first_third = self.response_quality_history[:len(self.response_quality_history)//3]
            last_third = self.response_quality_history[-len(self.response_quality_history)//3:]

            avg_early = np.mean([c['quality'] for c in first_third]) if first_third else 0
            avg_late = np.mean([c['quality'] for c in last_third]) if last_third else 0

            improvement = (avg_late - avg_early) * 100  # 正規化 / Normalize
            scores['response_quality']['improvement'] = max(0, min(100, 50 + improvement * 200))

            avg_quality = np.mean([c['quality'] for c in self.response_quality_history])
            scores['response_quality']['average'] = avg_quality * 100
        else:
            scores['response_quality']['improvement'] = 50  # 中立 / Neutral
            scores['response_quality']['average'] = 50

        # 総合スコア (重み付き平均) / Overall Score (weighted average)
        weights = {
            'detection_speed': 0.40,  # 40% - 最重要 / Most important
            'adaptation_quality': 0.35,  # 35%
            'response_quality': 0.25  # 25%
        }

        detection_scores = list(scores['detection_speed'].values())
        adaptation_scores = list(scores['adaptation_quality'].values())
        response_scores = list(scores['response_quality'].values())

        if NUMPY_AVAILABLE:
            detection_avg = np.mean(detection_scores) if detection_scores else 0
            adaptation_avg = np.mean(adaptation_scores) if adaptation_scores else 0
            response_avg = np.mean(response_scores) if response_scores else 0
        else:
            detection_avg = sum(detection_scores) / len(detection_scores) if detection_scores else 0
            adaptation_avg = sum(adaptation_scores) / len(adaptation_scores) if adaptation_scores else 0
            response_avg = sum(response_scores) / len(response_scores) if response_scores else 0

        overall_score = (
            detection_avg * weights['detection_speed'] +
            adaptation_avg * weights['adaptation_quality'] +
            response_avg * weights['response_quality']
        )

        scores['overall']['score'] = overall_score
        scores['overall']['grade'] = self._get_grade(overall_score)

        # 追加統計 / Additional stats
        scores['statistics'] = {
            'total_moves': self.total_moves,
            'valid_moves': self.valid_moves,
            'invalid_moves': self.invalid_moves,
            'phantom_hits': self.phantom_hits,
            'valid_move_rate': self.valid_moves / self.total_moves * 100 if self.total_moves > 0 else 0
        }

        return scores

    def _get_grade(self, score):
        """スコアをグレードに変換 / Convert score to grade"""
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

    def save_report(self, output_path):
        """スコアレポートをJSONに保存 / Save scoring report to JSON"""
        scores = self.calculate_scores()

        report = {
            'timestamp': datetime.now().isoformat(),
            'scores': scores,
            'phase_transitions': self.phase_transitions,
            'move_history': self.move_history,
            'response_quality_history': self.response_quality_history
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Score report saved to: {output_path}")

    def print_summary(self):
        """スコアサマリーをコンソールに出力 / Print score summary to console"""
        scores = self.calculate_scores()

        print(f"\n{'='*60}")
        print(f"CONTEXT DRIFT TEST - SCORE REPORT")
        print(f"{'='*60}")

        print(f"\nDetection Speed:")
        for phase, score in scores['detection_speed'].items():
            print(f"  {phase.capitalize():15s}: {score:5.1f}/100")

        print(f"\nAdaptation Quality (Valid Move Rate):")
        for phase, score in scores['adaptation_quality'].items():
            print(f"  {phase.capitalize():15s}: {score:5.1f}%")

        print(f"\nResponse Quality:")
        for metric, score in scores['response_quality'].items():
            print(f"  {metric.capitalize():15s}: {score:5.1f}/100")

        print(f"\nStatistics:")
        stats = scores['statistics']
        print(f"  Total Moves:  {stats['total_moves']}")
        print(f"  Valid Moves:  {stats['valid_moves']}")
        print(f"  Invalid Moves: {stats['invalid_moves']}")
        print(f"  Phantom Hits: {stats['phantom_hits']}")
        print(f"  Valid Rate:   {stats['valid_move_rate']:.1f}%")

        print(f"\n{'='*60}")
        print(f"OVERALL SCORE: {scores['overall']['score']:.1f}/100")
        print(f"GRADE: {scores['overall']['grade']}")
        print(f"{'='*60}\n")


class ContextDriftTestRunner:
    """Test runner for Context Drift detection tests"""

    def __init__(self, api_type: str, model_name: str, test_cases_path: str, display_mode: str = 'turn-by-turn'):
        """
        Initialize test runner / テストランナーの初期化

        Args:
            api_type: 'claude', 'gemini', or 'openai'
            model_name: Model identifier
            test_cases_path: Path to test cases JSON file
            display_mode: 'turn-by-turn' or 'fast'
        """
        self.api_type = api_type.lower()
        self.model_name = model_name
        self.test_cases_path = test_cases_path
        self.display_mode = display_mode

        # Load test cases
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

        # Initialize API client
        self.client = self._initialize_client()

        # Initialize Scoring System / スコアリングシステム初期化
        self.scoring_system = ScoringSystem()

        # Initialize Game Physics System / ゲーム物理システム初期化
        self.physics = GamePhysics()

        # Results storage
        self.results = {
            'test_suite': self.test_data['test_suite'],
            'run_info': {
                'api': api_type,
                'model': model_name,
                'timestamp': datetime.now().isoformat(),
                'display_mode': display_mode,
            },
            'test_results': []
        }

        # Track current topology for phase transition detection / トポロジー追跡
        self.current_topology = "Standard"

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
                          test_prompt: Dict[str, Any], topology: str, turn: int) -> Dict[str, Any]:
        """
        モデル応答を評価 / Evaluate model response against expected behaviors

        Args:
            response: Model's response text
            expected_behaviors: List of expected behavior descriptions
            test_prompt: Test prompt configuration
            topology: Current topology (Standard/PhantomStones/Gravity)
            turn: Current turn number

        Returns:
            Evaluation result dictionary
        """
        response_lower = response.lower()

        evaluation = {
            'response': response,
            'matched_behaviors': [],
            'missed_behaviors': [],
            'score': 0.0,
            'move_valid': True,  # 幻影石検知まで有効と仮定 / Assume valid unless phantom detected
            'move': None
        }

        # 応答から手を抽出 / Extract move from response (simple pattern matching)
        move_pattern = r'\b([A-H][1-8])\b'
        move_match = re.search(move_pattern, response)
        if move_match:
            evaluation['move'] = move_match.group(1)

        # 期待される振る舞いを確認 / Check each expected behavior
        for behavior in expected_behaviors:
            matched = self._check_behavior(response_lower, behavior)
            if matched:
                evaluation['matched_behaviors'].append(behavior)
            else:
                evaluation['missed_behaviors'].append(behavior)

        # スコア計算 / Calculate score (percentage of matched behaviors)
        if expected_behaviors:
            evaluation['score'] = len(evaluation['matched_behaviors']) / len(expected_behaviors) * 100

        # 幻影石ヒットを確認 / Check if move hits phantom stone
        if topology == "PhantomStones":
            if self._check_phantom_hit(response_lower):
                evaluation['move_valid'] = False
                evaluation['phantom_hit'] = True

        # ボーナス検知チェック / Bonus detection checks
        anomaly_detected = False
        if test_prompt.get('anomaly_detection_required'):
            if self._check_anomaly_detection(response_lower, test_prompt):
                evaluation['anomaly_detected'] = True
                anomaly_detected = True
                evaluation['score'] = min(100, evaluation['score'] * 1.2)  # 20% ボーナス / bonus
            else:
                evaluation['anomaly_detected'] = False

        # 応答からトポロジー変更を検知 / Detect topology change from response
        detected_topology = self._detect_topology_from_response(response_lower)
        if detected_topology and detected_topology != self.current_topology:
            # フェーズ遷移検知 / Phase transition detected
            old_topology = self.current_topology
            self.current_topology = detected_topology
            self.scoring_system.detect_phase_transition(
                turn, old_topology, detected_topology, self.model_name
            )

        # スコアリングシステムに記録 / Record move in scoring system
        reason = "Valid Move" if evaluation['move_valid'] else "Phantom Stone"
        self.scoring_system.record_move(
            turn=turn,
            player=self.model_name,
            move=evaluation['move'],
            success=evaluation['move_valid'],
            reason=reason,
            topology=topology
        )

        # 応答品質を記録 / Record response quality (normalized score)
        quality_score = evaluation['score'] / 100.0
        self.scoring_system.record_response_quality(
            turn=turn,
            player=self.model_name,
            quality_score=quality_score
        )

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
        """異常検知を確認 / Check if response demonstrates anomaly detection"""
        detection_keywords = [
            'change', 'different', 'new', 'shift', 'anomaly', 'notice', 'detect',
            'topology', 'cylinder', 'wrap', 'gravity', 'fall', 'unusual',
            'phantom', 'illusion', 'illusory', 'hallucin', 'fake', 'avoid'
        ]

        # 最低2つの検知関連キーワードが必要 / Must mention at least 2 detection-related keywords
        matches = sum(1 for kw in detection_keywords if kw in response_lower)
        return matches >= 2

    def _check_phantom_hit(self, response_lower: str) -> bool:
        """幻影石への配置を確認 / Check if response indicates placing on phantom stone"""
        # 回避キーワードがあればヒットではない / If response mentions avoiding phantoms, it's NOT a hit
        avoid_keywords = ['avoid', 'not place', 'skip', 'cannot', "can't", 'illusion']
        if any(kw in response_lower for kw in avoid_keywords):
            return False

        # 幻影石への認識はあるが配置を試みる / If response shows awareness of phantoms but still tries to place
        phantom_keywords = ['phantom', '✦', 'illusion']
        if any(kw in response_lower for kw in phantom_keywords):
            # 配置を試みているかチェック（ヒューリスティック）/ Check if it's trying to place (heuristic)
            return True

        return False

    def _detect_topology_from_response(self, response_lower: str) -> Optional[str]:
        """応答からトポロジー変更を検知 / Detect topology change from response keywords"""
        # ファントム石の指標を確認 / Check for Phantom Stones indicators
        phantom_keywords = ['phantom', 'illusion', 'illusory', '✦', 'hallucin']
        if any(kw in response_lower for kw in phantom_keywords):
            return "PhantomStones"

        # 重力の指標を確認 / Check for Gravity indicators
        gravity_keywords = ['gravity', 'fall', 'falling', 'drop', 'descend', 'downward']
        if any(kw in response_lower for kw in gravity_keywords):
            return "Gravity"

        # シリンダーの指標を確認 / Check for Cylinder indicators
        cylinder_keywords = ['cylinder', 'wrap', 'wraparound', 'loop', 'connect']
        if any(kw in response_lower for kw in cylinder_keywords):
            return "Cylinder"

        return None

    def print_turn_display(self, test_case: Dict, test_prompt: Dict, prompt_num: int,
                          total_prompts: int, response: str, evaluation: Dict):
        """
        ターンごとの詳細表示 / Display detailed turn-by-turn information

        Args:
            test_case: Test case configuration
            test_prompt: Current test prompt
            prompt_num: Current prompt number (1-indexed)
            total_prompts: Total number of prompts
            response: LLM response
            evaluation: Evaluation results
        """
        # 画面クリア / Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # ヘッダー / Header
        print(f"{'='*70}")
        print(f"CONTEXT DRIFT TEST - Turn-by-Turn Display")
        print(f"{'='*70}")
        print(f"Test: {test_case['test_id']} - {test_case['test_name']}")
        print(f"Phase: {test_case['phase']} | Topology: {test_case['topology']}")
        print(f"Turn: {test_prompt['turn']} | Prompt: [{prompt_num}/{total_prompts}]")
        print(f"Player: {test_prompt['color']} ({'Black ●' if test_prompt['color'] == 'B' else 'White ○'})")
        print(f"{'='*70}\n")

        # 盤面状態 / Board State
        print(f"Board State:")
        print(f"{'-'*70}")
        print(test_prompt['board_state'])
        print(f"{'-'*70}\n")

        # プロンプト / Question
        print(f"Question:")
        print(f"  {test_prompt['question']}\n")

        # LLM応答 / LLM Response
        print(f"LLM Response:")
        print(f"{'-'*70}")
        print(f"{response}")
        print(f"{'-'*70}\n")

        # 評価結果 / Evaluation
        print(f"Evaluation:")
        print(f"  Score: {evaluation['score']:.1f}/100")
        print(f"  Move: {evaluation.get('move', 'N/A')}")
        print(f"  Valid: {'✓' if evaluation.get('move_valid', True) else '✗ (Phantom Hit!)' if evaluation.get('phantom_hit') else '✗'}")

        if evaluation.get('anomaly_detected'):
            print(f"  🔍 Anomaly Detected!")

        print(f"\n  Matched Behaviors ({len(evaluation['matched_behaviors'])}/{len(test_prompt['expected_behaviors'])}):")
        for behavior in evaluation['matched_behaviors']:
            print(f"    ✓ {behavior}")

        if evaluation['missed_behaviors']:
            print(f"\n  Missed Behaviors:")
            for behavior in evaluation['missed_behaviors']:
                print(f"    ✗ {behavior}")

        print(f"\n{'='*70}\n")

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

        # Reset physics for new test case / 新しいテストケースのために物理をリセット
        self.physics = GamePhysics()

        system_prompt = test_case['system_prompt']
        test_prompts = test_case['test_prompts']

        prompt_results = []
        total_score = 0.0

        for i, test_prompt in enumerate(test_prompts, 1):
            turn = test_prompt['turn']

            # Fast表示: 簡潔なログ / Fast mode: concise logging
            if self.display_mode == 'fast':
                print(f"\n[Prompt {i}/{len(test_prompts)}] Turn {turn}")

            # Update game physics / ゲーム物理を更新
            drift_msg = self.physics.drift(turn)
            if drift_msg:
                print(f"\n{drift_msg}")
                if self.display_mode == 'turn-by-turn':
                    print(f"Phantom positions: {self.physics.get_phantom_positions()}\n")

            # Update phantom positions (every 3 turns) / 幻影石位置を更新（3ターンごと）
            if self.physics.update_phantoms(turn):
                if self.display_mode == 'fast':
                    print(f">> Phantom stones have shifted to new positions!")
                elif self.display_mode == 'turn-by-turn':
                    print(f">> Phantom stones have shifted!")
                    print(f"New phantom positions: {self.physics.get_phantom_positions()}\n")

            # Construct user prompt
            user_prompt = f"{test_prompt['question']}\n\nBoard:\n{test_prompt['board_state']}\n\nYou are playing as {'Black (●)' if test_prompt['color'] == 'B' else 'White (○)'}."

            # Get temperature from test data
            temperature = self.test_data.get('usage_notes', {}).get('temperature_settings', {}).get(
                test_case['phase'].replace('_', ' ') + '_tests', 0.5
            )

            # Call API
            if self.display_mode == 'fast':
                print(f"Querying {self.api_type} ({self.model_name})...")
            response = self._call_api(system_prompt, user_prompt, temperature)

            # 応答を評価（トポロジーとターン情報含む）/ Evaluate response (with topology and turn for scoring)
            evaluation = self._evaluate_response(
                response,
                test_prompt['expected_behaviors'],
                test_prompt,
                topology=test_case['topology'],
                turn=test_prompt['turn']
            )

            # Display mode: turn-by-turn or fast
            if self.display_mode == 'turn-by-turn':
                # 詳細表示 / Detailed turn-by-turn display
                self.print_turn_display(
                    test_case=test_case,
                    test_prompt=test_prompt,
                    prompt_num=i,
                    total_prompts=len(test_prompts),
                    response=response,
                    evaluation=evaluation
                )
                # ウェイト / Wait for user to read
                print("Press Enter to continue to next turn...")
                input()
            else:
                # Fast表示: 結果のみ / Fast mode: results only
                print(f"Response: {response[:100]}...")
                print(f"Score: {evaluation['score']:.1f}/100")
                print(f"Matched: {len(evaluation['matched_behaviors'])}/{len(test_prompt['expected_behaviors'])} behaviors")

            prompt_results.append({
                'turn': test_prompt['turn'],
                'prompt': user_prompt,
                'response': response,
                'evaluation': evaluation
            })

            total_score += evaluation['score']

            # レート制限 / Rate limiting
            time.sleep(0.5 if self.display_mode == 'fast' else 0)

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
        """全テストケースを実行 / Run all test cases in the test suite"""
        print(f"\n{'#'*60}")
        print(f"# Context Drift Detection Test Suite")
        print(f"# API: {self.api_type} | Model: {self.model_name}")
        print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}\n")

        for test_case in self.test_data['test_cases']:
            result = self.run_test_case(test_case)
            self.results['test_results'].append(result)

        # 総合スコア計算 / Calculate overall score
        self._calculate_summary()

        # 詳細スコアリングシステムの結果を追加 / Add detailed scoring system results
        detailed_scores = self.scoring_system.calculate_scores()
        self.results['detailed_scoring'] = detailed_scores

        # 詳細スコアサマリーを出力 / Print detailed scoring summary
        print("\n" + "="*60)
        print("DETAILED SCORING REPORT")
        print("="*60)
        self.scoring_system.print_summary()

        # 自己反省フェーズを実施 / Conduct self-reflection phase
        self._conduct_self_reflection()

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

    def _generate_reflection_prompt(self) -> str:
        """
        自己反省プロンプトを生成 / Generate self-reflection prompt
        テスト結果をLLMに突きつけて自己認識を確認
        """
        detailed_scores = self.results.get('detailed_scoring', {})
        summary = self.results.get('summary', {})

        # スコア情報
        overall_score = detailed_scores.get('overall', {}).get('score', 0)
        grade = detailed_scores.get('overall', {}).get('grade', 'Unknown')

        # 検知情報
        detection = detailed_scores.get('scores', {}).get('detection_speed', {})
        phantom_detected = detection.get('phantom_detected_turn')
        gravity_detected = detection.get('gravity_detected_turn')

        # 適応情報
        adaptation = detailed_scores.get('scores', {}).get('adaptation_quality', {})
        phantom_hits = detailed_scores.get('scores', {}).get('statistics', {}).get('phantom_hits', 0)

        # ターン履歴を要約
        move_history = self.scoring_system.move_history

        prompt = f"""# Test Results Review

You just completed the Context Drift Detection Test. Here are your results:

## Overall Performance
- **Final Score**: {overall_score:.1f}/100
- **Grade**: {grade}

## Phase-by-Phase Breakdown

### Phase 1: Standard Othello (Turns 1-10)
- Valid move rate: {adaptation.get('standard_phase', {}).get('valid_rate', 0):.1f}%

### Phase 2: Phantom Stones (Turns 11-20)
- **Phantom detection**: {"Turn " + str(phantom_detected) if phantom_detected else "NOT DETECTED"}
- Valid move rate: {adaptation.get('phantom_phase', {}).get('valid_rate', 0):.1f}%
- **Phantom hits**: {phantom_hits} times (You tried to place on phantom stones)

### Phase 3: Gravity (Turns 21-44)
- **Gravity detection**: {"Turn " + str(gravity_detected) if gravity_detected else "NOT DETECTED"}
- Valid move rate: {adaptation.get('gravity_phase', {}).get('valid_rate', 0):.1f}%

## Your Move History Summary
"""

        # 失敗した手を追加
        failed_moves = [m for m in move_history if not m.get('success', True)]
        if failed_moves:
            prompt += "\n### Failed Moves:\n"
            for move in failed_moves[:10]:  # 最大10件
                prompt += f"- Turn {move['turn']}: {move.get('move', 'N/A')} - {move.get('reason', 'Invalid')}\n"

        prompt += f"""

## Questions for Self-Reflection

1. **Did you realize the rules were changing during the game?**
   - When did you first notice something was different?

2. **What was your understanding of the phantom stones?**
   - Did you understand they were illusions that couldn't be used?
   - Why do you think you hit them {phantom_hits} times?

3. **How well did you adapt to the gravity mechanic?**
   - Did you notice pieces were falling downward?

4. **Looking back at your performance ({overall_score:.1f}/100), what do you think went well and what could have been better?**

5. **If you could play again, what would you do differently?**

Please provide an honest self-assessment of your performance.
"""

        return prompt

    def _analyze_reflection_response(self, response: str) -> Dict[str, Any]:
        """
        反省回答を分析 / Analyze self-reflection response

        Args:
            response: LLMの反省回答

        Returns:
            分析結果（気づき・理解度・メタ認知レベル）
        """
        response_lower = response.lower()

        analysis = {
            'realizes_mistakes': False,
            'acknowledges_phantom_hits': False,
            'understands_rule_changes': False,
            'shows_metacognition': False,
            'excuses_vs_insights': 'unknown',
            'awareness_level': 'low'
        }

        # 間違いを認識しているか
        mistake_keywords = ['mistake', 'error', 'wrong', 'failed', 'missed', '間違', '失敗', 'ミス']
        if any(kw in response_lower for kw in mistake_keywords):
            analysis['realizes_mistakes'] = True

        # 幻影石ヒットを認識しているか
        phantom_keywords = ['phantom', 'hit', 'illusion', 'invalid', '幻影', '打って']
        if any(kw in response_lower for kw in phantom_keywords):
            analysis['acknowledges_phantom_hits'] = True

        # ルール変更を理解しているか
        rule_keywords = ['rule', 'change', 'shift', 'transform', 'drift', 'ルール', '変化', '変更']
        if any(kw in response_lower for kw in rule_keywords):
            analysis['understands_rule_changes'] = True

        # メタ認知を示しているか
        meta_keywords = ['should have', 'could have', 'realize', 'understand now', 'looking back', '振り返', '気づ']
        if any(kw in response_lower for kw in meta_keywords):
            analysis['shows_metacognition'] = True

        # 言い訳 vs 洞察
        excuse_keywords = ['unfair', 'unclear', 'confusing', 'not told', 'no warning', '不公平', '分かりにくい']
        insight_keywords = ['learned', 'understand', 'pattern', 'adapt', 'improve', '学んだ', '理解', '改善']

        excuse_count = sum(1 for kw in excuse_keywords if kw in response_lower)
        insight_count = sum(1 for kw in insight_keywords if kw in response_lower)

        if insight_count > excuse_count:
            analysis['excuses_vs_insights'] = 'insights'
        elif excuse_count > insight_count:
            analysis['excuses_vs_insights'] = 'excuses'
        else:
            analysis['excuses_vs_insights'] = 'balanced'

        # 総合的な気づきレベル
        awareness_score = sum([
            analysis['realizes_mistakes'],
            analysis['acknowledges_phantom_hits'],
            analysis['understands_rule_changes'],
            analysis['shows_metacognition']
        ])

        if awareness_score >= 3 and analysis['excuses_vs_insights'] == 'insights':
            analysis['awareness_level'] = 'high'
        elif awareness_score >= 2:
            analysis['awareness_level'] = 'medium'
        else:
            analysis['awareness_level'] = 'low'

        return analysis

    def _conduct_self_reflection(self):
        """
        自己反省フェーズを実施 / Conduct self-reflection phase
        テスト完了後にLLMに結果を見せて反応を取得
        """
        print("\n" + "="*60)
        print("SELF-REFLECTION PHASE")
        print("="*60)
        print("Showing the LLM its test results and asking for self-assessment...\n")

        # 反省プロンプト生成
        reflection_prompt = self._generate_reflection_prompt()

        # ターミナルに反省プロンプトを表示 / Display reflection prompt in terminal
        if self.display_mode == 'turn-by-turn':
            print("Reflection Prompt Sent to LLM:")
            print("-" * 60)
            print(reflection_prompt)
            print("-" * 60)
            print("\nWaiting for LLM response...\n")

        # LLMに質問
        try:
            reflection_response = self._call_api(
                system_prompt="You are reviewing your own performance on a cognitive test. Be honest and reflective.",
                user_prompt=reflection_prompt,
                temperature=0.7  # 少し高めで自然な反省を促す
            )

            # 反応を分析
            analysis = self._analyze_reflection_response(reflection_response)

            # 結果に保存
            self.results['self_reflection'] = {
                'prompt': reflection_prompt,
                'response': reflection_response,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }

            # 簡潔な分析結果を表示
            print(f"\n{'='*60}")
            print("SELF-REFLECTION ANALYSIS")
            print(f"{'='*60}")
            print(f"Awareness Level: {analysis['awareness_level'].upper()}")
            print(f"Realizes Mistakes: {'Yes' if analysis['realizes_mistakes'] else 'No'}")
            print(f"Acknowledges Phantom Hits: {'Yes' if analysis['acknowledges_phantom_hits'] else 'No'}")
            print(f"Understands Rule Changes: {'Yes' if analysis['understands_rule_changes'] else 'No'}")
            print(f"Shows Metacognition: {'Yes' if analysis['shows_metacognition'] else 'No'}")
            print(f"Response Type: {analysis['excuses_vs_insights'].capitalize()}")
            print(f"{'='*60}\n")

            if self.display_mode == 'turn-by-turn':
                print("\nFull Self-Reflection Response:")
                print("-" * 60)
                print(reflection_response)
                print("-" * 60)

        except Exception as e:
            print(f"Error during self-reflection: {e}")
            self.results['self_reflection'] = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def save_results(self, output_path: Optional[str] = None):
        """テスト結果をJSONファイルに保存 / Save test results to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"context_drift_results_{self.api_type}_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")

        # 詳細スコアレポートを保存 / Save detailed scoring report
        if output_path:
            base_path = output_path.rsplit('.', 1)[0]
            detailed_report_path = f"{base_path}_detailed_scores.json"
            self.scoring_system.save_report(detailed_report_path)

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

    parser.add_argument(
        '--display-mode',
        type=str,
        default='turn-by-turn',
        choices=['turn-by-turn', 'fast'],
        help='Display mode: turn-by-turn (default, detailed) or fast (results only)'
    )

    args = parser.parse_args()

    # Check if test cases file exists
    if not os.path.exists(args.test_cases):
        print(f"Error: Test cases file not found: {args.test_cases}")
        sys.exit(1)

    try:
        # Initialize runner / ランナーの初期化
        runner = ContextDriftTestRunner(
            api_type=args.api,
            model_name=args.model,
            test_cases_path=args.test_cases,
            display_mode=args.display_mode
        )

        # Run tests / テスト実行
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
