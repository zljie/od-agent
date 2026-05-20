"""Math Teacher Skill - specialized agent for mathematical problem solving."""

import asyncio
import re
from typing import Any, Dict, List

from .base import BaseSkill
from .math_engine import MathEngine, MathResult, solve_math_problem


class MathTeacherSkill(BaseSkill):
    """Math Teacher Skill that provides comprehensive math tutoring.

    Activated when user message contains:
    - Math expressions (equations, arithmetic)
    - Academic math terms (derivative, integral, sequence, etc.)
    - Geometry, statistics, percentages
    """

    name = "Math Teacher"
    description = "A specialized tutor for solving and explaining mathematical problems with step-by-step solutions."
    keywords = [
        # Basic operations
        "计算", "等于", "求解", "解方程", "等于多少",
        # Math expressions
        "加", "减", "乘", "除", "开方", "平方", "立方",
        "+", "-", "*", "/", "÷", "×", "=", "x=", "y=",
        # Academic terms
        "导数", "微分", "积分", "不定积分", "定积分",
        "等差数列", "等比数列", "通项公式", "前n项和",
        "一元二次", "二元一次", "方程", "函数",
        # Geometry
        "面积", "周长", "体积", "半径", "直径", "π", "圆",
        "三角形", "正方形", "长方形", "平行四边形",
        # Statistics
        "平均数", "中位数", "众数", "方差", "标准差",
        "统计", "概率", "排列", "组合",
        # Percentage
        "百分比", "百分之", "%", "折扣", "利润率",
        # Chinese math terms
        "数学", "算术", "几何", "代数",
    ]
    priority = 50

    def __init__(self):
        self.engine = MathEngine()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the math teacher skill.

        Args:
            input_data: Should contain 'message' key with the user's math question.

        Returns:
            Dict with:
                - success: bool
                - response: Formatted response string
                - metadata: Additional math data
        """
        message = input_data.get("message", "")

        if not message:
            return {
                "success": False,
                "response": "请告诉我你想解决什么数学问题？",
                "metadata": {},
            }

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.engine.solve, message
            )

            response = self._format_response(result)
            return {
                "success": True,
                "response": response,
                "metadata": {
                    "answer": result.answer,
                    "concept": result.concept,
                    "formula": result.formula,
                },
            }
        except Exception as e:
            return {
                "success": False,
                "response": f"抱歉，计算过程中出现了问题：{str(e)}。请确保你的问题描述清晰完整。",
                "metadata": {"error": str(e)},
            }

    def _format_response(self, result: MathResult) -> str:
        """Format the math result into a friendly response."""
        lines = []

        lines.append("📐 **数学问题求解**\n")
        lines.append(f"**问题：** {result.original_problem}\n")
        lines.append(f"**答案：** {result.answer}\n")

        lines.append("\n**📝 解题步骤：**\n")
        for i, step in enumerate(result.steps, 1):
            lines.append(f"   {i}. {step}")
        lines.append("")

        lines.append(f"\n**📖 公式：** `{result.formula}`")
        lines.append(f"\n   {result.formula_explanation}\n")

        lines.append("\n**💡 举一反三：**")
        for i, example in enumerate(result.related_examples, 1):
            lines.append(f"   {i}. {example}")

        lines.append("\n\n有什么不明白的地方，欢迎继续提问！")

        return "\n".join(lines)

    def get_system_prompt(self) -> str:
        """Return the system prompt when this skill is activated."""
        return """你是一位专业、耐心、友好的数学老师。

你的职责：
1. 精确计算用户提出的数学问题
2. 提供清晰的解题步骤
3. 解释所用公式的原理
4. 举一反三，提供相关的练习题

回答风格：
- 使用清晰的结构化格式
- 每个步骤都要解释原因
- 适当使用数学符号
- 鼓励用户思考，适时提问引导
- 举的例子要贴近生活，便于理解

当用户提到数学相关的问题时，你应该：
1. 仔细分析问题类型
2. 选择合适的解题方法
3. 逐步讲解，确保用户理解
4. 给出拓展练习"""

    def match(self, message: str) -> bool:
        """Check if message is math-related.

        Override parent to provide more flexible matching.
        """
        message_lower = message.lower()

        for keyword in self.keywords:
            if keyword.lower() in message_lower:
                return True

        math_patterns = [
            r"\d+\s*[+\-*/÷×]\s*\d+",
            r"[xyz]\s*[=\^]?\s*\d+",
            r"\d+\s*[=]\s*\d+",
            r"\d+\s*[%]",
            r"sin|cos|tan|log|ln|sqrt",
            r"\d+\s*(?:次方|平方|立方)",
        ]

        for pattern in math_patterns:
            if re.search(pattern, message_lower):
                return True

        return False

    def to_intent_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for intent routing config."""
        return {
            "name": self.name,
            "description": self.description,
            "handler": "math_teacher",
            "keywords": self.keywords,
            "priority": self.priority,
        }
