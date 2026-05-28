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

        Supports three input modes:
        - params mode (Planner-driven): {"params": {"operation": "journey_avg", ...}}
          Dispatches to specialized journey_avg handler.
        - expression mode (Planner-driven): {"params": {"expression": "..."}}
          Passes expression directly to MathEngine.
        - message mode (legacy): {"message": "..."}
          Uses existing word-problem parsing.
        """
        params = input_data.get("params", {})
        message = input_data.get("message", "")

        operation = params.get("operation", "")

        # Journey average: specialized handler for travel/distance queries
        if operation == "journey_avg":
            return self._execute_journey_avg(params)

        expression = params.get("expression", "").strip()
        if expression:
            # Params-driven mode: pass clean expression to engine
            try:
                result = self.engine.solve(expression)
                if result.answer and result.answer != "需要更多信息来解答此问题":
                    response = self._format_response(result)
                    return {
                        "success": True,
                        "response": response,
                        "metadata": {
                            "answer": result.answer,
                            "concept": result.concept,
                            "formula": result.formula,
                            "expression": expression,
                        },
                    }
                # Engine couldn't compute → return partial for LLM fallback
                return {
                    "success": False,
                    "response": f"数学引擎无法直接计算表达式「{expression}」，需要更多信息",
                    "metadata": {
                        "expression": expression,
                        "answer": result.answer,
                        "concept": result.concept,
                    },
                }
            except Exception as e:
                return {
                    "success": False,
                    "response": f"计算出错：{str(e)}",
                    "metadata": {"expression": expression, "error": str(e)},
                }

        # Legacy message mode
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

    def _execute_journey_avg(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute average daily distance for a travel journey.

        Receives parameters from the Planner which may include:
        - total_km:    total distance (required)
        - days:        number of days (from Time Converter, via placeholder resolution)
        - remaining_km: distance remaining at midpoint (optional)
        - expression:  pre-computed expression string (optional)

        Returns a structured result with step-by-step journey analysis.
        """
        try:
            total_km = float(params.get("total_km", 0))
            days_str = params.get("days", "")
            remaining_km = float(params.get("remaining_km", 0))
            expression = params.get("expression", "").strip()

            # Resolve days: could be a concrete number or a placeholder string
            if days_str:
                try:
                    days = int(float(days_str))
                except (ValueError, TypeError):
                    # Still a placeholder string — use the expression approach
                    days = None
            else:
                days = None

            lines = []

            if expression and not days_str:
                # Expression provided but days not resolved: try to compute from expression
                # e.g. "2080 / 6" — extract divisor if it looks like a day count
                m = re.search(r"/\s*(\d+)", expression)
                if m:
                    days = int(m.group(1))

            if total_km <= 0:
                return {
                    "success": False,
                    "response": "无法计算：未检测到有效的总里程数据。",
                    "metadata": {},
                }

            if days and days > 0:
                avg = total_km / days
                lines.append(
                    f"📍 **行程计算结果**\n\n"
                    f"**总里程：** {total_km:.0f} km\n"
                    f"**总天数：** {days} 天\n"
                    f"**平均每天：** {total_km:.0f} ÷ {days} = **{avg:.1f} km/天**"
                )

                if remaining_km > 0:
                    first_leg_km = total_km - remaining_km
                    first_leg_days = days - 1 if days > 1 else 1
                    first_leg_avg = first_leg_km / first_leg_days if first_leg_days > 0 else 0
                    second_leg_avg = remaining_km / 1 if days == 1 else remaining_km / 1

                    lines.append(
                        f"\n**分段分析：**\n"
                        f"- 前半程：行驶 {first_leg_km:.0f} km"
                    )
                    if first_leg_days > 1:
                        lines.append(
                            f"  平均 {first_leg_avg:.1f} km/天"
                        )
                    lines.append(
                        f"- 最后一天：行驶 {remaining_km:.0f} km"
                    )

                    if first_leg_avg > 0 and second_leg_avg > 0:
                        if first_leg_avg > second_leg_avg:
                            lines.append(
                                f"\n结论：前半程走得更快（每天多 {first_leg_avg - second_leg_avg:.1f} km）"
                            )
                        else:
                            lines.append(
                                f"\n结论：最后一天走得更快（每天多 {second_leg_avg - first_leg_avg:.1f} km）"
                            )
            else:
                # No days info: compute with expression only
                if expression:
                    lines.append(
                        f"📍 **行程计算结果**\n\n"
                        f"**总里程：** {total_km:.0f} km\n"
                        f"根据表达式「{expression}」无法确定天数，"
                        f"请提供完整的行程日期信息以便计算平均值。"
                    )
                else:
                    lines.append(
                        f"📍 **行程计算结果**\n\n"
                        f"**总里程：** {total_km:.0f} km\n"
                        f"无法确定行驶天数，无法计算平均值。"
                    )

            response = "\n".join(lines)

            return {
                "success": True,
                "response": response,
                "metadata": {
                    "total_km": total_km,
                    "days": days,
                    "avg_km_per_day": round(avg, 1) if days and days > 0 else None,
                    "remaining_km": remaining_km,
                    "expression": expression,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "response": f"行程计算出错：{str(e)}",
                "metadata": {"params": params},
            }

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
