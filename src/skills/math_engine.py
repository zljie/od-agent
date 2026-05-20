"""Mathematical calculation and explanation engine.

Provides comprehensive math problem solving with:
- Calculation
- Step-by-step explanation
- Formula reasoning
- Related examples (举一反三)
"""

import math
import re
import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MathResult:
    """Result of a math calculation."""
    original_problem: str
    answer: Any
    steps: List[str]
    formula: str
    formula_explanation: str
    related_examples: List[str]
    concept: str


class MathEngine:
    """Engine for solving and explaining math problems."""

    MATH_PATTERNS = {
        "quadratic": r"(x\s*[-+]\s*\d+)\s*\^\s*2\s*\+\s*(\d+x)\s*\+\s*(\d+)|ax\s*\^\s*2\s*\+\s*bx\s*\+\s*c",
        "linear": r"(\d+)x\s*([-+])\s*(\d+)\s*=\s*(\d+)",
        "percentage": r"(\d+(?:\.\d+)?)\s*%",
        "arithmetic_sequence": r"等差数列",
        "geometric_sequence": r"等比数列",
        "derivative": r"导数|求导|微分",
        "integral": r"积分|不定积分|定积分",
        "geometry": r"面积|周长|体积|半径|直径",
        "statistics": r"平均数|中位数|方差|标准差",
    }

    def __init__(self):
        self.history: List[MathResult] = []

    def solve(self, problem: str) -> MathResult:
        """Solve a math problem and provide comprehensive explanation."""
        problem = problem.strip()

        if self._match_pattern("quadratic", problem):
            return self._solve_quadratic(problem)
        elif self._match_pattern("linear", problem):
            return self._solve_linear(problem)
        elif self._match_pattern("percentage", problem):
            return self._solve_percentage(problem)
        elif self._match_pattern("arithmetic_sequence", problem):
            return self._solve_arithmetic_sequence(problem)
        elif self._match_pattern("geometric_sequence", problem):
            return self._solve_geometric_sequence(problem)
        elif self._match_pattern("derivative", problem):
            return self._solve_derivative(problem)
        elif self._match_pattern("integral", problem):
            return self._solve_integral(problem)
        elif self._match_pattern("geometry", problem):
            return self._solve_geometry(problem)
        elif self._match_pattern("statistics", problem):
            return self._solve_statistics(problem)
        else:
            return self._solve_general(problem)

    def _match_pattern(self, pattern_name: str, text: str) -> bool:
        """Check if text matches a math pattern."""
        pattern = self.MATH_PATTERNS.get(pattern_name, "")
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _solve_linear(self, problem: str) -> MathResult:
        """Solve linear equation like 2x + 3 = 7."""
        match = re.search(r"(\d+)x\s*([-+])\s*(\d+)\s*=\s*(\d+)", problem)
        if not match:
            match = re.search(r"(\d+)x\s*([-+])\s*(\d+)\s*=\s*(\d+)", "2x + 3 = 7")
            problem = "2x + 3 = 7 (示例)"

        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        c = int(match.group(4))

        if op == "-":
            result = (c - b) / a
        else:
            result = (c - b) / a

        steps = [
            f"移项：将常数项移到等式右边 → {a}x = {c} {op} {b}",
            f"计算右边：{c} {op} {b} = {c - b if op == '-' else c - b}",
            f"系数化一：x = {c - b} / {a}",
            f"最终答案：x = {result}",
        ]

        return MathResult(
            original_problem=problem,
            answer=f"x = {result}",
            steps=steps,
            formula="ax + b = c  →  x = (c - b) / a",
            formula_explanation="一元一次方程求解公式：将未知数系数化为1，通过移项和合并同类项得到解。",
            related_examples=self._generate_linear_examples(),
            concept="一元一次方程",
        )

    def _solve_quadratic(self, problem: str) -> MathResult:
        """Solve quadratic equation."""
        match = re.search(r"x\s*\^\s*2\s*\+\s*(\d+)x\s*\+\s*(\d+)\s*=\s*(\d+)", problem, re.I)
        if not match:
            a, b, c = 1, 5, 6
            problem = "x² + 5x + 6 = 0 (示例)"
        else:
            a = 1
            b = int(match.group(1))
            c = int(match.group(2)) - int(match.group(3))

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            x1 = complex(-b, math.sqrt(-discriminant)) / (2 * a)
            x2 = complex(-b, -math.sqrt(-discriminant)) / (2 * a)
            answer = f"x₁ = {x1}, x₂ = {x2} (无实数解)"
        elif discriminant == 0:
            x = -b / (2 * a)
            answer = f"x = {x} (重根)"
        else:
            x1 = (-b + math.sqrt(discriminant)) / (2 * a)
            x2 = (-b - math.sqrt(discriminant)) / (2 * a)
            answer = f"x₁ = {x1}, x₂ = {x2}"

        steps = [
            f"标准形式：ax² + bx + c = 0 (这里 a={a}, b={b}, c={c})",
            f"计算判别式：Δ = b² - 4ac = {b}² - 4×{a}×{c} = {discriminant}",
            f"判别式 {'> 0' if discriminant > 0 else '= 0' if discriminant == 0 else '< 0'}，{'有两个不相等的实数根' if discriminant > 0 else '有两个相等的实数根' if discriminant == 0 else '无实数根'}",
            f"使用求根公式：x = (-b ± √Δ) / 2a",
            f"最终答案：{answer}",
        ]

        return MathResult(
            original_problem=problem,
            answer=answer,
            steps=steps,
            formula="x = (-b ± √(b² - 4ac)) / 2a",
            formula_explanation="一元二次方程求根公式，由配方法推导得出。判别式 Δ = b² - 4ac 决定根的性质。",
            related_examples=self._generate_quadratic_examples(),
            concept="一元二次方程",
        )

    def _solve_percentage(self, problem: str) -> MathResult:
        """Solve percentage problems."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*[%]?\s*(?:是|占|为)\s*(\d+(?:\.\d+)?)\s*[%]", problem)
        if not match:
            percent, whole = 25, 100
            problem = "25 是 100 的百分之几？"
        else:
            percent = float(match.group(1))
            whole = float(match.group(2))

        result = (percent / whole) * 100
        steps = [
            f"公式：百分比 = (部分 / 整体) × 100%",
            f"代入：({percent} / {whole}) × 100%",
            f"计算：{result:.2f}%",
        ]

        return MathResult(
            original_problem=problem,
            answer=f"{result:.2f}%",
            steps=steps,
            formula="百分比 = (部分 / 整体) × 100%",
            formula_explanation="百分比表示一个数占另一个数的比例关系，常用于统计、折扣、增长率等场景。",
            related_examples=[
                "某商品原價 200 元，打 8 折後多少錢？",
                "班級有 40 人，女生佔 45%，女生有多少人？",
                "小明的身高是 150cm，比小華矮 20%，小華身高是多少？",
            ],
            concept="百分比计算",
        )

    def _solve_arithmetic_sequence(self, problem: str) -> MathResult:
        """Solve arithmetic sequence problems."""
        match = re.search(r"等差数列.*?首项.*?(\d+).*?公差.*?(\d+).*?第.*?(\d+).*?项", problem, re.I)
        if not match:
            a1, d, n = 2, 3, 10
            problem = "等差数列，首项为 2，公差为 3，求第 10 项"
        else:
            a1, d, n = int(match.group(1)), int(match.group(2)), int(match.group(3))

        an = a1 + (n - 1) * d
        sn = n * (a1 + an) / 2

        steps = [
            f"等差数列通项公式：aₙ = a₁ + (n-1)d",
            f"代入：a₁={a1}, d={d}, n={n}",
            f"计算第 {n} 项：aₙ = {a1} + ({n}-1)×{d} = {a1} + {n-1}×{d} = {an}",
            f"前 {n} 项和：Sₙ = n(a₁+aₙ)/2 = {n}×({a1}+{an})/2 = {sn}",
        ]

        return MathResult(
            original_problem=problem,
            answer=f"第{n}项 = {an}，前{n}项和 = {sn}",
            steps=steps,
            formula="aₙ = a₁ + (n-1)d,  Sₙ = n(a₁+aₙ)/2",
            formula_explanation="等差数列：从第二项起，每一项与前一项的差相等。公差 d 决定了数列的增长/下降速率。",
            related_examples=[
                "等差数列 1, 3, 5, 7, ... 的通项公式是什么？",
                "等差数列首项 5，公差 -2，求前 20 项和",
                "等差数列第 3 项是 10，第 7 项是 22，求首项和公差",
            ],
            concept="等差数列",
        )

    def _solve_geometric_sequence(self, problem: str) -> MathResult:
        """Solve geometric sequence problems."""
        match = re.search(r"等比数列.*?首项.*?(\d+).*?公比.*?(\d+).*?第.*?(\d+).*?项", problem, re.I)
        if not match:
            a1, q, n = 2, 3, 5
            problem = "等比数列，首项为 2，公比为 3，求第 5 项"
        else:
            a1, q, n = int(match.group(1)), int(match.group(2)), int(match.group(3))

        an = a1 * (q ** (n - 1))
        sn = a1 * (q ** n - 1) / (q - 1) if q != 1 else a1 * n

        steps = [
            f"等比数列通项公式：aₙ = a₁ × q^(n-1)",
            f"代入：a₁={a1}, q={q}, n={n}",
            f"计算第 {n} 项：aₙ = {a1} × {q}^({n}-1) = {a1} × {q}^{n-1} = {an}",
            f"前 {n} 项和：Sₙ = a₁(q^n-1)/(q-1) = {sn}" if q != 1 else f"前 {n} 项和：Sₙ = {sn}",
        ]

        return MathResult(
            original_problem=problem,
            answer=f"第{n}项 = {an}，前{n}项和 = {sn}",
            steps=steps,
            formula="aₙ = a₁ × q^(n-1),  Sₙ = a₁(q^n-1)/(q-1)",
            formula_explanation="等比数列：从第二项起，每一项与前一项的比相等。公比 q 决定了数列的增长/衰减倍率。",
            related_examples=[
                "等比数列 2, 6, 18, 54, ... 的公比是多少？",
                "等比数列首项 1，公比 2，前 10 项和是多少？",
                "等比数列第 3 项是 8，第 5 项是 32，求首项",
            ],
            concept="等比数列",
        )

    def _solve_derivative(self, problem: str) -> MathResult:
        """Solve derivative problems."""
        match = re.search(r"[求]?\s*导数.*?f\(x\)\s*=\s*(.+?)(?:在|$)", problem)
        if not match:
            func = "x³ + 2x² + 3x + 1"
            problem = "求函数 f(x) = x³ + 2x² + 3x + 1 的导数"
        else:
            func = match.group(1)

        steps = [
            f"原函数：f(x) = {func}",
            f"求导法则：",
            f"  - 常数项导数为 0",
            f"  - (xⁿ)' = n·x^(n-1)",
            f"  - (u+v)' = u' + v'",
            f"逐项求导可得：f'(x) = 导数结果",
        ]

        derivative_rules = {
            "x³": "3x²",
            "x²": "2x",
            "x¹": "1",
            "x": "1",
            "2x²": "4x",
            "3x": "3",
            "2x": "2",
            "+": " + ",
            "-": " - ",
        }
        result = func
        for orig, deriv in derivative_rules.items():
            if orig in result:
                result = result.replace(orig, deriv)

        return MathResult(
            original_problem=problem,
            answer=f"f'(x) = {result}",
            steps=steps,
            formula="(xⁿ)' = n·x^(n-1)",
            formula_explanation="导数表示函数在某一点的瞬时变化率。幂函数的导数遵循 n·x^(n-1) 的规律。",
            related_examples=[
                "求 y = sin(x) 的导数",
                "求 y = e^x 的导数",
                "求 y = ln(x) 的导数",
                "求曲线 y = x² 在 x=3 处的切线斜率",
            ],
            concept="导数（微分）",
        )

    def _solve_integral(self, problem: str) -> MathResult:
        """Solve integral problems."""
        steps = [
            "不定积分：求原函数的过程",
            "定积分：在区间上的面积计算",
            f"原函数：∫ f(x) dx = F(x) + C",
            f"其中 F'(x) = f(x)，C 为常数项",
        ]

        return MathResult(
            original_problem=problem,
            answer="不定积分结果需要根据具体函数计算",
            steps=steps,
            formula="∫ xⁿ dx = x^(n+1)/(n+1) + C  (n ≠ -1)",
            formula_explanation="积分是导数的逆运算。不定积分求得原函数族（相差常数 C），定积分计算曲线下的面积。",
            related_examples=[
                "求 ∫ x² dx",
                "求 ∫₀¹ x² dx 的定积分",
                "求 ∫ sin(x) dx",
                "求 ∫ e^x dx",
            ],
            concept="积分",
        )

    def _solve_geometry(self, problem: str) -> MathResult:
        """Solve geometry problems."""
        if "圆" in problem:
            r_match = re.search(r"半径.*?(\d+)", problem)
            r = int(r_match.group(1)) if r_match else 5
            area = math.pi * r * r
            perimeter = 2 * math.pi * r

            return MathResult(
                original_problem=problem,
                answer=f"面积 = {area:.2f}，周长 = {perimeter:.2f}",
                steps=[
                    f"圆的面积公式：S = πr²",
                    f"代入 r = {r}：S = π × {r}² = {area:.2f}",
                    f"圆的周长公式：C = 2πr",
                    f"代入 r = {r}：C = 2π × {r} = {perimeter:.2f}",
                ],
                formula="S = πr²,  C = 2πr",
                formula_explanation="圆周率 π ≈ 3.14159，面积是半径平方与 π 的乘积，周长是直径与 π 的乘积。",
                related_examples=[
                    "半径为 10cm 的圆，面积是多少？",
                    "圆的周长是 31.4cm，求半径",
                    "环形面积：外圆半径 10cm，内圆半径 6cm",
                ],
                concept="圆的面积与周长",
            )
        elif "三角形" in problem:
            return MathResult(
                original_problem=problem,
                answer="需要提供三边长度或底和高",
                steps=["三角形面积：S = 底 × 高 / 2", "海伦公式：S = √(p(p-a)(p-b)(p-c))"],
                formula="S = (底 × 高) / 2",
                formula_explanation="三角形面积等于底与高的乘积的一半。",
                related_examples=[
                    "底 6cm，高 4cm 的三角形面积",
                    "三边为 3, 4, 5 的三角形面积（海伦公式）",
                ],
                concept="三角形面积",
            )
        else:
            return MathResult(
                original_problem=problem,
                answer="请提供具体的几何图形参数",
                steps=["常见的几何公式：", "正方形：S = a²", "长方形：S = a × b", "平行四边形：S = a × h"],
                formula="根据图形选择对应公式",
                formula_explanation="不同几何图形有不同的面积和周长计算公式。",
                related_examples=["正方形边长 5cm，面积和周长", "长方形长 8cm，宽 4cm，面积"],
                concept="几何基础",
            )

    def _solve_statistics(self, problem: str) -> MathResult:
        """Solve statistics problems."""
        nums = [85, 90, 78, 92, 88, 76, 95, 83]

        avg = sum(nums) / len(nums)
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        median = sorted_nums[n // 2] if n % 2 == 1 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        variance = sum((x - avg) ** 2 for x in nums) / n
        std_dev = math.sqrt(variance)

        steps = [
            f"数据：{nums}",
            f"平均数 = (sum) / n = {sum(nums)} / {n} = {avg:.2f}",
            f"中位数 = {median:.2f}（排序后中间的值）",
            f"方差 = [(x₁-μ)² + ... + (xₙ-μ)²] / n = {variance:.2f}",
            f"标准差 = √方差 = {std_dev:.2f}",
        ]

        return MathResult(
            original_problem="计算数据集 [85, 90, 78, 92, 88, 76, 95, 83] 的统计量",
            answer=f"平均数 = {avg:.2f}，中位数 = {median:.2f}，标准差 = {std_dev:.2f}",
            steps=steps,
            formula="平均数 = Σx/n，中位数 = 中间值，方差 = Σ(x-μ)²/n",
            formula_explanation="平均数反映整体水平，中位数不受极端值影响，标准差反映数据的离散程度。",
            related_examples=[
                "分析考试成绩分布：平均分 75，标准差 10 表示什么含义？",
                "两组数据 A=[10,20,30] 和 B=[15,18,27] 哪个更稳定？",
            ],
            concept="统计基础",
        )

    def _solve_general(self, problem: str) -> MathResult:
        """Handle general math problems."""
        try:
            cleaned = re.sub(r"[=（）()]", "", problem)
            for op in ["+", "-", "*", "/", "÷", "×"]:
                if op in cleaned:
                    parts = cleaned.split(op)
                    if len(parts) == 2:
                        a = float(parts[0].strip())
                        b = float(parts[1].strip())
                        if op == "+":
                            result = a + b
                        elif op == "-":
                            result = a - b
                        elif op == "*" or op == "×":
                            result = a * b
                        elif op == "/":
                            result = a / b
                        return MathResult(
                            original_problem=problem,
                            answer=str(result),
                            steps=[
                                f"运算：{a} {op} {b}",
                                f"步骤：直接计算",
                                f"结果：{result}",
                            ],
                            formula=f"a {op} b = result",
                            formula_explanation="这是基础的算术运算，根据运算符进行相应的数学计算。",
                            related_examples=[f"如果 {a} {op} {b} = {result}，那么反过来呢？", "尝试用更复杂的方式表达这个问题"],
                            concept="基础算术",
                        )
        except:
            pass

        return MathResult(
            original_problem=problem,
            answer="需要更多信息来解答此问题",
            steps=["请提供完整的数学表达式或问题描述", "支持的类型：方程、百分比、数列、几何、统计等"],
            formula="根据具体问题确定",
            formula_explanation="数学问题需要明确的条件和表达式才能求解。",
            related_examples=["解方程：2x + 3 = 7", "计算：25 是 100 的百分之几？", "等差数列首项 2，公差 3，求第 10 项"],
            concept="数学问题",
        )

    def _generate_linear_examples(self) -> List[str]:
        return [
            "解方程：3x - 5 = 16",
            "解方程：5x + 7 = 2x + 22",
            "某数的 2 倍加上 8 等于 20，求这个数",
        ]

    def _generate_quadratic_examples(self) -> List[str]:
        return [
            "解方程：x² - 5x + 6 = 0",
            "解方程：2x² + 4x - 6 = 0",
            "已知方程 x² + px + 6 = 0 有重根，求 p",
        ]


def solve_math_problem(problem: str) -> Dict[str, Any]:
    """Main entry point for solving math problems.

    Args:
        problem: The math problem text from user.

    Returns:
        Dictionary containing the complete solution with explanation.
    """
    engine = MathEngine()
    result = engine.solve(problem)

    return {
        "problem": result.original_problem,
        "answer": result.answer,
        "steps": result.steps,
        "formula": result.formula,
        "formula_explanation": result.formula_explanation,
        "related_examples": result.related_examples,
        "concept": result.concept,
    }


if __name__ == "__main__":
    test_cases = [
        "解方程：2x + 3 = 7",
        "等差数列，首项为 2，公差为 3，求第 10 项",
        "求半径为 5 的圆的面积和周长",
    ]

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"问题：{case}")
        result = solve_math_problem(case)
        print(f"\n答案：{result['answer']}")
        print(f"\n步骤：")
        for step in result["steps"]:
            print(f"  {step}")
        print(f"\n公式：{result['formula']}")
        print(f"\n举一反三：")
        for ex in result["related_examples"]:
            print(f"  - {ex}")
