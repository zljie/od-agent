"""Customer service agent implementation using AgentScope ReAct Agent."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agentscope.agent import Agent, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from dotenv import load_dotenv

from .intent import (
    IntentBinding,
    IntentBindingTable,
    IntentClassifierConfig,
    IntentRule,
    RuleBasedIntentClassifier,
    Strategy,
    extract_range_diff_entities,
    extract_math_entities,
    extract_day_of_week_entities,
)
from .models import get_model_config
from .planner import RuleBasedPlanner
from .skills import get_skill_manager, reload_skill_manager

load_dotenv()

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a professional customer service representative for a technology company.
Your name is OD Assistant.

Guidelines:
1. Be friendly, patient, and professional in all interactions
2. Listen carefully to customer concerns and questions
3. Provide accurate and helpful information
4. If you don't know something, admit it honestly and offer to find out
5. Suggest relevant products or services when appropriate
6. Handle complaints with empathy and work toward resolution
7. Keep responses concise but comprehensive
8. Always end with a question to continue the conversation if appropriate

Current product offerings:
- Enterprise AI Solutions
- Cloud Integration Services
- Custom Development
- Technical Support Plans
"""


def get_config_path() -> Path:
    """Get the path to the agent configuration file."""
    return Path(__file__).parent.parent / "config" / "agent_config.json"


def load_agent_config() -> Dict[str, Any]:
    """Load agent configuration from JSON file."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_agent_config(config: Dict[str, Any]) -> None:
    """Save agent configuration to JSON file."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


class CustomerServiceAgent:
    """Customer service dialogue agent powered by AgentScope Agent."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        agent_name: str = "OD_Assistant",
    ):
        # Load from config file if not provided
        config = load_agent_config()
        
        self.agent_name = agent_name or config.get("agent_name", "OD_Assistant")
        self.system_prompt = system_prompt or config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        
        # Get model config
        if model_config is None:
            model_cfg = config.get("model_config", {})
            mc = get_model_config()
            mc.provider_id = model_cfg.get("provider_id", mc.provider_id)
            mc.model_name = model_cfg.get("model_name", mc.model_name)
            mc.base_url = model_cfg.get("base_url", mc.base_url)
            mc.temperature = model_cfg.get("temperature", mc.temperature)
            mc.max_tokens = model_cfg.get("max_tokens", mc.max_tokens)
            mc.top_p = model_cfg.get("top_p", mc.top_p)
            mc.top_k = model_cfg.get("top_k", mc.top_k)
            mc.presence_penalty = model_cfg.get("presence_penalty", mc.presence_penalty)
            mc.frequency_penalty = model_cfg.get("frequency_penalty", mc.frequency_penalty)
            mc.seed = model_cfg.get("seed", mc.seed)
            mc.thinking = model_cfg.get("thinking", mc.thinking)
            mc.thinking_budget = model_cfg.get("thinking_budget", mc.thinking_budget)
            model_config = mc

        # Use provider-aware config to build model kwargs
        model_kwargs = model_config.to_dict()

        # Initialize OpenAI-compatible model (v2.0 API)
        credential = OpenAICredential(
            api_key=model_config.api_key,
            base_url=model_kwargs.get("base_url"),
        )
        params = OpenAIChatModel.Parameters(
            temperature=model_kwargs.get("temperature"),
            max_tokens=model_kwargs.get("max_tokens"),
            top_p=model_kwargs.get("top_p"),
            thinking_enable=model_kwargs.get("thinking_enable", False),
            reasoning_effort=model_kwargs.get("reasoning_effort"),
        )
        self.model = OpenAIChatModel(
            credential=credential,
            model=model_config.model_name,
            parameters=params,
            stream=True,
        )

        # Initialize AgentScope Agent (v2.0 API)
        self.agent = Agent(
            name=self.agent_name,
            system_prompt=self.system_prompt,
            model=self.model,
            react_config=ReActConfig(),
        )

        # Initialize skill manager
        self._skill_manager = get_skill_manager()
        self._active_skill: Optional[str] = None
        self._load_intent_rules()

        # Wire up the full Intent → Plan → Execute pipeline
        self._setup_intent_pipeline()

    def _load_intent_rules(self) -> None:
        """Load intent routing rules from config."""
        intent_config_path = Path(__file__).parent.parent / "config" / "intent_routing.json"
        if intent_config_path.exists():
            with open(intent_config_path, "r", encoding="utf-8") as f:
                rules = json.load(f)
                self._skill_manager.load_intent_rules(rules)

    def _setup_intent_pipeline(self) -> None:
        """Build and wire the Intent → Plan → Execute pipeline."""
        from .intent import IntentBindingTable, IntentClassifierConfig, IntentRule, Strategy
        from .temporal import TemporalParser

        # ── Phase 0: Temporal Parser ─────────────────────────────────────────────
        temporal_parser = TemporalParser()

        # ── Intent rules ──────────────────────────────────────────────────────
        # Keywords are intentionally tight to avoid false positives.
        # Entity extraction must SUCCEED for a rule to fire (checked in classify).
        intent_rules = [
            # Date range: "几天" is the most reliable trigger.
            # The entity extractor validates both ends are real dates.
            IntentRule(
                intent_type="date_range_diff",
                description="计算两个日期之间相差多少天",
                keywords=[
                    "几天", "共几天", "相差几天", "相距几天",
                    "隔几天", "相隔几天",
                ],
                entity_extractor=extract_range_diff_entities,
            ),
            IntentRule(
                intent_type="day_of_week",
                description="查询某个日期是星期几",
                keywords=["星期几", "周几", "礼拜几", "哪天", "今日星期几"],
                entity_extractor=extract_day_of_week_entities,
            ),
            IntentRule(
                intent_type="timezone",
                description="时区转换",
                keywords=["北京时间", "东京时间", "纽约时间", "伦敦时间", "UTC", "GMT", "时区", "时差"],
                entity_extractor=None,
            ),
            # Math: needs numbers + math operators/keywords
            # NOTE: do NOT include "共" or standalone "-" here — they conflict with date strings
            IntentRule(
                intent_type="math",
                description="数学计算（包含数字和运算符）",
                keywords=[
                    "计算", "等于", "加减乘除", "方程", "解",
                    "平均", "每天", "多远", "除以",
                    "+", "*", "/", "×", "÷",
                    "km", "km/", "公里", "元",
                ],
                entity_extractor=extract_math_entities,
            ),
            # Semantic query: natural language → semantic index search
            IntentRule(
                intent_type="semantic_query",
                description="语义查询：自然语言在业务本体中查找数据",
                keywords=[
                    "查一下", "查询", "看看", "获取", "获取数据",
                    "供应商", "采购", "订单", "物料", "客户",
                    "情况", "状态", "金额", "数量",
                    "最近", "三个月", "统计", "汇总",
                    "blocked", "active",
                ],
                entity_extractor=None,
            ),
        ]

        classifier_config = IntentClassifierConfig(rules=intent_rules)
        classifier = RuleBasedIntentClassifier(classifier_config)

        # ── Binding table ─────────────────────────────────────────────────────
        binding_table = IntentBindingTable(bindings=[
            IntentBinding.fixed_skill(
                "date_range_diff",
                "Time Converter",
                required_slots=["start", "end"],
                confidence_floor=0.3,
            ),
            IntentBinding.fixed_skill(
                "day_of_week",
                "Time Converter",
                required_slots=["date"],
                confidence_floor=0.3,
            ),
            IntentBinding.fixed_skill(
                "timezone",
                "Time Converter",
                required_slots=["tz"],
                confidence_floor=0.3,
            ),
            IntentBinding.fixed_skill(
                "math",
                "Math Teacher",
                required_slots=["expression"],
                confidence_floor=0.3,
            ),
            # Semantic query: NL → GraphQL via semantic index
            IntentBinding.fixed_skill(
                "semantic_query",
                "Semantic Query",
                required_slots=[],
                confidence_floor=0.3,
            ),
            # Default: delegate to LLM for unknown intents
            IntentBinding.llm_free("UNKNOWN"),
        ])

        # ── Planner ───────────────────────────────────────────────────────────
        planner = RuleBasedPlanner(binding_table)

        # ── Wire pipeline into SkillManager ────────────────────────────────────
        self._skill_manager.setup_pipeline(
            classifier, binding_table, planner, temporal_parser
        )

    async def reset_history(self) -> None:
        """Clear conversation history."""
        self.agent.state.context.clear()
        self._active_skill = None

    async def chat(self, user_input: str) -> str:
        """Process user input and return agent response."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{ts} [CHAT] (non-streaming) user_input={user_input}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [CHAT] model={self.model.model}, thinking_enable={self.model.parameters.thinking_enable}, reasoning_effort={self.model.parameters.reasoning_effort}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [CHAT] system_prompt={self.system_prompt[:100]}...")

        # Run the full four-phase pipeline: Temporal → Intent → Plan → Execute
        pipeline_result = await self._skill_manager.run_pipeline(user_input)
        decision = pipeline_result.get("decision", "")
        raw_response = pipeline_result.get("response", "")
        temporal = pipeline_result.get("temporal")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [CHAT] pipeline decision={decision}")

        # Non-execution decisions: respond directly
        if decision in ("reject", "clarify", "hitl_confirm", "slot_missing"):
            return raw_response

        # If full pipeline delegates to LLM (no skill matched), try simple fallback
        # detection first — it uses keyword-based matching which is more permissive
        if decision == "delegate_llm":
            simple_result = await self._skill_manager._detect_and_execute_simple(user_input, self)
            if simple_result and simple_result.get("executed"):
                return simple_result["result"].get("response", raw_response)
            return await self._llm_chat(user_input)

        # Check if response contains a DELEGATE_LLM marker (skill failed → LLM fallback)
        if raw_response.startswith("__DELEGATE_LLM__"):
            skill_context = raw_response[len("__DELEGATE_LLM__") :].strip()
            return await self._llm_chat(user_input, skill_context=skill_context)

        # EXECUTE: skill returned structured result
        return raw_response

    async def _llm_chat(self, user_input: str, skill_context: Optional[str] = None) -> str:
        """Delegate to the ReAct agent, optionally with skill result context."""
        # Build skill catalog so the LLM always knows what capabilities exist
        skill_catalog = self._skill_manager.get_skills_summary()
        skill_lines = []
        for s in skill_catalog:
            params = getattr(self._skill_manager.get_skill(s["name"]), "intent_params", {})
            param_str = ", ".join(params.keys()) if params else "无"
            skill_lines.append(
                f"- [{s['name']}] {s['description']} (参数: {param_str})"
            )
        skill_catalog_str = "\n".join(skill_lines) if skill_lines else "（无可用技能）"

        # Build temporal context for messages with multiple date anchors
        temporal = self._skill_manager.build_temporal_context(user_input)
        temporal_context_str = temporal.get("context_text", "") if temporal.get("has_multiple_anchors") else ""

        # Build enhanced user input with skill context
        content = user_input
        if skill_context:
            content = (
                f"用户原始问题：{user_input}\n\n"
                f"技能执行结果（供参考）：\n{skill_context}\n\n"
                f"请基于以上信息回答用户问题。"
            )
        else:
            # Always expose skill catalog + temporal context so the LLM knows what's available
            parts = [
                f"{user_input}\n\n",
                f"[系统技能辅助信息]\n",
                f"以下技能已注册可用：\n{skill_catalog_str}\n\n",
            ]
            if temporal_context_str:
                parts.append(f"{temporal_context_str}\n")
            parts.append(
                "如果用户问题可以用以上技能解决，请直接使用技能结果回答；"
                "如果技能列表中没有相关技能，再使用你的知识回答。"
            )
            content = "".join(parts)

        # v2.0: use UserMsg factory and Agent.reply()
        msg = UserMsg(name="user", content=content)
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{ts} [LLM CHAT] (non-streaming) calling DeepSeek model={self.model.model}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [LLM CHAT] thinking_enable={self.model.parameters.thinking_enable}, reasoning_effort={self.model.parameters.reasoning_effort}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [LLM CHAT] system_prompt={self.system_prompt[:100]}...")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [LLM CHAT] user_content={content[:300]}...")
        try:
            response = await self.agent.reply(msg)
            # v2.0: response.content is a list of ContentBlock; use get_text_content()
            text = response.get_text_content(separator="\n")
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{ts} [LLM CHAT] response={text[:500] if text else '(empty)'}...")
            return text if text else ""
        except Exception as e:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{ts} [LLM CHAT] error: {e}")
            return f"Error calling DeepSeek API: {str(e)}"

    async def chat_stream(self, user_input: str):
        """Streaming version of chat: yields SSE event dicts per docs/SSE流式响应规范.md.

        Single pass through Phase 0-2 (Temporal → Intent → Plan) via
        run_pipeline_stream(), then streams the result based on decision:
        - reject/clarify/slot_missing: content + done
        - delegate_llm: LLM stream via _llm_chat_stream()
        - EXECUTE: tool_call / tool_result / content events from TaskExecutor
        """
        from .sse_stream import (
            done,
        )

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{ts} [CHAT STREAM] user_input={user_input}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [CHAT STREAM] model={self.model.model}, thinking_enable={self.model.parameters.thinking_enable}, reasoning_effort={self.model.parameters.reasoning_effort}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [CHAT STREAM] system_prompt={self.system_prompt[:100]}...")

        async for event in self._skill_manager.run_pipeline_stream(user_input):
            ev_type = event.get("event", "")

            # Private routing events — not forwarded to the SSE client
            if ev_type in ("_decision", "_delegate_llm"):
                if ev_type == "_delegate_llm":
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"{ts} [CHAT STREAM] delegating to LLM (DeepSeek streaming)")
                    async for chunk in self._llm_chat_stream(user_input):
                        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"{ts} [LLM STREAM CHUNK] event={chunk.get('event')}, content={chunk.get('content', '')[:200]}")
                        yield chunk
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"{ts} [LLM STREAM] finished")
                    yield done()
                continue

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{ts} [CHAT STREAM YIELD] event={ev_type}, content={event.get('content', event.get('thinking', ''))[:200]}")
            yield event

    async def _llm_chat_stream(self, user_input: str, skill_context: Optional[str] = None):
        """Streaming LLM chat: yields SSE events from the model response.

        v2.0 API: Agent.reply_stream() yields AgentEvent chunks + final Msg.
        We emit think + think_done for ThinkingBlock events, content for TextBlock.
        """
        from .sse_stream import (
            think,
            think_done,
            content,
        )

        skill_catalog = self._skill_manager.get_skills_summary()
        skill_lines = []
        for s in skill_catalog:
            params = getattr(self._skill_manager.get_skill(s["name"]), "intent_params", {})
            param_str = ", ".join(params.keys()) if params else "无"
            skill_lines.append(f"- [{s['name']}] {s['description']} (参数: {param_str})")
        skill_catalog_str = "\n".join(skill_lines) if skill_lines else "（无可用技能）"

        temporal = self._skill_manager.build_temporal_context(user_input)
        temporal_context_str = temporal.get("context_text", "") if temporal.get("has_multiple_anchors") else ""

        user_content = user_input
        if skill_context:
            user_content = (
                f"用户原始问题：{user_input}\n\n"
                f"技能执行结果（供参考）：\n{skill_context}\n\n"
                f"请基于以上信息回答用户问题。"
            )
        else:
            parts = [
                f"{user_input}\n\n",
                f"[系统技能辅助信息]\n",
                f"以下技能已注册可用：\n{skill_catalog_str}\n\n",
            ]
            if temporal_context_str:
                parts.append(f"{temporal_context_str}\n")
            parts.append(
                "如果用户问题可以用以上技能解决，请直接使用技能结果回答；"
                "如果技能列表中没有相关技能，再使用你的知识回答。"
            )
            user_content = "".join(parts)

        msg = UserMsg(name="user", content=user_content)
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{ts} [LLM CHAT STREAM] calling DeepSeek model={self.model.model}, thinking_enable={self.model.parameters.thinking_enable}, reasoning_effort={self.model.parameters.reasoning_effort}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [LLM CHAT STREAM] system_prompt={self.system_prompt[:100]}...")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [LLM CHAT STREAM] user_content={user_content[:300]}...")
        try:
            full_think_parts: list[str] = []
            full_text_parts: list[str] = []

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{ts} [LLM STREAM] starting reply_stream, model={self.model.model}")
            iter_count = 0
            async for event in self.agent.reply_stream(msg):
                iter_count += 1
                # v2.0: AgentEvent objects are yielded during streaming
                from agentscope.message import ThinkingBlock, TextBlock
                event_type = type(event).__name__
                if isinstance(event, ThinkingBlock):
                    full_think_parts.append(event.thinking)
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"{ts} [LLM STREAM RAW] ThinkingBlock: {event.thinking[:100]}...")
                elif isinstance(event, TextBlock):
                    full_text_parts.append(event.text)
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"{ts} [LLM STREAM RAW] TextBlock: {event.text[:100]}...")
                else:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"{ts} [LLM STREAM RAW] unknown event type={event_type}, repr={repr(event)[:200]}")

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{ts} [LLM STREAM] reply_stream iteration count={iter_count}")

            # The final Msg is auto-consumed by reply_stream - use state.context for final text
            # last_msg.content may contain both ThinkingBlock and TextBlock — extract both
            if self.agent.state.context:
                last_msg = self.agent.state.context[-1]
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"{ts} [LLM STREAM] last_msg content blocks: {len(last_msg.content)} blocks")
                for idx, block in enumerate(last_msg.content):
                    block_type = type(block).__name__
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"{ts} [LLM STREAM] block[{idx}] type={block_type}")
                    if hasattr(block, "thinking"):
                        full_think_parts.append(block.thinking)
                        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"{ts} [LLM STREAM] block[{idx}] thinking: {block.thinking[:100]}...")
                    elif hasattr(block, "text"):
                        full_text_parts.append(block.text)
                        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"{ts} [LLM STREAM] block[{idx}] text: {block.text[:100]}...")
                    else:
                        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"{ts} [LLM STREAM] block[{idx}] unknown: {repr(block)[:200]}")
                # Also get plain text as fallback
                last_text = last_msg.get_text_content(separator="\n")
                if last_text:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"{ts} [LLM STREAM] get_text_content fallback: {last_text[:100]}...")
            else:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"{ts} [LLM STREAM] agent.state.context is empty")

            combined_think = "".join(full_think_parts)
            combined_text = "".join(full_text_parts)

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{ts} [LLM CHAT STREAM] combined_think length={len(combined_think)}, combined_text length={len(combined_text)}")
            if combined_think:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"{ts} [LLM CHAT STREAM] yielding think: {combined_think[:500]}...")
                yield think(combined_think)
                yield think_done()

            if combined_text:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"{ts} [LLM CHAT STREAM] yielding content: {combined_text[:500]}...")
                yield content(combined_text)

        except Exception as e:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{ts} [LLM CHAT STREAM] error: {e}")
            yield content(f"Error calling DeepSeek API: {str(e)}")


# Singleton instance
_agent: Optional[CustomerServiceAgent] = None


def get_agent() -> CustomerServiceAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        config = load_agent_config()
        agent_name = config.get("agent_name", "OD_Assistant")
        _agent = CustomerServiceAgent(agent_name=agent_name)
    return _agent


def reload_agent() -> CustomerServiceAgent:
    """Reload the agent with fresh configuration."""
    global _agent
    _agent = None
    # Also reload skill manager
    reload_skill_manager()
    return get_agent()


def create_agent() -> CustomerServiceAgent:
    """Create a customer service agent instance."""
    return CustomerServiceAgent()
