"""Run flow_sweep.FLOWS for one or more named intents only - for iterating on
a single flow's exemplar without paying for all twenty on every edit.

    LLM_TEMPERATURE=0 python -m backend.scripts.flow_probe appointment.confirm lab.book
"""
from __future__ import annotations

import asyncio
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import AgentClause, ConversationManager
from backend.llm import LlmClient
from backend.scripts.flow_sweep import FLOWS
from backend.settings import ConversationSettings, LlmSettings


async def main(intents: list[str]) -> None:
    convo = ConversationManager(ConversationSettings())
    convo.load()
    llm = LlmClient(LlmSettings())
    llm.load()

    for intent in intents:
        turns = FLOWS[intent]
        print("=" * 72)
        print(intent)
        print("=" * 72)
        cid = intent
        print(f"AGENT   {convo.start_call(cid, agent_name='Gayathri')}\n")
        for text in turns:
            print(f"CALLER  {text}")
            spoken = []
            async for event in convo.stream_utterance(cid, llm, text):
                if isinstance(event, AgentClause):
                    spoken.append(event.text)
            print(f"AGENT   {' '.join(spoken)}\n")
        convo.end_call(cid)


asyncio.run(main(sys.argv[1:]))
