#!/usr/bin/env python3
"""Example agent: a LangChain tool-calling loop over the MCD MCP server, with a local Ollama model.

    python examples/mcd_agent.py "Is a CPAP device (E0601) covered in Minnesota?"
    python examples/mcd_agent.py                     # interactive: one question per line

The agent launches server/mcd_server.py over stdio, exposes its five tools to the chat model and
loops (model -> tool calls -> tool results -> model) until the model answers without calling a tool.

Configuration: MCD_CHAT_MODEL (default gpt-oss:20b), MCD_NUM_CTX (default 16384), OLLAMA_HOST, and
the server's MCD_DSN / MCD_EMBED_MODEL. When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set,
every run is traced to Langfuse (LANGFUSE_BASE_URL selects the instance); without them nothing is sent.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from mcp import Client, StdioServerParameters

SERVER = Path(__file__).resolve().parents[1] / "server" / "mcd_server.py"
CHAT_MODEL = os.environ.get("MCD_CHAT_MODEL", "gpt-oss:20b")
NUM_CTX = int(os.environ.get("MCD_NUM_CTX", "16384"))   # Ollama's default window is too small for policy text
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MAX_STEPS = 25                                         # graph steps (model and tool turns) before giving up
SERVER_ENV = ("MCD_DSN", "MCD_EMBED_MODEL", "OLLAMA_HOST")  # stdio servers do not inherit the environment

SYSTEM_PROMPT = """\
You answer questions about Medicare coverage and coding policy using the tools, never from memory.

{instructions}

Rules:
- Pass the state to the tools whenever the question names one; do not guess a jurisdiction.
- Base every statement on tool results. If the tools return nothing relevant, say so.
- Cite the public_id of each policy you rely on and say where it applies.
- Keep the answer short: the conclusion first, then the criteria or codes that support it.
- This is research output, not billing or coverage advice."""


def langfuse_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def as_langchain_tool(client: Client, tool) -> StructuredTool:
    """One MCP tool -> a LangChain tool with the same name, description and JSON schema."""
    async def call(**kwargs) -> str:
        result = await client.call_tool(tool.name, kwargs)
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        return f"ERROR: {text}" if result.is_error else text  # errors go back to the model so it can retry

    return StructuredTool.from_function(coroutine=call, name=tool.name, description=tool.description or "",
                                        args_schema=tool.input_schema)


@asynccontextmanager
async def mcd_agent(model: str = CHAT_MODEL):
    """Start the MCP server and yield an agent bound to its tools."""
    env = {k: os.environ[k] for k in SERVER_ENV if k in os.environ}
    async with Client(StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=env)) as client:
        tools = [as_langchain_tool(client, t) for t in (await client.list_tools()).tools]
        llm = ChatOllama(model=model, base_url=OLLAMA_HOST, temperature=0, num_ctx=NUM_CTX)
        yield create_agent(llm, tools, system_prompt=SYSTEM_PROMPT.format(instructions=client.instructions or ""))


async def ask(agent, question: str) -> dict:
    """Run the loop for one question. Returns the answer and what the loop did to get there."""
    config = {"recursion_limit": MAX_STEPS, "run_name": "mcd_agent"}
    if langfuse_enabled():
        from langfuse.langchain import CallbackHandler
        config["callbacks"] = [CallbackHandler()]
    started = time.perf_counter()
    state = await agent.ainvoke({"messages": [{"role": "user", "content": question}]}, config)
    messages = state["messages"]
    ai = [m for m in messages if isinstance(m, AIMessage)]
    usage = [m.usage_metadata or {} for m in ai]
    return {"answer": messages[-1].text,
            "tool_calls": [{"name": c["name"], "args": c["args"]} for m in ai for c in m.tool_calls],
            "tool_results": [m.text for m in messages if isinstance(m, ToolMessage)],
            "model_turns": len(ai),
            "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
            "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
            "latency_s": round(time.perf_counter() - started, 2)}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question", nargs="?", help="omit to read questions from stdin, one per line")
    ap.add_argument("--model", default=CHAT_MODEL, help=f"Ollama chat model with tool support (default {CHAT_MODEL})")
    args = ap.parse_args()

    async with mcd_agent(args.model) as agent:
        questions = [args.question] if args.question else (line.strip() for line in sys.stdin)
        for q in filter(None, questions):
            out = await ask(agent, q)
            for c in out["tool_calls"]:
                print(f"  -> {c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})", file=sys.stderr)
            print(f"  [{out['model_turns']} model turns, {out['input_tokens']}+{out['output_tokens']} tokens, "
                  f"{out['latency_s']}s]", file=sys.stderr)
            print(out["answer"], flush=True)
    if langfuse_enabled():
        from langfuse import get_client
        get_client().flush()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
