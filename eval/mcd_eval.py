#!/usr/bin/env python3
"""Evaluate examples/mcd_agent.py on eval/mcd_eval_set.jsonl as a Langfuse experiment.

    python eval/mcd_eval.py                      # every item
    python eval/mcd_eval.py --limit 3            # the first three
    python eval/mcd_eval.py --only mcd-007       # chosen items
    python eval/mcd_eval.py --no-judge           # deterministic scores only (no judge model calls)

Needs eval/requirements.txt (the agent's packages plus ragas).

With LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY set (LANGFUSE_BASE_URL selects the instance), the eval
set is upserted as the Langfuse dataset --dataset and the run is recorded as a dataset run: one trace
per item (model turns, tool calls, tokens, latency) with the scores below attached. Without keys the
same experiment runs locally and only prints its report.

Scores per item
  citation_recall      share of the expected public_ids that the answer cites
  citation_grounded    share of the cited public_ids that appear in a tool result (1 - invented ids)
  retrieval_recall     share of the expected public_ids that any tool result returned
  tool_selection       1 when one of the expected tools was called
  state_filter         1 when the state in the question was passed to a tool (items with a state)
  answered             1 when the agent returned a non-empty answer
  clean_output         1 when the answer has no leaked model markup (chat-template tokens, raw tool calls)
  faithfulness         RAGAS Faithfulness: share of the answer's claims that the tool results support
  factual_correctness  RAGAS FactualCorrectness (recall): share of the reference's claims that the answer supports
  context_recall       RAGAS ContextRecall: share of the reference's claims found in the tool results
  latency_s, tool_calls, total_tokens, completed
Scores per run: averages of the above (reported by Langfuse), latency_p95_s.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Optional

from langfuse import Evaluation, get_client
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, FactualCorrectness, Faithfulness

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from mcd_agent import CHAT_MODEL, OLLAMA_HOST, ask, langfuse_enabled, mcd_agent  # noqa: E402

EVAL_SET = Path(__file__).with_name("mcd_eval_set.jsonl")
JUDGE_MODEL = os.environ.get("MCD_JUDGE_MODEL", "qwen3:8b")
PUBLIC_ID = re.compile(r"\b([LA]\d{5})\b|\bNCD\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
JUDGE_EVIDENCE_CHARS = 24000   # tool results shown to the judge, cut to fit its context window
JUDGE_MAX_TOKENS = 4096        # room for the claim lists the RAGAS prompts ask the judge to write
LEAKED_MARKUP = re.compile(r"<\|[^|>\n]*\|>|【|】|\bto=functions\.")   # chat-template tokens, raw tool calls

def public_ids(text: str) -> set[str]:
    """'see l33718 and NCD240.4.' -> {'L33718', 'NCD 240.4'}"""
    return {m.group(1).upper() if m.group(1) else f"NCD {m.group(2)}" for m in PUBLIC_ID.finditer(text or "")}


def field(item, name: str):
    """Experiment items are dicts for local data and DatasetItem objects for Langfuse datasets."""
    return item.get(name) if isinstance(item, dict) else getattr(item, name)


def load_items(only: Optional[list[str]], limit: Optional[int]) -> list[dict]:
    items = [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()]
    if only:
        missing = set(only) - {i["id"] for i in items}
        if missing:
            sys.exit(f"unknown item id(s): {', '.join(sorted(missing))}")
        items = [i for i in items if i["id"] in only]
    return items[:limit]


# ----------------------------------------------------------------------- task
def make_task(model: str):
    async def task(*, item, **kwargs) -> dict:
        try:
            async with mcd_agent(model) as agent:   # one MCP server per item; its startup is not timed
                return await ask(agent, field(item, "input")["question"])
        except Exception as e:  # a failed item is scored as a miss instead of dropping out of the run
            return {"answer": "", "error": f"{type(e).__name__}: {e}", "tool_calls": [], "tool_results": []}
    return task


# ----------------------------------------------------------------- evaluators
def citations(*, input, output, expected_output, metadata, **kwargs) -> list[Evaluation]:
    expected = set(expected_output["public_ids"])
    cited = public_ids(output["answer"])
    retrieved = public_ids("\n".join(output["tool_results"]))
    out = []
    if expected:
        out.append(Evaluation(name="citation_recall", value=len(expected & cited) / len(expected),
                              comment=f"missing: {sorted(expected - cited) or 'none'}"))
        out.append(Evaluation(name="retrieval_recall", value=len(expected & retrieved) / len(expected),
                              comment=f"not retrieved: {sorted(expected - retrieved) or 'none'}"))
    if cited:
        asked = public_ids(input["question"])   # ids the question itself names are not invented
        out.append(Evaluation(name="citation_grounded", value=len(cited & (retrieved | asked)) / len(cited),
                              comment=f"not in any tool result: {sorted(cited - retrieved - asked) or 'none'}"))
    return out


def tool_use(*, input, output, expected_output, metadata, **kwargs) -> list[Evaluation]:
    calls = output["tool_calls"]
    names = [c["name"] for c in calls]
    out = [Evaluation(name="tool_selection", value=float(bool(set(names) & set(metadata["tools"]))),
                      comment=f"called: {names or 'none'}; expected one of {metadata['tools']}")]
    if metadata.get("state"):
        want = {metadata["state"].lower(), metadata["state_name"].lower()}
        passed = [str(c["args"].get("state") or "").strip().lower() for c in calls]
        out.append(Evaluation(name="state_filter", value=float(any(p in want for p in passed)),
                              comment=f"state arguments: {[p for p in passed if p] or 'none'}"))
    return out


def performance(*, input, output, expected_output, metadata, **kwargs) -> list[Evaluation]:
    answer = output["answer"]
    leaked = sorted({m.group(0) for m in LEAKED_MARKUP.finditer(answer)})
    out = [Evaluation(name="completed", value=float("error" not in output), comment=output.get("error")),
           Evaluation(name="answered", value=float(bool(answer.strip())),
                      comment=None if answer.strip() else "empty answer"),
           Evaluation(name="clean_output", value=float(not leaked), comment=f"leaked markup: {leaked or 'none'}")]
    if "error" not in output:
        out += [Evaluation(name="latency_s", value=output["latency_s"]),
                Evaluation(name="tool_calls", value=len(output["tool_calls"])),
                Evaluation(name="total_tokens", value=output["input_tokens"] + output["output_tokens"])]
    return out


def contexts(output) -> list[str]:
    """Tool results as RAGAS retrieved_contexts, cut to JUDGE_EVIDENCE_CHARS in total."""
    out, room = [], JUDGE_EVIDENCE_CHARS
    for text in output["tool_results"]:
        if room <= 0:
            break
        out.append(text[:room])
        room -= len(text)
    return out or ["(no tool was called)"]


def make_ragas(model: str) -> list:
    """One evaluator per RAGAS metric, so a failed judge call costs one score and not all three."""
    # Ollama's OpenAI-compatible endpoint: it ignores the key and takes its context window from the
    # server (OLLAMA_CONTEXT_LENGTH), not from the request.
    client = AsyncOpenAI(base_url=f"{OLLAMA_HOST.rstrip('/')}/v1", api_key="ollama")
    # reasoning_effort: a thinking model otherwise spends its output budget before it reaches the verdict
    llm = llm_factory(model, provider="openai", client=client, temperature=0, max_tokens=JUDGE_MAX_TOKENS,
                      reasoning_effort="none")
    faithful = Faithfulness(llm=llm)
    correct = FactualCorrectness(llm=llm, mode="recall")   # recall: detail beyond the reference is not penalized
    recall = ContextRecall(llm=llm)

    async def faithfulness(*, input, output, expected_output, metadata, **kwargs) -> Evaluation:
        if not output["answer"]:
            return Evaluation(name="faithfulness", value=0.0, comment="no answer")
        r = await faithful.ascore(user_input=input["question"], response=output["answer"],
                                  retrieved_contexts=contexts(output))
        return Evaluation(name="faithfulness", value=r.value)

    async def factual_correctness(*, input, output, expected_output, metadata, **kwargs) -> Evaluation:
        if not output["answer"]:
            return Evaluation(name="factual_correctness", value=0.0, comment="no answer")
        r = await correct.ascore(response=output["answer"], reference=expected_output["answer"])
        return Evaluation(name="factual_correctness", value=r.value)

    async def context_recall(*, input, output, expected_output, metadata, **kwargs) -> Evaluation:
        r = await recall.ascore(user_input=input["question"], retrieved_contexts=contexts(output),
                                reference=expected_output["answer"])
        return Evaluation(name="context_recall", value=r.value)

    return [faithfulness, factual_correctness, context_recall]


def latency_p95(*, item_results, **kwargs) -> Evaluation:
    times = sorted(r.output["latency_s"] for r in item_results if r.output and "latency_s" in r.output)
    if not times:
        return Evaluation(name="latency_p95_s", value=0.0, comment="no item completed")
    p95 = times[0] if len(times) == 1 else statistics.quantiles(times, n=20, method="inclusive")[-1]
    return Evaluation(name="latency_p95_s", value=round(p95, 2), comment=f"over {len(times)} completed items")


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=CHAT_MODEL, help=f"Ollama chat model under test (default {CHAT_MODEL})")
    ap.add_argument("--judge-model", default=JUDGE_MODEL, help=f"Ollama model behind the RAGAS metrics (default {JUDGE_MODEL})")
    ap.add_argument("--no-judge", action="store_true", help="skip the RAGAS metrics")
    ap.add_argument("--dataset", default="mcd-coverage-qa", help="Langfuse dataset name")
    ap.add_argument("--run-name", help="Langfuse run name (default: experiment name + timestamp)")
    ap.add_argument("--only", nargs="+", metavar="ID", help="item ids to run")
    ap.add_argument("--limit", type=int, help="run the first N items")
    ap.add_argument("--concurrency", type=int, default=1, help="items in parallel (a local model serves one at a time)")
    ap.add_argument("--show-items", action="store_true", help="print every item's answer and scores")
    args = ap.parse_args()

    items = load_items(args.only, args.limit)
    evaluators = [citations, tool_use, performance] + ([] if args.no_judge else make_ragas(args.judge_model))
    run = dict(name=f"mcd-agent/{args.model}", run_name=args.run_name, task=make_task(args.model),
               description="examples/mcd_agent.py on eval/mcd_eval_set.jsonl",
               evaluators=evaluators, run_evaluators=[latency_p95], max_concurrency=args.concurrency,
               metadata={"model": args.model, "judge_model": "none" if args.no_judge else args.judge_model})

    langfuse = get_client()
    if langfuse_enabled():
        if not langfuse.auth_check():
            sys.exit("Langfuse rejected the credentials; check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL")
        langfuse.create_dataset(name=args.dataset, description="MCD coverage questions with reference answers and "
                                "expected public_ids (eval/mcd_eval_set.jsonl)")
        for i in items:  # item ids make this an upsert, so edits to the file reach Langfuse
            langfuse.create_dataset_item(dataset_name=args.dataset, id=i["id"], input=i["input"],
                                         expected_output=i["expected_output"], metadata=i["metadata"])
        dataset = langfuse.get_dataset(args.dataset)
        ids = {i["id"] for i in items}
        dataset.items = [d for d in dataset.items if d.id in ids]   # honor --only / --limit
        result = dataset.run_experiment(**run)
    else:
        print("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set: running locally, nothing is recorded.", file=sys.stderr)
        result = langfuse.run_experiment(data=items, **run)

    print(result.format(include_item_results=args.show_items))
    langfuse.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
