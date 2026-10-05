#!/usr/bin/env python3
"""Evaluate examples/mcd_agent.py on eval/mcd_eval_set.jsonl as a Langfuse experiment.

    python eval/mcd_eval.py                      # every item
    python eval/mcd_eval.py --limit 3            # the first three
    python eval/mcd_eval.py --only mcd-007       # chosen items
    python eval/mcd_eval.py --no-judge           # deterministic scores only (no judge model calls)

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
  correctness          judge model: the answer agrees with the reference answer
  faithfulness         judge model: every claim in the answer is supported by the tool results
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

from langchain_ollama import ChatOllama
from langfuse import Evaluation, get_client
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from mcd_agent import CHAT_MODEL, NUM_CTX, OLLAMA_HOST, ask, langfuse_enabled, mcd_agent  # noqa: E402

EVAL_SET = Path(__file__).with_name("mcd_eval_set.jsonl")
JUDGE_MODEL = os.environ.get("MCD_JUDGE_MODEL", CHAT_MODEL)
PUBLIC_ID = re.compile(r"\b([LA]\d{5})\b|\bNCD\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
JUDGE_EVIDENCE_CHARS = 24000   # tool results shown to the judge, cut to fit its context window

JUDGE_PROMPT = """\
You grade an assistant that answers Medicare coverage policy questions from retrieved policy documents.

Question:
{question}

Reference answer:
{reference}

Tool results the assistant retrieved:
{evidence}

Assistant answer:
{answer}

Grade two things independently.
correct: the assistant answer agrees with the reference answer on every fact the reference states \
(policy ids, yes/no, numbers). Extra detail is fine. A contradiction, a missing fact or a refusal is not correct.
faithful: every factual claim in the assistant answer is supported by the tool results. A claim that \
appears nowhere in the tool results makes it unfaithful, even if it happens to be true."""


class Verdict(BaseModel):
    correct: bool
    correct_reason: str = Field(description="one sentence")
    faithful: bool
    faithful_reason: str = Field(description="one sentence")


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
    out = [Evaluation(name="completed", value=float("error" not in output), comment=output.get("error"))]
    if "error" not in output:
        out += [Evaluation(name="latency_s", value=output["latency_s"]),
                Evaluation(name="tool_calls", value=len(output["tool_calls"])),
                Evaluation(name="total_tokens", value=output["input_tokens"] + output["output_tokens"])]
    return out


def make_judge(model: str):
    llm = ChatOllama(model=model, base_url=OLLAMA_HOST, temperature=0, num_ctx=NUM_CTX).with_structured_output(Verdict)

    async def judge(*, input, output, expected_output, metadata, **kwargs) -> list[Evaluation]:
        if not output["answer"]:
            return [Evaluation(name="correctness", value=0.0, comment="no answer"),
                    Evaluation(name="faithfulness", value=0.0, comment="no answer")]
        evidence = "\n---\n".join(output["tool_results"])[:JUDGE_EVIDENCE_CHARS] or "(no tool was called)"
        v = await llm.ainvoke(JUDGE_PROMPT.format(question=input["question"], reference=expected_output["answer"],
                                                  evidence=evidence, answer=output["answer"]))
        return [Evaluation(name="correctness", value=float(v.correct), comment=v.correct_reason),
                Evaluation(name="faithfulness", value=float(v.faithful), comment=v.faithful_reason)]
    return judge


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
    ap.add_argument("--judge-model", default=JUDGE_MODEL, help=f"Ollama model that grades answers (default {JUDGE_MODEL})")
    ap.add_argument("--no-judge", action="store_true", help="skip correctness / faithfulness")
    ap.add_argument("--dataset", default="mcd-coverage-qa", help="Langfuse dataset name")
    ap.add_argument("--run-name", help="Langfuse run name (default: experiment name + timestamp)")
    ap.add_argument("--only", nargs="+", metavar="ID", help="item ids to run")
    ap.add_argument("--limit", type=int, help="run the first N items")
    ap.add_argument("--concurrency", type=int, default=1, help="items in parallel (a local model serves one at a time)")
    ap.add_argument("--show-items", action="store_true", help="print every item's answer and scores")
    args = ap.parse_args()

    items = load_items(args.only, args.limit)
    evaluators = [citations, tool_use, performance] + ([] if args.no_judge else [make_judge(args.judge_model)])
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
