# Analysis-to-Action Pi Audit

Date: 2026-08-13

## Conclusion

ForgeLoop's stable prompt cache is no longer the limiting factor. The saved
SQLfmt canary reached 99.9849% weighted warm-prefix reuse, but V4-Flash returned
reasoning-only `length` responses at calls 13, 15, and 17. The first two bounded
recoveries led to complete actions, including a real source edit at call 16.

The next verified Pi parity gap is the per-response output window. ForgeLoop's
frozen v1.3 policy set `max_tokens=8192`. Pi's current provider base options use
a model-aware default bounded at 32,000 tokens, while DeepSeek's official Pi
integration metadata declares a 384,000-token V4-Flash model ceiling. ForgeLoop
therefore adds a new policy identity with a 32,000-token operational cap. It
does not rewrite the historical v1.3 manifest.

## Evidence from the saved trajectory

The three limited responses were not byte-equivalent continuation loops:

- call 13 designed the DDL parser and data model;
- call 15 designed the formatter/rules integration;
- call 17 inspected problems in the newly created implementation.

Their reasoning payloads were 32,980, 32,608, and 30,279 characters. Calls 14
and 16 returned complete actions, and call 16 created `src/sqlfmt/ddl.py`.
Deleting all recent reasoning history would therefore discard active design
state without evidence that it improves action conversion. ForgeLoop retains
recent reasoning, matching Pi's typed thinking-message history, while its
existing compaction strips stale historical chains.

## Pi source audit

Primary references:

- [Pi agent loop](https://github.com/earendil-works/pi/blob/main/packages/agent/src/agent-loop.ts)
  continues while tool calls or queued messages exist. A length-limited turn
  with no tool call otherwise settles; ForgeLoop's progress-aware continuation
  intentionally covers this known failure mode.
- [Pi provider base options](https://github.com/earendil-works/pi/blob/main/packages/ai/src/providers/simple-options.ts)
  use the caller override or a model-aware default capped at 32,000 tokens.
- [Pi OpenAI-compatible provider](https://github.com/earendil-works/pi/blob/main/packages/ai/src/providers/openai-completions.ts)
  preserves thinking blocks and maps DeepSeek-compatible reasoning history.
- [DeepSeek's official Pi integration](https://github.com/deepseek-ai/awesome-deepseek-agent/blob/main/docs/pi_mono.md)
  declares V4-Flash with a one-million-token context and 384,000 maximum output
  tokens.

The older DeepSWE mini-swe-agent reference used 8,192 explicitly and did solve
tasks, so a larger window is not claimed as sufficient for solve-rate gains.
It is an isolated operational change intended to stop an artificial per-turn
cap from consuming the entire response before an action can be emitted.

## Policy v1.4

`deepseek-v4-flash-controller-v1.4-pi-output-window` changes only the recorded
output-window policy:

- generation `max_tokens`: 8,192 -> 32,000;
- effective selection: `min(384000 model capability, 32000 Pi default cap)`;
- interactive V4 sessions select v1.4 by default.

It preserves the base model/revision, max thinking level, system and
`execution-first-v1` prompts, tool schemas, Hybrid Controller v1.3 Simplified,
Context Efficiency algorithm, provider retry policy, and execution budgets.
The old v1.3 policy remains bundled and loadable for exact historical replay.

## Ordinary-task cache gate

The same three ordinary Stage C tasks were run once under v1.4 before any hard
canary.

| Task | Result | Calls / tools | Input / cached / output | Warm hit | Cost / wall |
|---|---|---:|---:|---:|---:|
| none-handling | PASS | 6 / 8 | 16,483 / 13,568 / 760 | 97.3148% | $0.00065889 / 19.532s |
| duration-parser | PASS | 6 / 8 | 20,559 / 17,536 / 2,136 | 98.4273% | $0.00107040 / 25.317s |
| root-cause-cache | PASS | 6 / 9 | 17,492 / 14,720 / 1,020 | 98.5019% | $0.00071490 / 19.007s |
| **weighted total** | **3/3 PASS** | **18 / 25** | **54,534 / 45,824 / 3,916** | **41,954 / 42,762 = 98.1100%** | **$0.00244419 / 63.856s** |

The configured 98% weighted gate passed with no significant cache-miss call.
The larger output ceiling did not change the six-call completion pattern or
cause ordinary tasks to produce abnormally long responses.

## Hard canary

Pending one SQLfmt run under the new policy after the ordinary gate.
