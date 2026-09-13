# Inventor / polarity binding — before chat_facts_v8

Status: **plan + live audit on `data/chat_facts_v7.jsonl` (2026-09-13).**
Do **not** fix with more v7 steps, mix rebuilds for aliases, or fuzzy/embedding cabinet match.

## Problem

Shared templates (`Who invented X?`, `Who is credited with inventing X?`,
`What did Y invent?`) teach a **slot** `X is credited with inventing Y`. The net
fills X/Y from neighborhood pressure. Same failure mode as force↔disable:
lookup can be perfect (aliases / exact key) while **generate** still swaps the
bound answer.

Aliases fix the **wrong question**. Binding on the **exact trained key** is a
data + capacity problem for a **fresh** train.

## Live audit (v7)

- Cabinet unique ≈ **689**; repeats almost flat **20×** (exception: `Where is Paris?` **40×**).
- Inventor-family counts: who_invented **25**, credited **25**, what_did_invent **21**.
- Multi-inventor per invention: **0**. who↔what-did mismatches: **0**.
- Every inventor assistant uses the same shape: `… is credited with inventing …`.
- Noisy WDQS still present (e.g. Spirit Halloween / Cotton Candy Dan, homeoscope, FERMIAC, …).
- **Plough / Schiaparelli** not in current `chat_facts_v7.jsonl` (older probe / overlay).
- Force vs disable: answers in data are correct; question token Jaccard **~0.71** (antonym twins).

## Do next (v8 prep)

1. **Inventor pack hygiene** — keep ~40–80 sober inventions; drop meme/noisy rows; one person ↔ one invention; keep who + credited + what-did only when consistent.
2. **Polarity pairs** — either (a) rewrite force/disable answers to be lexically farther apart, or (b) **hold them out of the cabinet** and serve via an exact router map (recommended for antonym twins at ~55M).
3. **Flatten** `Where is Paris?` to **20×** like every other key.
4. Rebuild mix → new BPE → **`chat_facts_v8` from scratch**. No `--resume`.
5. Probe battery: France capital, Where is Paris, Cubitt, one hard inventor, force, disable, sequential layer streaming.

## Do not

- More v7 steps
- Extra paraphrases hoping the next wording sticks
- Substring / embedding cabinet match
- `--resume` v7 into v8

## Probe log (fill when retesting)

| Prompt | Expect | Result |
|---|---|---|
| `User: How do I disable layer streaming? Assistant:` | `--no-layer-stream` | |
| `User: How do I force layer streaming? Assistant:` | `--layer-stream` | |
| `User: What did William Cubitt invent? Assistant:` | penal treadmill | |
| `User: Who invented Plough? Assistant:` | (only if kept in v8 pack) | |
