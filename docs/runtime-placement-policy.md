# Runtime placement policy and its technical debt

Status: **implemented in the repo tree, not deployed**. R1-R3 are all in place; R2 is
opt-in and default-off. This document records what was measured, what changed, and what
is still wrong.

## The constraint that drove everything

A gateway routes to `llama-swap`, which runs one `llama-server` process per route
described in `config.yaml`. The GPU a route uses is baked into that route's command
as a `CUDA_VISIBLE_DEVICES` prefix.

Measured on this host: **any** write to `config.yaml` makes llama-swap (`--watch-config`)
unload **every** model and destroy **every** KV cache. Not a model, not the edited
route - everything, including models with a request in flight.

The old code treated `config.yaml` as runtime state: to move a model to a free GPU
it rewrote the model's `CUDA_VISIBLE_DEVICES` and re-rendered. Each such move cost a
full cold reload of the entire fleet. That is the root cause of the load churn this
policy exists to remove, and it was self-inflicted - llama-swap was not refusing, our
writer was choosing to use the wrong mechanism.

**Invariant: `config.yaml` is written only by the render, never by a request path.**

## The policy

1. **R1** - A free GPU gets the model, even when another model already occupies the
   other card.
2. **R2** - An idle GPU keeps a warm copy of a model.
3. **R3** - A request for a different model takes its VRAM from a disposable route.
   A base model that is serving traffic is never sacrificed.

R1-R3 are uniform across models. Nothing in the code path keys on a specific model id.

## What is implemented (repo tree, not deployed)

### Per-card route variants (`cli/replica.py`)

`gpu_route_variants(model, total_gpus)` publishes one extra route per non-pinned card
(`<model_id>__gpu0`, `<model_id>__gpu1`, ...) alongside the base route, each carrying
`metadata: {"route_variant_of": <model_id>, "gpu_variant": N}`. Restricted to
single-GPU, non-small, non-embedding, non-derived models.

Because a variant is now a *published route*, "put this model on the free GPU"
becomes a routing choice instead of a config rewrite.

### The routing switch (`_cli_impl.py`)

`select_model_route(model_name, catalog, config_path, host, port, *, allow_variant=False)`
returns `(route_id, reason)`: the base id normally, or a published `<model_id>__gpuN`
when the pinned card is occupied and another card is free. Pure read, never raises,
falls back to the base route.

`_already_loaded_route()` resolves *which* route is live. Without it a request for a
model whose live copy is a variant would ask llama-swap for the base route and load a
second copy of the same weights onto the pinned card - the exact OOM the switch exists
to prevent.

Wired into `Handler._proxy_request` (the OpenAI proxy path) and
`Handler._handle_ollama_chat`. **Bookkeeping stays on the catalog `model_id`**:
`request_started`/`request_finished`, `get_catalog_model_process`, `_gpu_occupants` and
affinity all receive the base id, because a `__gpuN` id is not in the catalog and would
silently break them. Only the forwarded payload carries the variant id.

### Eviction priority

`route_eviction_cost_factor()` - llama-swap evicts the **lowest** cost first, so the
tier order *is* the policy:

| route | factor | rationale |
|---|---|---|
| `<id>__gpuN` | 0.25 | exists only to park a model on a card its base does not own; rebuildable from the still-published base |
| `<id>__replica_N` | 0.5 | a conversation bound to it is re-affinitised; the base keeps serving |
| base | 1.0 | the id every client asks for; dropping it costs a cold reload for everyone |

`evict_disposable_route_for_load()` makes the admission gate
(`Handler._reject_if_gpu_busy`) unload a disposable route on the target card instead of
refusing outright. Victims are restricted to `__gpuN` / `__replica_N` ids with no
in-flight requests, and are unloaded through the existing
`unload_replica_via_llamaswap()` helper. A base route is never a candidate.

Note the asymmetry the in-flight guard must respect: a `__gpuN` variant is booked under
its **base** id (the handler rewrites the payload but calls `request_started` with the
base id), so a variant's guard is `max(in_flight(variant), in_flight(base))`. Reading
only the variant's own counter would read 0 while it streams.

### Idle-GPU warmer (R2)

`run_idle_warmer_tick` (with `idle_warm_candidates`, `warm_route_via_llamaswap`,
`_warm_idle_card`) runs from `cli/daemon.py:start_idle_warmer`, started by `daemon_mode`.

Per tick it reads which cards have no `llama-server` pid and how much VRAM each has
free, then loads the cheapest-to-start published route that lives on the card that has
been empty longest. Sorting is `(0 if weights_resident else 1, size_mib, route_id)`:

1. **weights resident** - the model already runs somewhere, so its bytes are in the page
   cache and llama-swap re-maps instead of reading weights off storage. This is why an
   idle card should hold a *second copy of a live model* rather than the catalog's first
   entry.
2. **size** - among equally cheap starts, the smallest leaves the card most able to
   admit the next real load under R3.
3. **route id** - deterministic tiebreak.

The inputs are "is a copy of these weights resident", "how big is it", "is it published".
Nothing reads a model id, so the rule is uniform across models.

Loading is a minimal generation request (`max_tokens: 1`, one-character prompt, no stream)
aimed at the route id - llama-swap performs the load, on the card that route's own
command names. **No `config.yaml` write, no `subprocess`, no direct `llama-server`
invocation**; a wiring test asserts none of those names appear in the warmer path.

Hysteresis on both edges. Leading: the idle timestamp only advances while a card has no
pid, so a card that gains a process loses both its timestamp and its cooldown - a request
landing mid-wait can never be preceded by a warm-up that takes the card. Trailing: a card
whose verdict is `insufficient_vram` gets `cooldown_s`, because that verdict only changes
when the fleet shrinks and without a cooldown the loop would re-refuse every tick.
Transient refusals (`already_loaded`, `model_in_flight`) deliberately get no cooldown -
they clear on their own, and cooling them would delay a useful warm-up.

Config: `experimental.idle_gpu_warmer`, read per tick (no restart needed), **off by
default**: `enabled: false`, `poll_s: 30`, `idle_after_s: 300`, `cooldown_s: 900`,
`warm_timeout_s: 300`. Documented in `README.md`.

Tests: 450 passing in the targeted suite, 1509 in the full suite.

## Technical debt

### 1. The admission gate can still write `config.yaml` (highest severity)

`Handler._reject_if_gpu_busy` (`_cli_impl.py:16751`) still calls
`apply_repin(plan, model_name, catalog, args)` at line 16835 when
`experimental.runtime_placement.mode` is `repin`. That re-renders the file, which is
the fleet-wide teardown this work exists to remove. The routing switch does not cover
it: the write happens *inside the gate*, before the switch runs.

Production is on `mode: retire_only`, so this is latent, not active. But `repin` is now
a footgun that contradicts the documented invariant.

Remediation: delete the `repin` mode, or make it refuse to write while any route is
loaded. `retire_idle_occupants` (retiring an *idle* base) is consistent with R3 as
stated - "never from a base **serving traffic**" - but it still goes through the
planner's writer and should be re-expressed as an unload of an already-published route.

### 2. Four handlers have the gate but not the switch

| handler | def | gate | switch |
|---|---|---|---|
| `Handler._proxy_request` | 16434 | 16503 | 16512 |
| `Handler._handle_ollama_chat` | 16928 | 17077 | 17120 |
| `Handler._handle_openai_chat_completions` | 17343 | 17409 | - |
| `Handler._handle_openai_responses` | 18271 | 18375 | - |
| `Handler._handle_ollama_generate` | 19770 | 19865 | - |
| `Handler._handle_ollama_embeddings` | 19997 | 20081 | - |

R1 and R3 (the gate, which can now sacrifice a disposable route) apply everywhere.
R1's *routing* half does not: the four unswitched handlers always load the pinned card,
so a free second card stays unused for those endpoints. They will still reject when
the pinned card is busy even though a variant could have served the request.

Remediation: factor the `select_model_route` + payload-rewrite block into one helper and
call it from all six. The wiring tests added for the two switched handlers are
source-level assertions on that block, so they are reusable.

### 3. R2 is opt-in and has never run against real hardware

The warmer is implemented, tested and **default-off**, which means R2 has never actually
exercised a warm-up on this host. The risk is not the code path but the VRAM bill: a warm
copy holds its weights resident indefinitely, so turning it on trades guaranteed
headroom for latency. Nothing yet measures whether the warm-up actually pays for itself
on this workload, or whether `idle_after_s: 300` is the right threshold here.

Remediation: enable it deliberately, then read `idle_warmer_warmed` /
`idle_warmer_refused` in the request log over a real day before trusting it. Two things
worth deciding from that data: whether a warm copy is ever *used* before it is evicted by
R3, and whether the refusals are dominated by `insufficient_vram` (card genuinely too
small - then it is doing nothing but costing probes) or by `model_in_flight`.

Known gaps in the warmer itself: no cooldown on a warm-up *failure* (a transient
llama-swap 503 retries every `poll_s`, bounded but noisy), and no cross-process
persistence of the idle timestamp, so a manager restart re-warms from scratch.

### 4. `config.yaml` grows with catalog x GPUs

Measured on the live catalog (28 models, 2 GPUs): 28 routes today, 36 after the change -
28 base + 8 variants, so **+8 routes, +29%**. Growth is bounded by
`big models x (gpus - 1)`, so a wider multi-GPU host costs more. `llama-swap` reads the
whole file on every change and our own render is O(catalog x gpus).

Remediation: none needed yet at this size. Revisit if the card count or the number of
big models grows materially - consider a config-level switch to publish variants only
for a named subset. Watch the file's parse time as the catalog grows.

### 5. Two `/v1/models` round trips on the already-loaded path

`_already_loaded_route` calls `get_loaded_model_ids` after `live_gpu_variants` already
fetched `/v1/models`. They were not merged because `published` needs loaded *and*
unloaded ids while `get_loaded_model_ids` returns only loaded ones. Only on the
already-loaded path, and that path is not hot, so this is a tidiness item rather than a
performance one.

### 6. Placement has one notion only because it must

R1 uses the static pin; the gate uses `plan.pinned` / `effective_model_gpu_set`; the
switch uses `cached_model_gpu_sets` plus live occupancy. These agree today. A second
notion of "where does this model go" was explicitly rejected rather than deferred: the
previous attempt to make `reserved_base_gpus` depend on live occupancy broke the purity
of `render_llamaswap_config` and failed three tests. **The render must stay a pure
function of the catalog**; live state belongs in the resolver and the gate only.

## Deploy status

Nothing from this document is deployed. The deployed snapshot at
`~/.local/opt/llm-server/python/llamacpp_stack/` still has the old static placement and
the `repin` writer. Deployment needs a window with no active streams, because the
initial `update --model-id ...` that publishes the variants is itself one deliberate
fleet-wide reload.

Size of that one reload, measured from the live catalog (28 models, 2 GPUs): 28 routes
today, 36 after the change. Deploying is therefore not primarily a VRAM event, it is the
single unavoidable reload that buys every subsequent placement change for free.

Sequence:

1. `sha256sum ~/.local/state/llm-server/config.yaml` before anything, for comparison.
2. Sync the repo tree into the deployed snapshot.
3. One `update --model-id ...` to publish the variants, with no request streaming.
4. Restart `llm-server-manager` and `llm-server-router`.
5. Leave `experimental.idle_gpu_warmer.enabled` false until the switch, eviction priority
   and routing have been observed working.

Verification after deploy, in order:

1. `sha256sum config.yaml` before and after a request that should route to a variant -
   it must not change.
2. Two routes of one model resident simultaneously, each on its own card.
3. A request for a different model unloads the variant and leaves the base streaming.
4. A request for a model whose live copy is a variant reaches that variant rather than
   starting a second copy.
5. No `route_multiple_live_routes` event in the request log - its presence means a double
   load already happened.