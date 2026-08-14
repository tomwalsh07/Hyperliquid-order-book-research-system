# Hyperliquid API notes

Confirmed by live probing (`wss://api.hyperliquid.xyz/ws`) and cross-checked
against the official docs (hyperliquid.gitbook.io) and the reference Python
SDK (`hyperliquid-dex/hyperliquid-python-sdk`) on **2026-08-13**. Exchange
APIs change — re-verify before relying on any of this in a new session,
especially the fee/funding numbers.

## 1. `l2Book` is a full snapshot, not a diff

This is the single most consequential fact for the order-book engine
(`src/hlmicro/orderbook/book.py`).

**Evidence**: two `l2Book` messages for BTC captured ~5.3s apart both
contained the complete 20-level ladder on each side (verbatim two consecutive
captures, prices only, sizes changing between them):

```json
// t=1786635979877
{"channel":"l2Book","data":{"coin":"BTC","time":1786635979877,
  "levels":[
    [{"px":"63668.0","sz":"10.0005","n":25}, {"px":"63667.0","sz":"1.45644","n":4}, "...18 more bid levels"],
    [{"px":"63669.0","sz":"5.16729","n":16}, {"px":"63670.0","sz":"1.56074","n":15}, "...18 more ask levels"]
  ]}}
// t=1786635982085 (+2.2s)
{"channel":"l2Book","data":{"coin":"BTC","time":1786635982085,
  "levels":[
    [{"px":"63668.0","sz":"9.0306","n":24}, {"px":"63667.0","sz":"1.45644","n":4}, "...18 more"],
    [{"px":"63669.0","sz":"4.76904","n":15}, {"px":"63670.0","sz":"0.53886","n":14}, "...18 more"]
  ]}}
```

Every message is a **complete replacement** of the ladder — `levels` is
`[bids, asks]`, each a list of `{px, sz, n}` (`n` = resting order count at
that price). There is no sequence number and no partial/delta message type
for `l2Book`.

This matches the official docs: the `WsBook` interface is described as a
**"Snapshot feed, pushed on each block that is at least 0.5s since last
push."**

**Implementation consequence**: `OrderBook.load_snapshot()` does a full
replace of the internal price→size map on every message — there is no
"apply diff" path to get wrong. The order book class *also* exposes a
lower-level `apply_level_update()` (upsert-or-delete-if-zero for a single
price) so it can be unit-tested against the resets/upsert/delete/crossed-book
scenarios worth covering, and so the same class could serve an `l4Book` or
other diff-based feed later — but on live Hyperliquid `l2Book` traffic today,
only `load_snapshot()` is ever exercised. This is documented rather than
silently assumed so a future maintainer knows which path is production-tested.

## 2. Subscription payload shape

Confirmed fields (from `subscriptionResponse` echoes). Note that `nLevels`,
which appears in a lot of third-party examples, is **not** a real field —
it's silently ignored:

```json
{"method":"subscribe","subscription":{"type":"l2Book","coin":"BTC"}}
{"method":"subscribe","subscription":{"type":"trades","coin":"BTC"}}
{"method":"subscribe","subscription":{"type":"activeAssetCtx","coin":"BTC"}}
```

`l2Book` accepts optional `nSigFigs` (int, price rounding — **left null/unset
deliberately**, see below), `mantissa`, and `fast` (bool). Per the docs,
`fast: true` gives **5** levels per side instead of 20 — it is *not* a
diff-mode toggle as the name might suggest. We leave it unset (→ `false` →
20 levels, confirmed in the response echo `"fast": false`).

We deliberately do **not** set `nSigFigs`: it aggregates price levels and
would corrupt imbalance/microprice math.
Full price precision is confirmed in the captures above (e.g. `63668.0`,
not rounded to fewer significant figures).

We subscribe to `l2Book`, `trades`, and `activeAssetCtx` per symbol. We do
**not** additionally subscribe to the lightweight `bbo` channel even though
it works (captured successfully, see below) — `l2Book`'s top level already
*is* the BBO, with the added benefit of order count (`n`); subscribing to
both would just double message volume for redundant information. This is a
deliberate scope decision, not an oversight.

```json
{"channel":"bbo","data":{"coin":"BTC","time":1786635980280,
  "bbo":[{"px":"63668.0","sz":"9.03076","n":24},{"px":"63669.0","sz":"5.16511","n":16}]}}
```

## 3. `trades` payload and `side` semantics

```json
{"channel":"trades","data":[
  {"coin":"BTC","side":"B","px":"63683.0","sz":"0.00049","time":1786635967504,
   "hash":"0xfc8e...","tid":592865463416094,
   "users":["0x6413...","0xcbba..."]}
]}
```

`side` is not documented in the gitbook spec (it just lists `side: string`
with no enum). By exchange convention (consistent with Hyperliquid's own
fills/order docs elsewhere and standard CLOB convention) we treat:
- `"B"` = the **taker bought** → this trade consumed resting **ask**
  liquidity.
- `"A"` = the **taker sold** → this trade consumed resting **bid**
  liquidity.

This mapping matters for the fill-simulation heuristic (`backtest/fills.py`):
we check consumption against the book side the trade's `side` implies was hit.
`users` is `[buyer_address, seller_address]` — unused (would only matter for
per-account analysis, out of scope here; we never key on it).

Occasional trades have an all-zero `hash` — these are liquidations /
ADL-adjacent internal matches per community reports; we keep them, they are
real prints that moved size out of the book.

## 4. `activeAssetCtx` — current funding, mark/oracle/mid, OI

```json
{"channel":"activeAssetCtx","data":{"coin":"BTC","ctx":{
  "funding":"0.0000124451","openInterest":"40313.7273799999",
  "prevDayPx":"63513.0","dayNtlVlm":"1371501207.1637816429",
  "premium":"-0.0004238885","oraclePx":"63696.0","markPx":"63673.0",
  "midPx":"63668.5","impactPxs":["63668.0","63669.0"],"dayBaseVlm":"21569.5501"}}}
```

`funding` is the **current hourly rate** (fraction, not %). Confirmed against
the funding docs: rate = premium component (5s-sampled, hourly-averaged) +
`clamp(interest_rate - premium, -0.0005, 0.0005)`, where the interest rate
component is fixed at **0.00125%/hour** (11.6% APR). Our captured value
`0.0000124451` (≈0.00124%/hour) is the right order of magnitude for a
near-zero-premium market, consistent with this being an hourly rate.

**Payment formula** (confirmed from docs): `funding_payment = position_size
× oracle_price × funding_rate` — uses **oracle price**, not mark price.
**Rate cap**: 4%/hour, universal. Both are config values
(`config/config.yaml: funding.*`), not hardcoded.

**Predicted funding** is a *separate* mechanism from this stream: the
`POST /info {"type":"predictedFundings"}` REST endpoint returns per-venue
predicted rates. Confirmed live response shape:

```json
[["BTC", [
  ["BinPerp", {"fundingRate":"0.00007945","nextFundingTime":1786636800000,"fundingIntervalHours":8}],
  ["HlPerp",  {"fundingRate":"0.0000125", "nextFundingTime":1786633200000,"fundingIntervalHours":1}],
  ["BybitPerp",{"fundingRate":"0.0000607","nextFundingTime":1786636800000,"fundingIntervalHours":8}]
]]]
```

Hyperliquid's own predicted rate is the `"HlPerp"` venue entry
(`fundingIntervalHours: 1`, matching the hourly settlement confirmed above);
other venues are captured too (useful context, cross-venue funding is a
common trade signal) but settle on 8h cycles so are not directly comparable
without adjustment. Our funding tracker (`analytics/funding.py`) combines
the WS `activeAssetCtx.funding` (current/last-settled, pushed continuously)
with a periodic poll of `predictedFundings` (default: every 60s — it's a
REST endpoint, not a subscription) to expose both the current and the
predicted series.

## 5. Fees (base tier, confirmed 2026-08-13)

From the official fee schedule: **Tier 0 (no VIP/staking discount)** —
taker **0.045%** (4.5 bps), maker **0.015%** (1.5 bps). These carry
volume/staking discounts that we do not model; exposed as
`config/config.yaml: fees.{maker_bps,taker_bps}`, never hardcoded into
strategy/backtest logic.

## 6. Connection keepalive

Confirmed against the reference Python SDK
(`hyperliquid-dex/hyperliquid-python-sdk`, `websocket_manager.py`): the
client sends `{"method":"ping"}` on a periodic timer and expects a
`{"channel":"pong"}` reply. The server drops idle connections at roughly
60s and the SDK's own default interval cuts that uncomfortably close, so we
use a **20s** ping interval
(`config/config.yaml: ingestion.ping_interval_s`) for more margin. No
official doc page enumerates the exact idle-timeout value or a pong
contract beyond the channel tag — this is inferred from the reference SDK's
behavior, not an explicit spec.

## 7. Reconnection / gap handling

Disconnects happen without warning, as on any public WS feed under load; I
did not force-test a disconnect during the probe. The client
(`ingestion/ws_client.py`) reconnects with exponential backoff + jitter
(`config/config.yaml: ingestion.reconnect_*`) and **always re-subscribes and
treats the next `l2Book` message as an authoritative fresh snapshot** rather
than assuming continuity — this falls out naturally from §1 (every message
already is a full snapshot), so there is no diff-continuity invariant to
violate on reconnect. Gaps are detected and logged (not silently absorbed)
by comparing wall-clock receive time against the previous message's `time`
field per (symbol, channel); a gap larger than a configurable threshold is
logged at `WARNING` and counted in the collector's run summary.

**Empirical cadence** (from both the initial probe and a live smoke-test run
of the actual collector): `l2Book` pushes land roughly every **2-5.5s** per
symbol in normal conditions — the docs' "at least 0.5s between pushes" is a
floor, not the typical interval. An initial 5s gap-warning threshold produced
false-positive warnings on essentially every update; the threshold is set to
**15s** (`config/config.yaml: ingestion.gap_warn_threshold_s`) to leave real
margin above normal cadence variation. This also means literal "millions of
raw `l2Book` messages" from live collection alone requires a genuinely long
collection window — see the README/methodology for the actual window used
and resulting row counts; `trades` volume scales with market activity and
can be substantially higher (a single aggressor trade can print many
same-hash fills against different resting orders).

## 8. Historical S3 archive — not used in this build

Hyperliquid publishes L2 book snapshots to a requester-pays S3 bucket
(`s3://hyperliquid-archive/market_data/[date]/[hour]/l2Book/[coin].lz4`),
which can bootstrap history far faster than collecting forward. **This build
uses live WS collection only** — the S3 route needs AWS credentials and
incurs egress charges, so I left it out. This is a scope decision, not an
oversight: it means
the research dataset is bounded by however long the live collector actually
ran, which is stated plainly in the README/methodology rather than padded.
Re-enabling S3 backfill later is a matter of adding a `scripts/backfill_s3.py`
that decompresses the `.lz4` snapshots into the same `data/raw/` layout the
live collector uses — the downstream normalize/analytics/research code does
not care which source raw messages came from.
