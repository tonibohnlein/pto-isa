# Single-core cube GEMM — the design space (untangled)

Scope: **one core, one cube unit, operands L1-resident** (the L1→L0 boundary).
`C[m,n] += A[m,k]·B[k,n]`, `A∈L0A`, `B∈L0B`, `C` accumulating in `L0C`, staged from
L1. No parallelism, no multi-core, no cross-worker reduction. This is the inner
tiling a closed-form chooser solves.

Confidence tags below: **[V]** verbatim in a primary source · **[D]** derived
(sound, not a named result) · **[U]** recalled/unverified.

## The tangle we fixed

The earlier "four variants" (split-K, full-K, accumulator-blocking,
double-buffering) are **not** four algorithms — they are settings on independent
axes that compose freely (e.g. L0C double-buffering applies to all of them).

## The minimal decomposition — 3 orthogonal axes

The canonical single-core frame is the **GotoBLAS/BLIS "loops around the
micro-kernel"** (Goto & van de Geijn, TOMS 2008; Smith et al., IPDPS 2014; Van Zee
& van de Geijn, TOMS 2015). Mapping our axes onto it collapses **4 → 3**:

| # | Axis | our old name(s) | the choice |
| --- | --- | --- | --- |
| **1** | **Tile sizes per memory level** | A (tile) **+** C (N_acc) | L0 block `(m,n,k)` **and** the accumulator micro-tile `(N_acc = mr×nr)` — the *same* sizing decision at two levels |
| **2** | **Loop permutation → stationarity** | B | which operand is pinned: output-stationary (pin C) / A-stationary / B-stationary — a *consequence* of loop order, not a free dial |
| **3** | **Double-buffer choice per buffer** | D | `dbA, dbB, dbC ∈ {1,2}` (NOT multistage — see Realizability) |

**Why 4→3:** "tile (m,n,k)" and "accumulator block N_acc" are both *tile-size*
choices — the L0 block (`mc,nc,kc`) and the MAC micro-tile (`mr,nr`); BLIS sizes
them level-by-level as one parameter family (Low'16). And "stationarity" is just
*which loop is innermost*, induced by axis 2. Net: **A+C = axis 1; B = axis 2;
D = axis 3.** Residency map, verbatim [V] (Smith'14 Fig.1): *"An `mr × nr` block of
C is in the registers; a `kc × nr` sliver of B̃ in L1; the `mr × kc` sliver of Ã is
streamed from L2."*

## Realizability in pypto (the reality check)

The goal is a highly-optimized `AutoTileMatmulL0` whose chooser navigates this whole
space; the lowering for each algorithm is implemented incrementally. What is /
isn't realizable:

| knob | navigable? | realized by | status today |
| --- | --- | --- | --- |
| **A** tile `(m,n,k)` | **yes** | `ChooseL0Tile` output | implemented |
| **A** micro-tile `N_acc` | **no — pruned** | — | evaluated in the perf-sim, found dominated/marginal → removed; fixed `N_acc=1` |
| **B** stationarity / loop order | yes (later) | new pass lowering | only **output-stationary** today (#1855) |
| **D** double-buffer choice `dbA/dbB/dbC ∈ {1,2}` | **yes** | `pipeline_stages` attr + `LowerPipelineLoops` | `dbA=dbB=2` fixed; `dbC=1` (the L0C-DB win is `dbC=2`) |
| **D′** multistage prefetch (depth `> 2`) | **NO** | — | not realizable on this hardware |

So the chooser's output grows from `(m,n,k)` to the **design point**
`(m, n, k, stationarity, dbC)` (with `dbA/dbB` derived from stationarity) — **minus
N_acc** (evaluated in the perf-sim and pruned: dominated by tile-sizing for clean
dims, marginal/padding-contingent otherwise) and **minus prefetch depth** (fixed ≤2,
no multistage). The cost model is scoped to exactly these knobs.

## Terminology — fix the "full-K" collision

"full-K" is overloaded three ways; we drop it for unambiguous terms:

| term seen | what it actually is | our axis term |
| --- | --- | --- |
| pypto **"full-K"** (#1855) | each output tile accumulates the *complete* K reduction in L0C, M/N-tiled, direct-store | **output-stationary (OS)** — axis B |
| this study's earlier **"full-K"** | a `k=K` tile (whole reduction in one L0 load) that *enables* operand reuse | **k=K single-pass** — an axis-A tile-size value |
| literature/GPU **"split-K"** | partition K across *parallel* workers + reduce | **not us** (single core) |

Going forward: **stationarity ∈ {OS, A-stationary, B-stationary}** (axis B);
**reduction ∈ {k-blocked (k<K), k=K single-pass}** (axis A). Avoid "full-K"/"split-K".

## Untangling: old "variant" → axis settings

| old "variant" | axis 1 (sizes) | axis 2 (stationarity) | axis 3 (depth) |
| --- | --- | --- | --- |
| "split-K" *(misnomer — serial-K accum.)* | `k<K`, `N_acc=1` | **output-stationary** (pin C) | — |
| full-K | `k=K` | **A/B-stationary** (pin operand) | — |
| accumulator-blocking | `k<K`, `N_acc>1` | output-stationary + operand reuse | — |
| double-buffering | — | — | depth **2** on moving operand |
| L0C double-buffering | — | — | `depthC=2` |

Two structural facts this exposes:
- **The K reduction is always accumulated in L0C** (the rank-k update). So
  *output-stationary (pin C) is the base* whenever `k<K`. Operand reuse is layered
  on top, by **two routes for the same effect**: hold the operand panel in L0
  (`k=K` → full-K) **or** hold `N_acc` accumulators in L0C (`k<K` → register
  tiling). Capacity picks the route.
- Our **"split-K" is a misnomer** — GPU split-K means *parallel* K + reduction;
  ours is serial-K accumulation. Worth renaming in the pass.

## Capacity constraints

```
m·k·bytes_a       ≤ |L0A| / depthA
k·n·bytes_b       ≤ |L0B| / depthB
N_acc·m·n·bytes_c ≤ |L0C| / depthC
```

## Decision rules per axis (cited)

### Axis 1 — tile sizes

- **Roofline ridge [V]** (Williams-Waterman-Patterson, CACM 2009):
  `Attainable = min(Peak_FLOPs, Peak_BW·AI)`; memory-bound iff `AI <
  AI_ridge = Peak_FLOPs/Peak_BW`.
- **Per-tile arithmetic intensity [D]:** bf16 operands → `AI_operands =
  (1/2)·HM(m,n)`, `HM = 2mn/(m+n)`. **`k` cancels** for operand AI — so deep K does
  *not* raise operand intensity, it amortizes the *output write*: this is the
  analytical reason to **accumulate full K in L0C** before draining. `HM` is
  maximized at **m=n** for a fixed footprint (square is most operand-efficient,
  absent port asymmetry).
- **Footprint ∝ √(buffer) [V]** (Hong-Kung STOC'81 `Ω(n³/√M)`; Irony-Toledo-Tiskin
  JPDC'04): optimal L0 tile side scales as **√(L0 capacity)**.
- **Closed-form block size [V]** (Low, Igual, Smith, Quintana-Ortí, "Analytical
  Modeling Is Enough for High-Performance BLIS," TOMS 2016):
  `kc = N_L1·C_L1 / (2·mr·S_data)` (2-way), and per cache level reserve ≥1 line for
  C and split the remaining `W−1` ways between kept/streamed operands in proportion
  to their tile dims (`C_Ar:C_Br = mr:nr`).
- **Asymmetric-port aspect [D, validated]:** minimizing `T_load = mk·b/BW_A +
  nk·b/BW_B` at fixed `mn` gives **`m:n = BW_A:BW_B`** → **2:1** for us (elongate
  along the fast A port; both ports finish their stream together). No named source —
  derived; matches our sweep.
- **Register/accumulator micro-tile:** latency floor `mr·nr ≥ N_vec·L_vfma·N_vfma`
  [V, Low'16 Eq.1]; bandwidth rule to hide the streamed load under compute
  `nr ≥ R_comp/(2·R_load)` [V, Goto §4.2.1 Eq.3]; `mr≈nr`, half the accumulator
  budget holds C [V, Goto §6.2]. ⇒ pick the **smallest** micro-tile meeting the
  floor, then grow `N_acc` toward the bandwidth rule until L0C fills.

### Axis 2 — loop order / stationarity

- **Traffic identity [D]:** `total = (#streams_A)·|A| + (#streams_B)·|B| + psum`;
  pinning an operand sets its stream-count to 1.
- **Which to pin [V]** (Goto §4.1–4.2.1): hold stationary the operand **most
  expensive to refetch** — make its resident block as large as the buffer allows,
  roughly square (`mc=kc`, max-area/min-perimeter), and place the most-reused
  operand in the **largest/slowest** buffer.
- **Default by K [V/D]:** **deep K → output-stationary** (pin the wide fp32 C
  accumulator in L0C — it is the highest-traffic, widest-precision quantity);
  **short K + a dominant input → operand-stationary.** For us: deep K → OS +
  `N_acc` reuse; short K (`k=K`) → **B-stationary** (pin the slow-L0B operand,
  stream A on the fast port) at aspect 2:1 — matches our bandwidth-weighted result.

### Axis 3 — pipeline depth per buffer

- **Prefetch distance [V]** (Mowry-Lam-Gupta ASPLOS'92): stages `= ⌈l/s⌉ + 1`
  (`l`=load latency, `s`=compute through the loop body). Corroborated [V] by ALCOP
  (ASPLOS'23): load hidden iff `T_load ≤ (N_pipe−1)·T_use`.
- **Double-buffer iff `T_load ≤ T_use`** (depth 2); **multistage (`>2`) only when a
  single stage's compute can't cover the load latency** [D from the V inequality].
- **Stationary operand → depth 1 [V]** (Goto §6.2): a held panel is in-place; only
  *streamed* operands are multi-buffered. Per-port: the **slow B port may need a
  deeper queue** than the fast A port.
- **L0C drain [D, validated]:** `depthC = ⌈drain_time/compute⌉ + 1`. The BLIS
  "single-buffer C" assumes a *cheap, hidden* store; our cube has **one L0C that
  the next MAD must wait on**, and a slow FIXPIPE drain — so when the drain is
  exposed (`drain > compute`, i.e. small-K / large-output) `depthC=2` wins
  **13–37%** (validated). For deep-K compute-bound tiles the drain hides under
  compute and `depthC=1` suffices — reconciling our result with the BLIS default.

## Cost model — scoring a design point (first formulation)

The chooser scores a design point `P = (m, n, k, N_acc, stationarity, dbC)` and
returns the min-`wall(P)`. `dbA/dbB` are *derived* from stationarity (the stationary
operand is single-buffered and uses the full buffer; the moving operand is
double-buffered); prefetch depth is not a variable (≤2). All hardware params come
from `BackendHandler::GetCostModel()` (a2a3 real, a5 placeholder).

```
# per-pipe cycle costs
C_mad   = ⌈M/m⌉·⌈N/n⌉·⌈K/k⌉ · (6 + cpr·⌈m/16⌉·⌈k/kt⌉·⌈n/16⌉)        # cube; tile-only
C_drain = M·N·bytes_c / BW_drain                                    # FIXPIPE; shape-only
C_load  =                                                            # MTE1, BW-weighted, by reuse
   OS      : ba·M·K·⌈N/n⌉/(N_acc·BW_A) + bb·K·N·⌈M/m⌉/BW_B
   A-stat  : ba·M·K/BW_A               + bb·K·N·⌈M/m⌉/BW_B           # A loaded once/row (k=K)
   B-stat  : ba·M·K·⌈N/n⌉/BW_A         + bb·K·N/BW_B                 # B loaded once/col (k=K)

wall(P) = (dbC==2) ? max(C_load, C_mad, C_drain)                     # drain hidden
                   : max(C_load, C_mad) + C_drain                    # drain exposed (cube stalls)
```

Legality (capacity):
`m·k·ba ≤ L0A/dbA`, `k·n·bb ≤ L0B/dbB`, `N_acc·m·n·bc ≤ L0C/dbC`.

Search: a handful of `(m,n,k)` candidates (BW-aspect seed `m=√(C0·bb·BW_A/(ba·BW_B))`
+ max-area + min-tile) × `stationarity ∈ {OS,AS,BS}` × `N_acc ∈ {1,2,4,…}` ×
`dbC ∈ {1,2}` — pick the min `wall` **among points whose lowering is implemented**
(a "realizable mask"). Today the mask = `{OS, N_acc=1, dbC=1}` (just re-scores the
tile); each new lowering widens the mask without touching the cost model.

## When to use which — the bound decides (validated regimes)

| bound | regime | deciding axis | setting |
| --- | --- | --- | --- |
| **MTE1** (L1→L0) | skinny / heavy reload | 1+2 | operand reuse: full-K (`k=K`) or `N_acc` |
| **FIXP** (drain) | small-K, large output | 3 | `depthC=2` |
| **CUBE** | large-K square | 1 | biggest tile (least MAD head), `depthC=1` |

## Recommendation for our cube (L0A=L0B=64KB, L0C=128KB, bf16/fp32, fast A port)

- **Deep K:** output-stationary — pin the fp32 C tile in L0C, accumulate full K
  before draining (AI on the operand ceiling); operand reuse via `N_acc`.
- **Short K (`k=K` fits L0):** B-stationary (pin slow-port operand), stream A.
- **Aspect `m:n = 2:1`**, tile side `~√(L0)`; **double-buffer (depth 2) the moving
  operand(s)** — multistage (>2) is not realizable here, so a load that exceeds one
  stage's compute is simply exposed; `dbC = 2` when drain-bound (small-K), else `1`.

## Caveats / not-yet-verified

- `m:n = BW_A:BW_B` aspect and the OS-vs-operand-stationary rule are **[D]** (sound,
  no named publication). BLIS `mc`/`nc` exact forms are reconstructed/`[U]`.
  "Input-Stationary" is **not** an ISCA'16 term. Core equations (roofline,
  `Ω(n³/√M)`, Goto cost ratio + Eq.3, Low'16 `kc`+ways, Mowry `⌈l/s⌉`) are **[V]**.
- **Cache-oblivious / recursive blocking** (Frigo'99) reaches the `√M` bound with no
  tuning but is inferior to explicit BLIS sizing for fixed hardware buffers — not
  pursued.
