# Single-core cube GEMM — the design space (untangled)

Scope: **one core, one cube unit, operands L1-resident** (the L1→L0 boundary).
`C[m,n] += A[m,k]·B[k,n]`, with `A∈L0A`, `B∈L0B`, `C` accumulating in `L0C`, all
staged from L1. No parallelism, no multi-core, no reduction across workers — this
is the inner tiling a closed-form chooser solves.

## The tangle we are fixing

The earlier "four variants" (split-K, full-K, accumulator-blocking,
double-buffering) are **not four algorithms**. They are settings on *different*,
*independent* axes that compose freely — e.g. L0C double-buffering applies to all
of them; accumulator-blocking works with or without operand reuse. Listing them as
mutually-exclusive alternatives conflated orthogonal choices. The fix: name the
axes, and treat a "variant" as a point in their product.

## The orthogonal axes

A design point is a tuple

```
( m, n, k , reuse , N_acc , depthA, depthB, depthC )
```

### Axis A — Tile sizes `(m, n, k)`
The L0 tile dims. Continuous (16-aligned). `k = K` is the special case "the whole
reduction fits one L0 load"; `k < K` splits the reduction and accumulates in L0C
over `⌈K/k⌉` blocks. Capacity constraints (depth from Axis D):

```
m·k·bytes_a ≤ |L0A| / depthA
k·n·bytes_b ≤ |L0B| / depthB
N_acc·m·n·bytes_c ≤ |L0C| / depthC
```

### Axis B — Operand reuse / loop order  `reuse ∈ {OS, AS, BS}`
Which operand (if any) is held and reused across the orthogonal output loop:
- **OS** output-stationary — stream *both* A and B for every output tile (no
  operand reuse; C written once).
- **AS** A-stationary — hold the A panel, reuse it across the N loop, stream B.
- **BS** B-stationary — hold the B panel, reuse it across the M loop, stream A.

Two *implementations* of operand reuse, picked by capacity (not a separate axis):
hold the operand **panel in L0** (possible when its panel fits — i.e. `k=K`, the
"full-K" case), **or** hold the orthogonal **accumulators in L0C** (Axis C, the
split-K case). Same reuse, different buffer spent.

### Axis C — Accumulator blocking  `N_acc ≥ 1`
How many output C-tiles are kept co-resident in L0C. `N_acc>1` lets one streamed
operand panel be reused across `N_acc` output tiles **while K is split** — i.e. it
is the L0C-resident way to get Axis-B reuse when `k<K`. `N_acc=1` is a single
accumulator.

### Axis D — Buffering depth  `depthA, depthB, depthC ∈ {1, 2, S}`
Per buffer. `2` = ping-pong so the next load overlaps the current compute; `S>2` =
multistage for longer L1→L0 latency; `1` = single (no overlap, full buffer). The
**stationary** operand uses depth `1` (it is held, not reloaded — buffering it just
wastes capacity).

## Untangling the old "variants" → axis settings

| old "variant" | really means | A (k) | B | C | D |
| --- | --- | --- | --- | --- | --- |
| split-K *(misnomer: serial-K)* | output-stationary, no reuse | `k<K` | **OS** | 1 | — |
| full-K | operand-stationary via L0 residency | `k=K` | **AS/BS** | 1 | — |
| accumulator-blocking | operand reuse via L0C residency | `k<K` | **AS/BS** | `>1` | — |
| double-buffering | overlap load/compute | — | — | — | **2** |
| asymmetric-buffered full-K | full-K + DB only the moving side | `k=K` | AS/BS | 1 | depthA≠depthB |
| L0C double-buffering | hide the drain | — | — | — | depthC=2 |

They are not six things — they are choices on A/B/C/D. Any consistent combination
is a valid kernel (e.g. `AS` + `k<K` + `N_acc=4` + `depthB=2,depthC=2`).

## Decision rules (when to use which) — to be grounded in the literature

> A focused single-core literature pass is recalling the analytical rules
> (Goto–van de Geijn block-panel; BLIS loops-around-the-micro-kernel; Low et al.
> "Analytical Modeling Is Enough for High-Performance BLIS"; Williams roofline).
> The rules below are our first-principles + perf-sim-validated reading; citations
> land when that pass returns.

- **A (sizes):** maximize the tile (amortize MAD startup; reuse ∝ tile area),
  subject to capacity; aspect biased by the operand-bandwidth balance
  (`m:n = bytes_b·BW_A : bytes_a·BW_B`, here **2:1** tall — validated). `k=K`
  whenever the panel fits, to unlock cheap operand reuse.
- **B (reuse direction):** hold the operand that is **more expensive to reload**
  stationary — for us the slow-L0B operand (B), reload the fast-L0A operand (A);
  decided by the **bandwidth-weighted** `T_row/T_col`, not bytes (validated).
- **C (N_acc):** raise `N_acc` to amortize the streamed-operand load across more
  output tiles, until either the load is hidden under the MACs or L0C fills.
- **D (buffering):** depth `2` on the **moving** operand(s) to overlap load with
  compute (the roofline assumption); depth `1` on the stationary operand; `depthC=2`
  when the drain is **exposed** (single-L0C stalls the cube) — validated 13–37%.

## Which axis matters is set by the bound (validated regimes)

| bound | regime | the deciding axis |
| --- | --- | --- |
| MTE1 (L1→L0) | skinny / large-K, heavy reload | **B/C** — operand reuse |
| FIXP (drain) | small-K, large output | **D** — L0C double-buffer |
| CUBE | large-K square | **A** — biggest tile (least head) |

So no axis dominates globally; the chooser picks each axis by the predicted bound.
