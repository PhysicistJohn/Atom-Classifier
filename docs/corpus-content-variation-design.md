# Corpus-only content variation design

This design replaces the known phase-only corpus path without changing any
SignalLab fixed catalog artifact, public measurement API, or product
provenance. A fixed artifact is not a safe place to hide a `contentSeed`: the
fixed catalog hashes, its independent-oracle claims, and the Atomizer
measurement receipts would become false for varied bytes.

## Boundary

```text
Atom-Classifier corpus generator
  ├─ standards-fixed / analytic profiles -> synthesizeAnalyticComplexIq
  ├─ Bluetooth long-dwell profiles        -> synthesizeAnalyticComplexIq
  └─ operational-content profiles         -> corpus-only SignalLab generator
                                               (mandatory contentSeed)
```

Corpus-only outputs are training data, not catalog assets. They must never be
routed through `acquireIq`, `AtomizerMeasurementService`, fixed-digital
bindings, catalog descriptors, or artifact-hash receipts.

The seed is a mandatory uint32 and is independent of phase (`OFFSET_SEED`) and
receiver impairments. It is recorded per row with a versioned recipe name.
`Atom-SignalLab/src/corpus-content-prng.ts` supplies stateless draws so whole
and split windows are byte-identical.

## Profile policy

| Profile group | Count | Corpus action |
|---|---:|---|
| CW, AM, FM; LTE E-TM; NR TM; NTM/NB-IoT | 15 | Preserve standards-fixed/analytic content; random phase only. |
| Bluetooth long-dwell | 2 | Preserve fixed packet payload; use indexed hop/event timing. Report fully silent 20 ms LE rows. |
| GERAN | 7 | Implemented in `geran-corpus-iq.ts`, with a mandatory content seed and a restricted 7×8 proof. |
| Operational LTE/NR | 4 | Implemented in `operational-carrier-iq.ts`, varying PDSCH data only while retaining the fixed reference-grid geometry. |
| Wi-Fi | 6 | Implemented in `wlan-corpus-iq.ts`, constructing seeded non-qualified PPDUs while preserving fixed PHY geometry. |

## GERAN

The two GMSK xCCH profiles require a real TS 45.003 xCCH encoder. Their
current four encoded bursts are one pinned libosmocore result and must not be
randomized directly. `geran-xcch-corpus-codec.ts` derives a synthetic 23-octet
L2 block from `(contentSeed, blockIndex)`, encodes/interleaves it into four
xCCH bursts, and retains TSC0, tails, TS0 placement, and (for the loaded
profile) the TS1–TS7 dummy bursts. It is pinned to the retained libosmocore
dummy-frame fixture and to a separate checked-in libosmocore coding-test
fixture at a different input; both encode and decode exactly, in addition to
the synthetic round-trip tests.

The other five GERAN profiles make no existing channel-coding claim. Their
corpus path may vary only the declared encrypted/payload fields, while keeping
tails, training sequences, active duration, modulation, rotation, and slot
schedule unchanged.

## Operational LTE/NR

Keep PCI, timing, allocation, reference signals, DM-RS, PDCCH, and TDD
structure fixed. A corpus-only path changes only PDSCH data symbols via a
stateless seed-derived bit stream, then renders the same OFDM geometry.
`operational-carrier-iq.ts` starts from each fixed reference grid, changes only
the QPSK PDSCH components, and re-renders only OFDM symbols containing PDSCH.
All non-PDSCH resource elements and all non-PDSCH time symbols remain exact
copies of the fixed reference; the n78 inactive TDD intervals remain zero.

This is a standards-shaped PDSCH-content derivative, not an E-TM/TM artifact
and not a claim that arbitrary seed values form a decode-valid transport block
without a full transport-block encoder, rate matcher, CRC, and control channel.

## Wi-Fi

Keep PHY preamble, SIG fields, RU allocation, MCS, packet length, and duration
fixed. The corpus-only WLAN path covers all six profiles. Its legacy paths use:

- HR-DSSS: a locally administered ACK receiver address, recomputed FCS, and a
  non-zero scrambler state; the PLCP fields and packet geometry remain fixed.
- ERP-OFDM: a locally administered ACK receiver address, recomputed FCS, and
  non-zero scrambler state; the rendered training, SIGNAL, and signal-extension
  samples remain byte-identical to the fixed PPDU.

The same path implements HE PPDUs per row:

- HR-DSSS and ERP OFDM: vary locally administered ACK receiver address, FCS,
  and nonzero scrambler state.
- HE SU/ER/TB: vary the QoS payload and FCS after the fixed MAC header, plus
  scrambler state. The renderer replaces data-resource elements only, leaving
  the fixed preamble, SIG fields, RU allocation, MCS, pilots, packet length,
  and duration byte-identical.
- HE MU: derive independent payload/FCS/scrambler streams per user while
  leaving both RUs and HE-SIG-B unchanged. The renderer preserves all
  non-data samples byte-identically.

Version 1 is cyclic PPDU replay with no idle/backoff model. That is explicit:
it separates content generalization from traffic/cadence modeling and must not
be labeled CSMA behavior.

## Evidence gate

For every corpus-only generator, prove all of the following before the 34×8
proof slice:

1. Same seed and coordinates are byte-identical; arbitrary chunk stitching is
   byte-identical.
2. Eight prescribed content seeds at one fixed phase produce eight distinct
   clean hashes for every required profile.
3. Non-content geometry is unchanged: GERAN tails/training/slot placement,
   LTE/NR non-PDSCH resource elements and TDD silence, and Wi-Fi control and
   training regions.
4. Existing SignalLab catalog hashes, public dispatch, and independent-oracle
   suites remain unchanged.
5. The stage-1 manifest records `contentSeed`, recipe version, generation
   mode, a fixed-phase content probe hash, and makes no fixed-artifact claim.

All 17 operational-content paths have dedicated 8-row evidence. `ALLOW_PHASE_ONLY=1`
must not be used for the combined 34-profile × 8-row proof slice or production
invocation; that combined proof remains the final stage-1 evidence artifact.
