/**
 * Nearest-prototype classification with open-set "unknown" and few-shot
 * enrollment. The decision rule is the exact one the prototypical objective was
 * trained on: assign to the nearest class prototype by squared Euclidean
 * distance, but abstain to "unknown" when even the nearest prototype is beyond
 * the calibrated distance threshold.
 */

export interface PrototypeSet {
  classes: string[];
  prototypes: Float64Array[]; // one unit-norm centroid per class
  embedDim: number;
  unknownThreshold: number;
  temperature: number;
}

export interface Classification {
  label: string | 'unknown';
  index: number; // nearest prototype index (even when unknown)
  distanceToNearest: number;
  isUnknown: boolean;
  confidence: number; // posterior mass on the nearest class
  posterior: Record<string, number>; // temperature-softmax over classes
  distances: number[];
}

export function loadPrototypeSet(json: {
  classes: string[];
  embed_dim: number;
  prototypes: number[][];
  unknown_threshold: number;
  temperature: number;
}): PrototypeSet {
  return {
    classes: json.classes.slice(),
    prototypes: json.prototypes.map((p) => Float64Array.from(p)),
    embedDim: json.embed_dim,
    unknownThreshold: json.unknown_threshold,
    temperature: json.temperature,
  };
}

function sqDist(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let d = 0; d < a.length; d++) {
    const diff = a[d]! - b[d]!;
    s += diff * diff;
  }
  return s;
}

export function classify(set: PrototypeSet, embedding: Float64Array): Classification {
  const n = set.prototypes.length;
  const distances = new Array<number>(n);
  let best = 0;
  for (let c = 0; c < n; c++) {
    distances[c] = sqDist(embedding, set.prototypes[c]!);
    if (distances[c]! < distances[best]!) best = c;
  }
  // temperature-softmax over -distance for a calibrated posterior
  const logits = distances.map((d) => -d / set.temperature);
  const maxL = Math.max(...logits);
  let z = 0;
  const exp = logits.map((l) => {
    const e = Math.exp(l - maxL);
    z += e;
    return e;
  });
  const posterior: Record<string, number> = {};
  for (let c = 0; c < n; c++) posterior[set.classes[c]!] = exp[c]! / z;

  const bestClass = set.classes[best]!;
  const isUnknown = distances[best]! > set.unknownThreshold;
  return {
    label: isUnknown ? 'unknown' : bestClass,
    index: best,
    distanceToNearest: distances[best]!,
    isUnknown,
    confidence: posterior[bestClass]!,
    posterior,
    distances,
  };
}

/**
 * Few-shot enrollment: add (or replace) a class prototype from K example
 * embeddings by averaging and re-normalising — no retraining, no gradient step.
 * Returns a new PrototypeSet; the input is not mutated.
 */
export function enroll(
  set: PrototypeSet,
  className: string,
  examples: Float64Array[],
): PrototypeSet {
  if (examples.length === 0) throw new Error('enroll requires at least one example');
  const d = set.embedDim;
  const proto = new Float64Array(d);
  for (const e of examples) for (let k = 0; k < d; k++) proto[k] = proto[k]! + e[k]!;
  let norm = 0;
  for (let k = 0; k < d; k++) {
    const v = proto[k]! / examples.length;
    proto[k] = v;
    norm += v * v;
  }
  norm = Math.sqrt(norm) + 1e-12;
  for (let k = 0; k < d; k++) proto[k] = proto[k]! / norm;

  const classes = set.classes.slice();
  const prototypes = set.prototypes.slice();
  const existing = classes.indexOf(className);
  if (existing >= 0) {
    prototypes[existing] = proto;
  } else {
    classes.push(className);
    prototypes.push(proto);
  }
  return { ...set, classes, prototypes };
}
