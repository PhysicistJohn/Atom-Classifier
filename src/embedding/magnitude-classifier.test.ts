import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { forwardChannels, standardizeFeatures, type EmbeddingModel } from './embedding-runtime.js';
import { classify, loadPrototypeSet } from './prototype-classifier.js';
import { MagnitudeWaveformClassifier } from './magnitude-classifier.js';

const assets = new URL('./assets/', import.meta.url);
const model = JSON.parse(readFileSync(new URL('magnitude-weights.json', assets), 'utf8')) as EmbeddingModel;
const proto = JSON.parse(readFileSync(new URL('magnitude-prototypes.json', assets), 'utf8'));
const fixture = JSON.parse(readFileSync(new URL('magnitude-parity-fixture.json', assets), 'utf8'));

describe('magnitude flavor', () => {
  it('forward-pass parity with the numpy reference (<= 1e-4)', () => {
    let worst = 0;
    for (const entry of fixture.forward as Array<{ shape: number[]; features: number[]; embedding: number[] }>) {
      const feat = standardizeFeatures(model, entry.features);
      const z = forwardChannels(model, [Float64Array.from(entry.shape)], feat);
      expect(z.length).toBe(model.embed_dim);
      for (let d = 0; d < z.length; d++) worst = Math.max(worst, Math.abs(z[d]! - entry.embedding[d]!));
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it('has the 7 SignalLab classes and a 1-channel input', () => {
    const clf = new MagnitudeWaveformClassifier(model, proto);
    expect(clf.classes.length).toBe(7);
    expect(model.convs[0]!.in).toBe(1); // magnitude = single spectral channel
  });

  it('classifies the reference spectra to their own class', () => {
    const set = loadPrototypeSet(proto);
    for (const entry of fixture.forward as Array<{ class: string; shape: number[]; features: number[] }>) {
      const feat = standardizeFeatures(model, entry.features);
      const z = forwardChannels(model, [Float64Array.from(entry.shape)], feat);
      expect(classify(set, z).label).toBe(entry.class);
    }
  });
});
