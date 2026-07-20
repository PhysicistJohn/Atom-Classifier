import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { embed, type EmbeddingModel } from './embedding-runtime.js';

const assets = new URL('./assets/', import.meta.url);
const model = JSON.parse(readFileSync(new URL('embedding-weights.json', assets), 'utf8')) as EmbeddingModel;
const fixture = JSON.parse(readFileSync(new URL('parity-fixture.json', assets), 'utf8'));

describe('embedding forward-pass parity with the numpy/torch reference', () => {
  const L = model.input_len;

  it('reproduces the exported reference embeddings within 1e-4', () => {
    let worst = 0;
    for (const entry of fixture.forward as Array<{ input: number[]; embedding: number[] }>) {
      const i = Float64Array.from(entry.input.slice(0, L));
      const q = Float64Array.from(entry.input.slice(L, 2 * L));
      const z = embed(model, i, q);
      expect(z.length).toBe(model.embed_dim);
      for (let d = 0; d < z.length; d++) worst = Math.max(worst, Math.abs(z[d]! - entry.embedding[d]!));
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it('produces unit-norm embeddings', () => {
    const entry = fixture.forward[0];
    const i = Float64Array.from(entry.input.slice(0, L));
    const q = Float64Array.from(entry.input.slice(L, 2 * L));
    const z = embed(model, i, q);
    let n = 0;
    for (const v of z) n += v * v;
    expect(Math.sqrt(n)).toBeCloseTo(1, 5);
  });
});
