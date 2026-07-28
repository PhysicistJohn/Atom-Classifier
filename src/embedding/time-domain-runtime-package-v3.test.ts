import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

interface RuntimePackageManifest {
  schema: string;
  status: string;
  assets: Record<string, { bytes: number; sha256: string }>;
}

const packageRoot = new URL('./assets-v3-staging/', import.meta.url);
const manifest = JSON.parse(
  readFileSync(new URL('runtime-package-manifest.json', packageRoot), 'utf8'),
) as RuntimePackageManifest;

function sha256(path: URL): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

describe('tracked v3 staging runtime package', () => {
  it('contains every runtime asset at the manifest-bound bytes and SHA-256', () => {
    expect(manifest.schema).toBe(
      'atomos.v3.time-domain-classifier.runtime-package',
    );
    expect(manifest.status).toBe('staging_not_release');
    expect(Object.keys(manifest.assets).sort()).toEqual([
      'time-domain-classifier-weights-v1.json',
      'time-domain-fusion-weights-v3.json',
      'time-domain-openset-smoke-v1.json',
      'time-domain-openset-weights-v1.json',
      'time-domain-probe-fixture-v3.json',
    ]);
    for (const [name, expected] of Object.entries(manifest.assets)) {
      const path = new URL(name, packageRoot);
      expect(statSync(path).size, `${name} bytes`).toBe(expected.bytes);
      expect(sha256(path), `${name} SHA-256`).toBe(expected.sha256);
    }
  });

  it('ships raw-IQ smoke coverage for accept, stage 1, and stage 2 at both lengths', () => {
    const fixture = JSON.parse(
      readFileSync(
        new URL('time-domain-openset-smoke-v1.json', packageRoot),
        'utf8',
      ),
    ) as {
      rows: Array<{
        capture_length: number;
        iq: { in_phase: number[]; quadrature: number[] };
        rejected_stage: 1 | 2 | null;
      }>;
    };
    expect(new Set(fixture.rows.map((row) => row.capture_length))).toEqual(
      new Set([4096, 8192]),
    );
    expect(new Set(fixture.rows.map((row) => row.rejected_stage))).toEqual(
      new Set([null, 1, 2]),
    );
    for (const row of fixture.rows) {
      expect(row.iq.in_phase).toHaveLength(row.capture_length);
      expect(row.iq.quadrature).toHaveLength(row.capture_length);
    }
  });
});
