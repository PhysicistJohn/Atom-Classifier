import { describe, expect, it } from 'vitest';
import {
  RADIO_OPERATING_BAND_CONTEXT,
  compatibleRadioDuplexModes,
} from './radio-operating-band-context.js';

describe('versioned radio operating-band context', () => {
  it('pins an auditable primary-standard revision and clause for every air interface', () => {
    expect(RADIO_OPERATING_BAND_CONTEXT).toMatchObject({
      id: 'standards-operating-band-context-v1',
      sources: {
        geran: { revision: '19.0.0', clause: '2 Frequency bands and channel arrangement' },
        'e-utra': { revision: '18.5.0', clause: 'Table 5.5-1 E-UTRA operating bands' },
        nr: { revision: '18.12.0', clause: 'Table 5.2-1 NR operating bands in FR1' },
      },
    });
    expect(Object.values(RADIO_OPERATING_BAND_CONTEXT.sources)
      .every((source) => source.url.startsWith('https://www.etsi.org/deliver/'))).toBe(true);
    expect(Object.values(RADIO_OPERATING_BAND_CONTEXT.sources)
      .every((source) => /^[a-f0-9]{64}$/.test(source.documentSha256))).toBe(true);
    expect(Object.isFrozen(RADIO_OPERATING_BAND_CONTEXT)).toBe(true);
    expect(Object.isFrozen(RADIO_OPERATING_BAND_CONTEXT.entries)).toBe(true);
    expect(RADIO_OPERATING_BAND_CONTEXT.entries.every((entry) => Object.isFrozen(entry)
      && Object.isFrozen(entry.ranges))).toBe(true);
  });

  it('contains only finite ordered link ranges and unique air-interface/band rows', () => {
    const keys = new Set<string>();
    for (const entry of RADIO_OPERATING_BAND_CONTEXT.entries) {
      const key = `${entry.airInterface}:${entry.band}`;
      expect(keys.has(key), key).toBe(false);
      keys.add(key);
      expect(entry.ranges.length).toBeGreaterThan(0);
      for (const range of entry.ranges) {
        expect(Number.isSafeInteger(range.startHz), `${key} start`).toBe(true);
        expect(Number.isSafeInteger(range.stopHz), `${key} stop`).toBe(true);
        expect(range.stopHz, key).toBeGreaterThan(range.startHz);
        if (entry.duplexMode === 'tdd') expect(range.direction, key).toBe('shared');
      }
      if (entry.duplexMode === 'fdd') {
        expect(entry.ranges.map((range) => range.direction), key).toEqual(['uplink', 'downlink']);
      }
      if (entry.duplexMode === 'sdl') expect(entry.ranges[0]!.direction, key).toBe('downlink');
      if (entry.duplexMode === 'sul') expect(entry.ranges[0]!.direction, key).toBe('uplink');
    }
  });

  it('distinguishes fitted paired, shared, and supplemental-only contexts', () => {
    expect([...compatibleRadioDuplexModes('geran', 947_300_000, 947_500_000)]).toEqual(['fdd']);
    expect(compatibleRadioDuplexModes('e-utra', 1_838_000_000, 1_842_000_000)).toEqual(new Set(['fdd']));
    expect(compatibleRadioDuplexModes('e-utra', 2_590_000_000, 2_600_000_000)).toEqual(new Set(['tdd', 'sdl']));
    expect(compatibleRadioDuplexModes('nr', 3_480_000_000, 3_520_000_000)).toEqual(new Set(['tdd']));
    // Release-18 n109 and n83 overlap n29. Preserve all three contexts rather
    // than silently relabeling the n29 row as paired spectrum.
    expect(compatibleRadioDuplexModes('nr', 717_200_000, 727_800_000)).toEqual(new Set(['sdl', 'sul', 'fdd']));
  });

  it('preserves overlap instead of forcing one duplex story from frequency alone', () => {
    expect(compatibleRadioDuplexModes('nr', 1_715_000_000, 1_775_000_000)).toEqual(new Set(['fdd', 'sul']));
    expect(compatibleRadioDuplexModes('nr', 1_440_000_000, 1_460_000_000)).toEqual(new Set(['fdd', 'tdd', 'sdl']));
  });

  it('requires complete interval containment and makes edge tolerance explicit', () => {
    expect(compatibleRadioDuplexModes('nr', 5_900_000_000, 5_950_000_000).has('tdd')).toBe(false);
    expect(compatibleRadioDuplexModes('nr', 5_925_000_000, 6_025_000_000).has('tdd')).toBe(true);
    expect(compatibleRadioDuplexModes('nr', 3_299_900_000, 3_400_000_000).has('tdd')).toBe(false);
    expect(compatibleRadioDuplexModes('nr', 3_299_900_000, 3_400_000_000, 100_000).has('tdd')).toBe(true);
  });

  it('rejects malformed intervals instead of silently broadening support', () => {
    expect(() => compatibleRadioDuplexModes('nr', Number.NaN, 1)).toThrow(/finite nondecreasing/i);
    expect(() => compatibleRadioDuplexModes('nr', 2, 1)).toThrow(/finite nondecreasing/i);
    expect(() => compatibleRadioDuplexModes('nr', 1, 2, -1)).toThrow(/finite nondecreasing/i);
  });
});
