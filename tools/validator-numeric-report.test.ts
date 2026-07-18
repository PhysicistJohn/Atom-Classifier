import { describe, expect, it } from 'vitest';
import { nonFiniteReportNumberPaths } from './validator-numeric-report.js';

describe('validator numeric report boundary', () => {
  it('rejects NaN and both infinities before JSON can coerce them to null', () => {
    expect(nonFiniteReportNumberPaths({
      finite: 0.94,
      nullableDiagnostic: null,
      metrics: {
        nan: Number.NaN,
        positiveInfinity: Number.POSITIVE_INFINITY,
      },
      rows: [1, Number.NEGATIVE_INFINITY],
    })).toEqual([
      '$.metrics.nan',
      '$.metrics.positiveInfinity',
      '$.rows[1]',
    ]);
  });

  it('accepts finite numeric leaves and nonnumeric report fields', () => {
    expect(nonFiniteReportNumberPaths({
      minimum: 0,
      maximum: 1,
      nested: [{ score: 0.5 }],
      qualification: 'synthetic-only',
      optional: null,
    })).toEqual([]);
  });
});
