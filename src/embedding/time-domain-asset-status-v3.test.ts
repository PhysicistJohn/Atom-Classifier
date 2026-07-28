import { describe, expect, it } from 'vitest';
import {
  admitTimeDomainAssetStatusV3,
  expectedTimeDomainAssetStatusV3,
} from './time-domain-asset-status-v3.js';

describe('v3 asset deployment admission', () => {
  it('keeps staging as the development default', () => {
    expect(admitTimeDomainAssetStatusV3(
      'staging_not_release',
      'asset.status',
    )).toBe('staging_not_release');
    expect(expectedTimeDomainAssetStatusV3('staging')).toBe(
      'staging_not_release',
    );
  });

  it('requires an explicit release status in production', () => {
    expect(admitTimeDomainAssetStatusV3(
      'release',
      'asset.status',
      { admission: 'production' },
    )).toBe('release');
    expect(() => admitTimeDomainAssetStatusV3(
      'staging_not_release',
      'asset.status',
      { admission: 'production' },
    )).toThrow(/must be release for production admission/);
  });

  it('does not let release assets silently enter staging tests', () => {
    expect(() => admitTimeDomainAssetStatusV3(
      'release',
      'asset.status',
      { admission: 'staging' },
    )).toThrow(/must be staging_not_release for staging admission/);
  });
});
