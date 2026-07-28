/**
 * Deployment admission for the v3 time-domain asset family.
 *
 * A release promotion changes all three cooperating assets (encoder/fusion,
 * classifier head, and staged open-set policy) to `release`.  Runtime callers
 * choose an admission channel and the loaders fail closed if an asset from the
 * other channel is supplied.  This keeps development artifacts usable without
 * making it possible for a production integration to load one accidentally.
 */

export const TIME_DOMAIN_STAGING_STATUS_V3 = 'staging_not_release' as const;
export const TIME_DOMAIN_RELEASE_STATUS_V3 = 'release' as const;

export type TimeDomainAssetStatusV3 =
  | typeof TIME_DOMAIN_STAGING_STATUS_V3
  | typeof TIME_DOMAIN_RELEASE_STATUS_V3;

export type TimeDomainAssetAdmissionV3 = 'staging' | 'production';

export interface TimeDomainAssetLoadOptionsV3 {
  /**
   * Defaults to staging for backwards-compatible development tooling.
   * Production entry points must pass `production` explicitly.
   */
  admission?: TimeDomainAssetAdmissionV3;
}

export function expectedTimeDomainAssetStatusV3(
  admission: TimeDomainAssetAdmissionV3,
): TimeDomainAssetStatusV3 {
  return admission === 'production'
    ? TIME_DOMAIN_RELEASE_STATUS_V3
    : TIME_DOMAIN_STAGING_STATUS_V3;
}

/** Validate and admit one untrusted asset status. */
export function admitTimeDomainAssetStatusV3(
  value: unknown,
  path: string,
  options: TimeDomainAssetLoadOptionsV3 = {},
): TimeDomainAssetStatusV3 {
  const admission = options.admission ?? 'staging';
  const expected = expectedTimeDomainAssetStatusV3(admission);
  if (value !== expected) {
    throw new RangeError(
      `${path} must be ${expected} for ${admission} admission; got `
      + `${String(value)}`,
    );
  }
  return expected;
}
