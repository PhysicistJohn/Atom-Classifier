/**
 * Trusted acquisition provenance -> v4 prototype-bank routing.
 *
 * This module is intentionally independent of the encoders.  Controllers can
 * choose the prototype source synchronously without pulling CNN code into
 * their bundle.  SignalLab is routed to the current bank only for the exact
 * 31-profile inventory used to fit that bank; analog, reference-constellation,
 * custom, physical-SDR, and untagged captures use the historical bank.
 */

export type TimeDomainPublicClassV4 =
  'am' | 'bluetooth' | 'cw' | 'dsss' | 'fm' | 'gsm' | 'ofdm';
export type TimeDomainPrototypeSourceV4 = 'current' | 'historical';
export type TimeDomainAcquisitionSourceV4 =
  'signal-lab' | 'physical-sdr' | 'untagged';

export const TIME_DOMAIN_PROTOTYPE_SOURCES_V4 = Object.freeze([
  'current',
  'historical',
] as const);

export const TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_KIND_V4 =
  'trusted_acquisition_source_to_prototype_source_v1' as const;
export const TIME_DOMAIN_SIGNAL_LAB_PROFILE_ROUTING_RULE_V4 =
  'current_only_for_exact_current_profile_inventory_else_historical' as const;

/**
 * Independent admission map for current-source profile centroids.  In
 * particular, WiFi HR-DSSS is always mapped to the DSSS public class.
 */
export const TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4 = Object.freeze({
  'bluetooth-classic-connected': 'bluetooth',
  'bluetooth-le-advertising': 'bluetooth',
  'gsm-16qam-higher-symbol-rate-burst': 'gsm',
  'gsm-32qam-higher-symbol-rate-burst': 'gsm',
  'gsm-8psk-normal-burst': 'gsm',
  'gsm-900-loaded-bcch': 'gsm',
  'gsm-aqpsk-normal-burst': 'gsm',
  'gsm-normal-burst': 'gsm',
  'gsm-qpsk-higher-symbol-rate-burst': 'gsm',
  'lte-band3-fdd-20m': 'ofdm',
  'lte-band38-tdd-10m': 'ofdm',
  'lte-etm1.1': 'ofdm',
  'lte-etm3.1': 'ofdm',
  'lte-etm3.1a': 'ofdm',
  'lte-etm3.1b': 'ofdm',
  'lte-nbiot-guard-isolated-component': 'ofdm',
  'lte-nbiot-inband-isolated-component': 'ofdm',
  'lte-ntm': 'ofdm',
  'nr-fr1-tm1.1': 'ofdm',
  'nr-fr1-tm3.1': 'ofdm',
  'nr-fr1-tm3.1a': 'ofdm',
  'nr-fr1-tm3.1b': 'ofdm',
  'nr-n3-fdd-20m': 'ofdm',
  'nr-n78-tdd-100m': 'ofdm',
  'nr-nbiot-inband-isolated-component': 'ofdm',
  'wifi-hr-dsss-11m': 'dsss',
  'wifi-ofdm-20m': 'ofdm',
  'wifi6-he-er-su': 'ofdm',
  'wifi6-he-mu': 'ofdm',
  'wifi6-he-su': 'ofdm',
  'wifi6-he-tb': 'ofdm',
} as const satisfies Readonly<Record<string, TimeDomainPublicClassV4>>);

export const TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4 = Object.freeze(
  Object.keys(TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4).sort(),
);

export interface TimeDomainTrustedSourceRoutingV4 {
  kind: typeof TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_KIND_V4;
  default_untagged: 'historical';
  acquisition_source_map: {
    'signal-lab': 'selected-profile-conditioned';
    'physical-sdr': 'historical';
    untagged: 'historical';
  };
  signal_lab_selected_profile_rule:
    typeof TIME_DOMAIN_SIGNAL_LAB_PROFILE_ROUTING_RULE_V4;
  current_profile_inventory: readonly string[];
}

export const TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4:
  Readonly<TimeDomainTrustedSourceRoutingV4> = Object.freeze({
  kind: TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_KIND_V4,
  default_untagged: 'historical',
  acquisition_source_map: Object.freeze({
    'signal-lab': 'selected-profile-conditioned',
    'physical-sdr': 'historical',
    untagged: 'historical',
  }),
  signal_lab_selected_profile_rule:
    TIME_DOMAIN_SIGNAL_LAB_PROFILE_ROUTING_RULE_V4,
  current_profile_inventory: TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4,
});

export function isTimeDomainCurrentProfileV4(
  selectedProfileId: unknown,
): selectedProfileId is keyof typeof TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4 {
  return (
    typeof selectedProfileId === 'string'
    && Object.prototype.hasOwnProperty.call(
      TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4,
      selectedProfileId,
    )
  );
}

/**
 * Resolve the only prototype bank permitted for one trusted acquisition.
 * Unknown source kinds are rejected; callers should explicitly mark them
 * `untagged`, which safely selects the historical all-public-class bank.
 */
export function prototypeSourceForAcquisitionV4(
  sourceKind: TimeDomainAcquisitionSourceV4,
  selectedProfileId?: string | null,
): TimeDomainPrototypeSourceV4 {
  if (sourceKind === 'signal-lab') {
    return isTimeDomainCurrentProfileV4(selectedProfileId)
      ? 'current'
      : 'historical';
  }
  if (sourceKind === 'physical-sdr' || sourceKind === 'untagged') {
    return 'historical';
  }
  throw new RangeError(`unsupported v4 acquisition source ${String(sourceKind)}`);
}
