/**
 * Versioned operating-band context used only as a structural support mask.
 *
 * The entries transcribe the cited standards tables. They are not a survey
 * prior, a deployment database, a regulatory authorization, or evidence that
 * an observed transmission implements the named air interface. In particular,
 * overlapping rows deliberately produce more than one compatible mode.
 */

export type RadioAirInterface = 'geran' | 'e-utra' | 'nr';
export type RadioDuplexMode = 'fdd' | 'tdd' | 'sdl' | 'sul';
export type RadioLinkDirection = 'uplink' | 'downlink' | 'shared';

export interface RadioOperatingBandRange {
  direction: RadioLinkDirection;
  startHz: number;
  stopHz: number;
}

export interface RadioOperatingBand {
  airInterface: RadioAirInterface;
  band: string;
  duplexMode: RadioDuplexMode;
  ranges: readonly RadioOperatingBandRange[];
}

export const RADIO_OPERATING_BAND_CONTEXT = {
  id: 'standards-operating-band-context-v1',
  sources: {
    geran: {
      organization: '3GPP / ETSI',
      specification: 'TS 45.005 / ETSI TS 145 005',
      revision: '19.0.0',
      clause: '2 Frequency bands and channel arrangement',
      url: 'https://www.etsi.org/deliver/etsi_ts/145000_145099/145005/19.00.00_60/ts_145005v190000p.pdf',
      documentSha256: '25ac7bebb2c3a6fd03899f799acc222c0babcb56ec806b3836df4b69a95d9850',
    },
    'e-utra': {
      organization: '3GPP / ETSI',
      specification: 'TS 36.101 / ETSI TS 136 101',
      revision: '18.5.0',
      clause: 'Table 5.5-1 E-UTRA operating bands',
      url: 'https://www.etsi.org/deliver/etsi_ts/136100_136199/136101/18.05.00_60/ts_136101v180500p.pdf',
      documentSha256: '2bbf97cb1478b8a30a25099aa55b55f54f7f41c61e1f07b57155f693e1f4bbc6',
    },
    nr: {
      organization: '3GPP / ETSI',
      specification: 'TS 38.104 / ETSI TS 138 104',
      revision: '18.12.0',
      clause: 'Table 5.2-1 NR operating bands in FR1',
      url: 'https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/18.12.00_60/ts_138104v181200p.pdf',
      documentSha256: '2009563761e39ddc758be42024b072c90aa48e304720edc0e4d1882d1ac6c54f',
    },
  },
  entries: [
    // TS 45.005 clause 2. Nested GSM-900 variants remain explicit so the
    // transcription can be checked against the named standard rows.
    paired('geran', 'T-GSM 380', 380.2, 389.8, 390.2, 399.8),
    paired('geran', 'T-GSM 410', 410.2, 419.8, 420.2, 429.8),
    paired('geran', 'GSM 450', 450.4, 457.6, 460.4, 467.6),
    paired('geran', 'GSM 480', 478.8, 486, 488.8, 496),
    paired('geran', 'GSM 710', 698, 716, 728, 746),
    paired('geran', 'GSM 750', 777, 793, 747, 763),
    paired('geran', 'T-GSM 810', 806, 821, 851, 866),
    paired('geran', 'GSM 850', 824, 849, 869, 894),
    paired('geran', 'P-GSM 900', 890, 915, 935, 960),
    paired('geran', 'E-GSM 900', 880, 915, 925, 960),
    paired('geran', 'R-GSM 900', 876, 915, 921, 960),
    paired('geran', 'ER-GSM 900', 873, 915, 918, 960),
    paired('geran', 'DCS 1800', 1710, 1785, 1805, 1880),
    paired('geran', 'PCS 1900', 1850, 1910, 1930, 1990),

    // TS 36.101 table 5.5-1. Reserved/inapplicable bands 6, 15, 16, 23 and 64 are
    // omitted. Downlink-only rows are represented as SDL rather than FDD so
    // they cannot create a paired-duplex classification on their own.
    paired('e-utra', '1', 1920, 1980, 2110, 2170),
    paired('e-utra', '2', 1850, 1910, 1930, 1990),
    paired('e-utra', '3', 1710, 1785, 1805, 1880),
    paired('e-utra', '4', 1710, 1755, 2110, 2155),
    paired('e-utra', '5', 824, 849, 869, 894),
    paired('e-utra', '7', 2500, 2570, 2620, 2690),
    paired('e-utra', '8', 880, 915, 925, 960),
    paired('e-utra', '9', 1749.9, 1784.9, 1844.9, 1879.9),
    paired('e-utra', '10', 1710, 1770, 2110, 2170),
    paired('e-utra', '11', 1427.9, 1447.9, 1475.9, 1495.9),
    paired('e-utra', '12', 699, 716, 729, 746),
    paired('e-utra', '13', 777, 787, 746, 756),
    paired('e-utra', '14', 788, 798, 758, 768),
    paired('e-utra', '17', 704, 716, 734, 746),
    paired('e-utra', '18', 815, 830, 860, 875),
    paired('e-utra', '19', 830, 845, 875, 890),
    paired('e-utra', '20', 832, 862, 791, 821),
    paired('e-utra', '21', 1447.9, 1462.9, 1495.9, 1510.9),
    paired('e-utra', '22', 3410, 3490, 3510, 3590),
    paired('e-utra', '24', 1626.5, 1660.5, 1525, 1559),
    paired('e-utra', '25', 1850, 1915, 1930, 1995),
    paired('e-utra', '26', 814, 849, 859, 894),
    paired('e-utra', '27', 807, 824, 852, 869),
    paired('e-utra', '28', 703, 748, 758, 803),
    supplemental('e-utra', '29', 'sdl', 'downlink', 717, 728),
    paired('e-utra', '30', 2305, 2315, 2350, 2360),
    paired('e-utra', '31', 452.5, 457.5, 462.5, 467.5),
    supplemental('e-utra', '32', 'sdl', 'downlink', 1452, 1496),
    ...tddBands('e-utra', [
      ['33', 1900, 1920], ['34', 2010, 2025], ['35', 1850, 1910], ['36', 1930, 1990],
      ['37', 1910, 1930], ['38', 2570, 2620], ['39', 1880, 1920], ['40', 2300, 2400],
      ['41', 2496, 2690], ['42', 3400, 3600], ['43', 3600, 3800], ['44', 703, 803],
      ['45', 1447, 1467], ['46', 5150, 5925], ['47', 5855, 5925], ['48', 3550, 3700],
      ['49', 3550, 3700], ['50', 1432, 1517], ['51', 1427, 1432], ['52', 3300, 3400],
      ['53', 2483.5, 2495], ['54', 1670, 1675],
    ]),
    paired('e-utra', '65', 1920, 2010, 2110, 2200),
    paired('e-utra', '66', 1710, 1780, 2110, 2200),
    supplemental('e-utra', '67', 'sdl', 'downlink', 738, 758),
    paired('e-utra', '68', 698, 728, 753, 783),
    supplemental('e-utra', '69', 'sdl', 'downlink', 2570, 2620),
    paired('e-utra', '70', 1695, 1710, 1995, 2020),
    paired('e-utra', '71', 663, 698, 617, 652),
    paired('e-utra', '72', 451, 456, 461, 466),
    paired('e-utra', '73', 450, 455, 460, 465),
    paired('e-utra', '74', 1427, 1470, 1475, 1518),
    supplemental('e-utra', '75', 'sdl', 'downlink', 1432, 1517),
    supplemental('e-utra', '76', 'sdl', 'downlink', 1427, 1432),
    paired('e-utra', '85', 698, 716, 728, 746),
    paired('e-utra', '87', 410, 415, 420, 425),
    paired('e-utra', '88', 412, 417, 422, 427),
    paired('e-utra', '103', 787, 788, 757, 758),
    paired('e-utra', '106', 896, 901, 935, 940),

    // TS 38.104 table 5.2-1, complete FR1 table at revision 18.12.0.
    paired('nr', 'n1', 1920, 1980, 2110, 2170),
    paired('nr', 'n2', 1850, 1910, 1930, 1990),
    paired('nr', 'n3', 1710, 1785, 1805, 1880),
    paired('nr', 'n5', 824, 849, 869, 894),
    paired('nr', 'n7', 2500, 2570, 2620, 2690),
    paired('nr', 'n8', 880, 915, 925, 960),
    paired('nr', 'n12', 699, 716, 729, 746),
    paired('nr', 'n13', 777, 787, 746, 756),
    paired('nr', 'n14', 788, 798, 758, 768),
    paired('nr', 'n18', 815, 830, 860, 875),
    paired('nr', 'n20', 832, 862, 791, 821),
    paired('nr', 'n24', 1626.5, 1660.5, 1525, 1559),
    paired('nr', 'n25', 1850, 1915, 1930, 1995),
    paired('nr', 'n26', 814, 849, 859, 894),
    paired('nr', 'n28', 703, 748, 758, 803),
    supplemental('nr', 'n29', 'sdl', 'downlink', 717, 728),
    paired('nr', 'n30', 2305, 2315, 2350, 2360),
    paired('nr', 'n31', 452.5, 457.5, 462.5, 467.5),
    ...tddBands('nr', [
      ['n34', 2010, 2025], ['n38', 2570, 2620], ['n39', 1880, 1920], ['n40', 2300, 2400],
      ['n41', 2496, 2690], ['n46', 5150, 5925], ['n48', 3550, 3700], ['n50', 1432, 1517],
      ['n51', 1427, 1432], ['n53', 2483.5, 2495], ['n54', 1670, 1675],
    ]),
    paired('nr', 'n65', 1920, 2010, 2110, 2200),
    paired('nr', 'n66', 1710, 1780, 2110, 2200),
    supplemental('nr', 'n67', 'sdl', 'downlink', 738, 758),
    paired('nr', 'n70', 1695, 1710, 1995, 2020),
    paired('nr', 'n71', 663, 698, 617, 652),
    paired('nr', 'n72', 451, 456, 461, 466),
    paired('nr', 'n74', 1427, 1470, 1475, 1518),
    supplemental('nr', 'n75', 'sdl', 'downlink', 1432, 1517),
    supplemental('nr', 'n76', 'sdl', 'downlink', 1427, 1432),
    ...tddBands('nr', [
      ['n77', 3300, 4200], ['n78', 3300, 3800], ['n79', 4400, 5000],
    ]),
    supplemental('nr', 'n80', 'sul', 'uplink', 1710, 1785),
    supplemental('nr', 'n81', 'sul', 'uplink', 880, 915),
    supplemental('nr', 'n82', 'sul', 'uplink', 832, 862),
    supplemental('nr', 'n83', 'sul', 'uplink', 703, 748),
    supplemental('nr', 'n84', 'sul', 'uplink', 1920, 1980),
    paired('nr', 'n85', 698, 716, 728, 746),
    supplemental('nr', 'n86', 'sul', 'uplink', 1710, 1780),
    supplemental('nr', 'n89', 'sul', 'uplink', 824, 849),
    ...tddBands('nr', [['n90', 2496, 2690]]),
    paired('nr', 'n91', 832, 862, 1427, 1432),
    paired('nr', 'n92', 832, 862, 1432, 1517),
    paired('nr', 'n93', 880, 915, 1427, 1432),
    paired('nr', 'n94', 880, 915, 1432, 1517),
    supplemental('nr', 'n95', 'sul', 'uplink', 2010, 2025),
    ...tddBands('nr', [['n96', 5925, 7125]]),
    supplemental('nr', 'n97', 'sul', 'uplink', 2300, 2400),
    supplemental('nr', 'n98', 'sul', 'uplink', 1880, 1920),
    supplemental('nr', 'n99', 'sul', 'uplink', 1626.5, 1660.5),
    paired('nr', 'n100', 874.4, 880, 919.4, 925),
    ...tddBands('nr', [
      ['n101', 1900, 1910], ['n102', 5925, 6425], ['n104', 6425, 7125],
    ]),
    paired('nr', 'n105', 663, 703, 612, 652),
    paired('nr', 'n106', 896, 901, 935, 940),
    paired('nr', 'n109', 703, 733, 1432, 1517),
  ] as readonly RadioOperatingBand[],
} as const;

for (const entry of RADIO_OPERATING_BAND_CONTEXT.entries) {
  Object.freeze(entry.ranges);
  Object.freeze(entry);
}
Object.freeze(RADIO_OPERATING_BAND_CONTEXT.entries);
for (const source of Object.values(RADIO_OPERATING_BAND_CONTEXT.sources)) Object.freeze(source);
Object.freeze(RADIO_OPERATING_BAND_CONTEXT.sources);
Object.freeze(RADIO_OPERATING_BAND_CONTEXT);

/** Returns every standards-table mode whose one link range contains the complete observed interval. */
export function compatibleRadioDuplexModes(
  airInterface: RadioAirInterface,
  observedStartHz: number,
  observedStopHz: number,
  edgeToleranceHz = 0,
): ReadonlySet<RadioDuplexMode> {
  if (![observedStartHz, observedStopHz, edgeToleranceHz].every(Number.isFinite)
    || observedStopHz < observedStartHz
    || edgeToleranceHz < 0) {
    throw new Error('Radio operating-band context requires a finite nondecreasing observed interval');
  }
  const modes = new Set<RadioDuplexMode>();
  for (const entry of RADIO_OPERATING_BAND_CONTEXT.entries) {
    if (entry.airInterface !== airInterface) continue;
    if (entry.ranges.some((range) => observedStartHz >= range.startHz - edgeToleranceHz
      && observedStopHz <= range.stopHz + edgeToleranceHz)) {
      modes.add(entry.duplexMode);
    }
  }
  return modes;
}

function paired(
  airInterface: RadioAirInterface,
  band: string,
  uplinkStartMhz: number,
  uplinkStopMhz: number,
  downlinkStartMhz: number,
  downlinkStopMhz: number,
): RadioOperatingBand {
  return {
    airInterface,
    band,
    duplexMode: 'fdd',
    ranges: [
      range('uplink', uplinkStartMhz, uplinkStopMhz),
      range('downlink', downlinkStartMhz, downlinkStopMhz),
    ],
  };
}

function tddBands(
  airInterface: RadioAirInterface,
  bands: readonly (readonly [band: string, startMhz: number, stopMhz: number])[],
): readonly RadioOperatingBand[] {
  return bands.map(([band, startMhz, stopMhz]) => ({
    airInterface,
    band,
    duplexMode: 'tdd',
    ranges: [range('shared', startMhz, stopMhz)],
  }));
}

function supplemental(
  airInterface: RadioAirInterface,
  band: string,
  duplexMode: 'sdl' | 'sul',
  direction: 'downlink' | 'uplink',
  startMhz: number,
  stopMhz: number,
): RadioOperatingBand {
  return { airInterface, band, duplexMode, ranges: [range(direction, startMhz, stopMhz)] };
}

function range(direction: RadioLinkDirection, startMhz: number, stopMhz: number): RadioOperatingBandRange {
  return { direction, startHz: Math.round(startMhz * 1_000_000), stopHz: Math.round(stopMhz * 1_000_000) };
}
