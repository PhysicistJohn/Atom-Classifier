import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  createTimeDomainDualFusionClassifierV3,
  TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID,
  TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET,
  TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET,
  TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET,
} from './time-domain-dual-fusion-classifier-v3.js';

const PACKAGE_SCHEMA =
  'atomos.v3.time-domain-classifier.dual-runtime-package';
const BINDING_ASSET = 'time-domain-v3-dual-binding.json';
const RELEASE_STATUS = 'release';
const RELEASE_SEED = 20260736;
const RELEASE_REPORT_SHA256 =
  '2b0ad1f06d066db4892124b039355eb96778dd4e9ba842fa5c2d4064423e9ddb';
const RELEASE_MANIFEST_SHA256 =
  '6e6ff5d3015028b35d264bbc1addccbfcb1d1158d8056e36c354c5b2d74a30ce';

interface AssetRecord {
  path: string;
  bytes: number;
  sha256: string;
  status: string;
}

interface ReleaseManifest {
  schema: string;
  schema_version: number;
  status: string;
  candidate_id: string;
  architecture: {
    execution_order: string[];
    classifier_runs_only_after_rejector_acceptance: boolean;
    public_known_label_from_classifier_only: boolean;
  };
  assets: Record<string, AssetRecord>;
  promotion: {
    release_seed: number;
    evaluation: {
      status: string;
      release_evidence: boolean;
      all_release_gates_pass: boolean;
      gate_contract: string;
      report_sha256: string;
    };
    operation: {
      development_data_loaded: boolean;
      recalibration_performed: boolean;
      sealed_corpus_opened_by_promoter: boolean;
      training_performed: boolean;
    };
  };
}

const packageRoot = new URL('./assets-v3-release/', import.meta.url);
const manifestUrl = new URL('runtime-package-manifest.json', packageRoot);

function readBytes(path: URL): Buffer {
  return readFileSync(path);
}

function sha256(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function parseJson(bytes: Buffer): unknown {
  return JSON.parse(bytes.toString('utf8')) as unknown;
}

const manifestBytes = readBytes(manifestUrl);
const manifest = parseJson(manifestBytes) as ReleaseManifest;
const assetNames = [
  TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET,
  TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET,
  TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET,
  BINDING_ASSET,
] as const;
const assetBytes = Object.fromEntries(
  assetNames.map((name) => [name, readBytes(new URL(name, packageRoot))]),
) as Record<(typeof assetNames)[number], Buffer>;
const assetDigests = Object.fromEntries(
  assetNames.map((name) => [name, sha256(assetBytes[name])]),
) as Record<(typeof assetNames)[number], string>;

describe('promoted dual-fusion q97 production package', () => {
  it('is the exact sealed release manifest and byte-bound asset set', () => {
    expect(sha256(manifestBytes)).toBe(RELEASE_MANIFEST_SHA256);
    expect(manifest).toMatchObject({
      schema: PACKAGE_SCHEMA,
      schema_version: 1,
      status: RELEASE_STATUS,
      candidate_id: TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID,
      architecture: {
        execution_order: [
          'stage_one_noise_gate',
          'rejector_known_unknown',
          'classifier_known_label',
        ],
        classifier_runs_only_after_rejector_acceptance: true,
        public_known_label_from_classifier_only: true,
      },
      promotion: {
        release_seed: RELEASE_SEED,
        evaluation: {
          status: 'complete',
          release_evidence: true,
          all_release_gates_pass: true,
          gate_contract: 'strict_imported_v2_23_of_23',
          report_sha256: RELEASE_REPORT_SHA256,
        },
        operation: {
          development_data_loaded: false,
          recalibration_performed: false,
          sealed_corpus_opened_by_promoter: false,
          training_performed: false,
        },
      },
    });
    expect(Object.keys(manifest.assets).sort()).toEqual(
      [...assetNames].sort(),
    );
    for (const name of assetNames) {
      const record = manifest.assets[name]!;
      expect(record.path, `${name}.path`).toBe(name);
      expect(record.status, `${name}.status`).toBe(RELEASE_STATUS);
      expect(statSync(new URL(name, packageRoot)).size, `${name}.bytes`).toBe(
        record.bytes,
      );
      expect(assetDigests[name], `${name}.sha256`).toBe(record.sha256);
    }
  });

  it('loads the exact promoted bytes through production admission', () => {
    const classifier = createTimeDomainDualFusionClassifierV3({
      bindingAsset: parseJson(assetBytes[BINDING_ASSET]),
      rejectorFusion: {
        asset: parseJson(
          assetBytes[TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET],
        ),
        preverifiedAssetSha256:
          assetDigests[TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET],
      },
      classifierFusion: {
        asset: parseJson(
          assetBytes[TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET],
        ),
        preverifiedAssetSha256:
          assetDigests[TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET],
      },
      opensetPolicy: {
        asset: parseJson(
          assetBytes[TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET],
        ),
        preverifiedAssetSha256:
          assetDigests[TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET],
      },
      admission: 'production',
    });

    expect(classifier.binding.status).toBe(RELEASE_STATUS);
    expect(classifier.binding.candidate_id).toBe(
      TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID,
    );
    expect(classifier.binding.frontend.uses_frequency_transform).toBe(false);
    expect(classifier.rejectorFusion.status).toBe(RELEASE_STATUS);
    expect(classifier.classifierFusion.status).toBe(RELEASE_STATUS);
    expect(classifier.openSet.asset.status).toBe(RELEASE_STATUS);
    expect(classifier.openSet.asset.schema_version).toBe(4);
  });
});
