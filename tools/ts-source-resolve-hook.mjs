/**
 * Minimal TypeScript source loader for the corpus generators.
 *
 * SignalLab is authored in TypeScript with NodeNext module resolution, so its
 * intra-package specifiers carry the *emitted* extension (`./contracts.js`)
 * while only `./contracts.ts` exists on disk. Node >= 22.18 strips types from
 * `.ts` files natively, but its resolver is deliberately extension-literal and
 * will not try `.ts` for a `.js` specifier. That single gap -- not the type
 * stripping -- is what stopped `generate-longdwell-probe-corpus.mjs` from
 * importing `src/complex-iq.ts`.
 *
 * This hook closes exactly that gap and nothing else: for a relative or
 * absolute specifier ending in `.js`/`.mjs`/`.cjs` that does not exist, it
 * retries the sibling `.ts`/`.mts`/`.cts`/`.tsx`. Bare specifiers, node:
 * builtins, and any specifier that resolves normally are untouched, so package
 * exports (`@atomos/dsp`) keep resolving through Node's own algorithm.
 *
 * Usage:
 *   node --import ./tools/ts-source-resolve-hook.mjs tools/<generator>.mjs
 *
 * Requires Node >= 23.5 / 22.15 for `module.registerHooks` (synchronous, no
 * worker thread) and >= 22.18 for on-by-default type stripping. Verified on
 * the project's Node 24.18.1 (~/.local/node/bin/node).
 */
import { registerHooks } from 'node:module';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const EMITTED_TO_SOURCE = new Map([
  ['.js', ['.ts', '.tsx']],
  ['.mjs', ['.mts']],
  ['.cjs', ['.cts']],
]);

function candidateSourceUrls(specifier, parentURL) {
  const dot = specifier.lastIndexOf('.');
  if (dot === -1) return [];
  const sourceExts = EMITTED_TO_SOURCE.get(specifier.slice(dot));
  if (sourceExts === undefined) return [];
  const stem = specifier.slice(0, dot);
  try {
    return sourceExts.map((ext) => new URL(stem + ext, parentURL));
  } catch {
    return [];
  }
}

registerHooks({
  resolve(specifier, context, nextResolve) {
    // Only relative/absolute specifiers name files directly; bare specifiers go
    // through package resolution (exports maps, node_modules) and must not be
    // rewritten.
    const isPathLike = specifier.startsWith('./')
      || specifier.startsWith('../')
      || specifier.startsWith('/')
      || specifier.startsWith('file:');
    if (!isPathLike) return nextResolve(specifier, context);

    try {
      const resolved = nextResolve(specifier, context);
      // Node resolves relative specifiers without stat-ing the target, so a
      // successful resolve still has to be checked against the filesystem.
      if (existsSync(fileURLToPath(resolved.url))) return resolved;
    } catch {
      // fall through to the .ts retry
    }

    for (const candidate of candidateSourceUrls(specifier, context.parentURL)) {
      if (existsSync(fileURLToPath(candidate))) {
        // `format` is deliberately omitted: leaving it undefined lets the
        // default load step classify the URL by extension and route `.ts`
        // through Node's built-in type stripper. Pinning it to 'module' would
        // hand the annotated source to the plain-JS parser.
        return { url: candidate.href, shortCircuit: true };
      }
    }
    return nextResolve(specifier, context);
  },
});
