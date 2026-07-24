import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // The `tools/*.test.mjs` files are deliberately absent here: they are
    // written against the `node:test` runner, which Vitest cannot execute
    // ("No test suite found"). They run via `npm run check` instead, through
    // `check:observable-training-launcher` and
    // `check:signal-classifier-publication`.
    include: ['tools/**/*.test.ts', 'src/**/*.test.ts'],
  },
});
