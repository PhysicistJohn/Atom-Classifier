import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tools/**/*.test.ts', 'src/**/*.test.ts'],
    coverage: { provider: 'v8', reporter: ['text', 'html'] },
  },
});
