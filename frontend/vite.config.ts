/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  server: {
    port: 5174,
    strictPort: true,
    fs: {
      // The canonical layout lives at <repo>/layouts so backend and browser
      // read exactly the same file.
      allow: [fileURLToPath(new URL('..', import.meta.url))],
    },
  },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
