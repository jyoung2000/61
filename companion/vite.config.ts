import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Tauri dev server settings — port must match tauri.conf.json devUrl.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 14420,
    strictPort: true,
  },
  build: {
    target: ['es2021', 'chrome100', 'safari15'],
    outDir: 'dist',
  },
});
