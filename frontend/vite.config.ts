import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('error', (err: any) => {
            // Suppress connection reset logs when backend restarts or drops sockets
            if (err.code === 'ECONNRESET' || err.code === 'ECONNREFUSED') return;
            console.warn('[Vite WS Proxy]', err.message);
          });
        },
      },
    },
  },
});
