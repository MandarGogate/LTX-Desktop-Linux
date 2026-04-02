import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import electron from 'vite-plugin-electron'
import renderer from 'vite-plugin-electron-renderer'
import path from 'path'

const isWebMode = process.env.WEB_MODE === 'true'

export default defineConfig({
  define: {
    '__ELECTRON__': isWebMode ? false : 'void 0',
  },
  plugins: [
    react(),
    ...(isWebMode ? [] : [
      electron([
        {
          entry: 'electron/main.ts',
          onstart(options) {
            if (process.env.ELECTRON_DEBUG) {
              options.startup(['--inspect=9229', '--remote-debugging-port=9222', '.', '--no-sandbox'])
            } else {
              options.startup()
            }
          },
          vite: {
            build: {
              outDir: 'dist-electron',
              sourcemap: true,
              rollupOptions: {
                external: ['electron']
              }
            }
          }
        },
        {
          entry: 'electron/preload.ts',
          onstart(options) {
            options.reload()
          },
          vite: {
            build: {
              outDir: 'dist-electron',
              sourcemap: true,
              rollupOptions: {
                output: {
                  format: 'cjs'
                }
              }
            }
          }
        }
      ]),
      renderer()
    ]),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './frontend')
    }
  },
  base: './',  // Use relative paths for Electron file:// protocol
  server: {
    host: process.env.VITE_HOST || '0.0.0.0',
    proxy: {
      '/api': {
        target: process.env.BACKEND_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: process.env.BACKEND_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/web': {
        target: process.env.BACKEND_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/outputs': {
        target: process.env.BACKEND_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/serve': {
        target: process.env.BACKEND_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: process.env.BACKEND_URL || 'http://127.0.0.1:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist'
  }
})
