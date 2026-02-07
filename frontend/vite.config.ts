import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      // You can still provide your own manifest.json in the public directory
      // or configure it here. The plugin will merge them.
      manifest: {
        name: 'Scholar Agent - AI-Powered Research Assistant',
        short_name: 'ScholarAgent',
        description: 'An intelligent agent for academic research and literature analysis.',
        theme_color: '#ffffff',
        icons: [
          {
            src: 'icons/icon-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'icons/icon-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      },
      // Workbox configuration for caching strategies
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        maximumFileSizeToCacheInBytes: 5000000,
      }
    })
  ],
})