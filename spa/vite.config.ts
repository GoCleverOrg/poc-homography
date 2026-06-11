import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/gcp': 'http://localhost:8000',
      '/point-picker': 'http://localhost:8000',
      '/line-picker': 'http://localhost:8000',
      '/camera-annotator': 'http://localhost:8000',
      '/camera-line-annotator': 'http://localhost:8000',
      '/camera-diagnostic': 'http://localhost:8000',
      '/camera-evaluation': 'http://localhost:8000',
      '/homography-precision': 'http://localhost:8000',
      '/lens-calibration': 'http://localhost:8000',
      '/distortion-validator': 'http://localhost:8000',
      '/clean-plate': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
