import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({ plugins: [react()], server: {host:'127.0.0.1', port:18900, proxy:{'/api':'http://127.0.0.1:18901','/artifacts':'http://127.0.0.1:18901'}}, preview:{port:18900} });
