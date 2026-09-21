import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'
export default defineConfig({
  plugins: [viteSingleFile(), {
    name: 'singlefile-slidev-output',
    enforce: 'post',
    configResolved(config) {
      for (const buildOptions of [config.build.rollupOptions, config.build.rolldownOptions]) {
        if (!buildOptions) continue
        const outputs = Array.isArray(buildOptions.output) ? buildOptions.output : [buildOptions.output]
        for (const output of outputs) if (output) delete output.manualChunks
      }
    },
  }],
  build: { cssCodeSplit: false },
})
