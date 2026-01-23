import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Vinayak Goyal",
  description: "Personal Website",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vinayakankugoyal' }
    ]
  }
})
