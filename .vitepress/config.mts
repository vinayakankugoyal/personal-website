import { defineConfig } from 'vitepress'
// @ts-ignore
import footnote from 'markdown-it-footnote'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Vinayak Goyal",
  description: "Personal Website",
  markdown: {
    math: true,
    config: (md) => {
      md.use(footnote)
    }
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    sidebar: [
      { text: 'Home', link: '/' },
      {
        text: 'Blog',
        items: [
          { text: 'Tiny LLM', link: '/blog/tinyLLM' },
          { text: 'Tiny LLM go brrrrr', link: '/blog/KVCache' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vinayakankugoyal' }
    ]
  }
})
