import { defineConfig } from 'vitepress'
// @ts-ignore
import footnote from 'markdown-it-footnote'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Vinayak Goyal",
  description: "Personal Website",
  head: [
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&display=swap' }]
  ],
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
          { text: 'Tiny LLM go brrrrr', link: '/blog/KVCache' },
          { text: 'Tiny LLM gets a real tokenizer', link: '/blog/BPE' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vinayakankugoyal' }
    ]
  }
})
