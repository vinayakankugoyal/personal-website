import { createContentLoader, defineConfig, type SiteConfig } from 'vitepress'
import { Feed } from 'feed'
import { writeFileSync } from 'node:fs'
import path from 'node:path'
// @ts-ignore
import footnote from 'markdown-it-footnote'

const hostname = 'https://vinayak.purelydysfunctional.com'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Vinayak Goyal",
  description: "Personal Website",
  head: [
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&display=swap' }],
    ['link', { rel: 'alternate', type: 'application/rss+xml', title: 'Vinayak Goyal', href: `${hostname}/feed.rss` }]
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
          { text: 'Tiny LLM gets a real tokenizer', link: '/blog/BPE' },
          { text: 'Booting Linux in a microVM in long mode', link: '/blog/pataka' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vinayakankugoyal' },
      {
        icon: {
          svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M6.503 20.752c0 1.794-1.456 3.248-3.251 3.248S0 22.546 0 20.752s1.456-3.248 3.252-3.248 3.251 1.454 3.251 3.248zM1.677 6.462v4.199c6.05 0 10.951 4.9 10.951 10.95h4.199c0-8.365-6.784-15.149-15.15-15.149zM1.677.001V4.2C11.157 4.2 19.8 12.844 19.8 22.323H24C24 10.526 13.474.001 1.677.001z"/></svg>'
        },
        link: '/feed.rss',
        ariaLabel: 'RSS feed'
      }
    ]
  },
  async buildEnd(config: SiteConfig) {
    const feed = new Feed({
      title: 'Vinayak Goyal',
      description: 'Personal Website',
      id: hostname,
      link: hostname,
      language: 'en',
      copyright: `Copyright © ${new Date().getFullYear()} Vinayak Goyal`
    })

    const posts = await createContentLoader('blog/*.md', {
      excerpt: true,
      render: true
    }).load()

    posts
      .filter((post) => post.frontmatter.date)
      .sort((a, b) => +new Date(b.frontmatter.date) - +new Date(a.frontmatter.date))
      .forEach((post) => {
        feed.addItem({
          title: post.frontmatter.title ?? post.url,
          id: `${hostname}${post.url}`,
          link: `${hostname}${post.url}`,
          description: post.excerpt,
          content: post.html,
          date: new Date(post.frontmatter.date)
        })
      })

    writeFileSync(path.join(config.outDir, 'feed.rss'), feed.rss2())
  }
})
