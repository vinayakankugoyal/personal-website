---
layout: page
title: WebGPU Dictation
date: 2026-09-13
---

<script setup>
import Dictation from '../.vitepress/theme/components/Dictation.vue'
</script>

<div class="dictation-page">
  <p class="dictation-lead">Push-to-talk speech to text. Whisper runs in your browser on WebGPU, so nothing you say leaves your machine.</p>
  <ClientOnly><Dictation /></ClientOnly>
</div>

<style>
.dictation-page {
  display: flex;
  flex-direction: column;
  min-height: calc(100vh - var(--vp-nav-height));
}
.dictation-lead {
  margin: 0;
  padding: 32px 24px 0;
  text-align: center;
  color: var(--vp-c-text-2);
}
</style>
