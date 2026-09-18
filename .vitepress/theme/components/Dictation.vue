<script setup>
// Push-to-talk dictation. Whisper runs in a Web Worker on WebGPU; the mic is
// captured through an AudioWorklet. Both scripts are served from public/ so
// they bypass the bundler (see public/dictation/).
import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue'
import { withBase } from 'vitepress'

const SAMPLE_RATE = 16000

const HINTS = {
  loading: 'Downloading speech model...',
  ready: 'Hold space to talk',
  recording: 'Listening...',
  transcribing: 'Transcribing...',
  error: '',
}

// Reactive UI state.
const state = ref('loading')        // loading | ready | recording | transcribing | error
const errorMessage = ref('')
const progress = ref(0)             // 0–100, model download
const paragraphs = ref([])          // { datetime, time, text }
const transcript = ref(null)

// No space bar to hold on a phone.
const touch = window.matchMedia('(pointer: coarse)').matches

const hint = computed(() => {
  if (state.value === 'error') return errorMessage.value
  if (state.value === 'ready' && touch) return 'Hold the button to talk'
  return HINTS[state.value]
})

function setState(next, detail) {
  state.value = next
  errorMessage.value = detail ?? ''
}

// Non-reactive audio plumbing. Lives outside Vue on purpose: the sample
// buffer is large and mutated in a hot path.
let worker = null
let ctx = null
let audioNode = null
let disposed = false

const session = {
  buffer: new Float32Array(0),   // all 16 kHz audio since the press
  stream: null,
  source: null,
}

function resetSession() {
  session.buffer = new Float32Array(0)
  session.stream = null
  session.source = null
}

async function startRecording() {
  const resumed = ctx.resume()   // iOS only allows this inside the gesture, so before any await
  session.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  session.source = ctx.createMediaStreamSource(session.stream)
  session.source.connect(audioNode)
  await resumed
}

function stopRecording() {
  session.source?.disconnect()
  session.stream?.getTracks().forEach((t) => t.stop())
}

function addParagraph(text) {
  if (!text) return
  const now = new Date()
  paragraphs.value.push({
    datetime: now.toISOString(),
    time: now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
    text,
  })
  nextTick(() => {
    transcript.value?.lastElementChild?.scrollIntoView({ block: 'end' })
  })
}

// Click any paragraph to fix a word. Enter or click away to finish, Esc to undo.
let before = ''
function onEditFocus(event) {
  before = event.target.textContent
}
function onEditEscape(event) {
  event.target.textContent = before
  event.target.blur()
}

function copy() {
  navigator.clipboard.writeText(paragraphs.value.map((p) => p.text).join('\n\n'))
}

function clear() {
  paragraphs.value = []
}

// press/release are shared by the space bar and the on-screen button.
let held = false

async function press() {
  if (held || state.value !== 'ready') return
  held = true
  try {
    await startRecording()
  } catch {
    held = false
    setState('error', 'Microphone access was blocked. Allow it in your browser settings and reload.')
    return
  }
  if (!held) {                  // let go while the permission prompt was up
    stopRecording()
    resetSession()
    return
  }
  setState('recording')
}

function release() {
  if (!held) return
  held = false
  if (state.value !== 'recording') return
  stopRecording()

  const audio = session.buffer
  resetSession()
  if (audio.length < SAMPLE_RATE / 2) {           // < 0.5 s: nothing to say
    setState('ready')
    return
  }
  setState('transcribing')
  worker.postMessage({ type: 'transcribe', audio }, [audio.buffer])
}

function onKeyDown(event) {
  if (event.target.isContentEditable) return   // typing a correction, not talking
  if (event.code !== 'Space' || event.repeat) return
  event.preventDefault()
  press()
}

function onKeyUp(event) {
  if (event.target.isContentEditable) return
  if (event.code !== 'Space') return
  release()
}

function onPointerDown(event) {
  if (event.button !== 0) return
  event.currentTarget.setPointerCapture(event.pointerId)   // keep the release even if the finger slides off
  press()
}

onMounted(async () => {
  worker = new Worker(withBase('/dictation/worker.js'), { type: 'module' })
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') progress.value = data.progress ?? 0
    if (data.type === 'ready') setState('ready')
    if (data.type === 'result') {
      addParagraph(data.result.text.trim())
      setState('ready')
    }
    if (data.type === 'error') setState('error', data.message)
  }
  worker.postMessage({ type: 'load' })

  ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
  try {
    await ctx.audioWorklet.addModule(withBase('/dictation/processor.js'))
  } catch (e) {
    setState('error', `Could not start audio capture: ${e.message}`)
    return
  }
  if (disposed) return
  audioNode = new AudioWorkletNode(ctx, 'capture')
  audioNode.port.onmessage = ({ data }) => {
    const next = new Float32Array(session.buffer.length + data.length)
    next.set(session.buffer)
    next.set(data, session.buffer.length)
    session.buffer = next
  }

  window.addEventListener('keydown', onKeyDown)
  window.addEventListener('keyup', onKeyUp)
  window.addEventListener('blur', release)   // the keyup never arrives if focus leaves mid-press
})

onUnmounted(() => {
  disposed = true
  window.removeEventListener('keydown', onKeyDown)
  window.removeEventListener('keyup', onKeyUp)
  window.removeEventListener('blur', release)
  stopRecording()
  resetSession()
  worker?.terminate()
  ctx?.close()
})
</script>

<template>
  <div class="dictation" :class="[`is-${state}`, { 'has-text': paragraphs.length > 0 }]">
    <div v-if="paragraphs.length" class="actions">
      <button type="button" @click="copy">Copy</button>
      <button type="button" @click="clear">Clear</button>
    </div>

    <div ref="transcript" class="transcript">
      <p v-for="(p, i) in paragraphs" :key="i">
        <time :datetime="p.datetime">{{ p.time }}</time>
        <span
          contenteditable="plaintext-only"
          spellcheck="false"
          @focus="onEditFocus"
          @blur="p.text = $event.target.textContent"
          @keydown.enter.prevent="$event.target.blur()"
          @keydown.esc="onEditEscape"
        >{{ p.text }}</span>
      </p>
    </div>

    <div class="status">
      <p class="hint">{{ hint }}</p>
      <progress v-if="state === 'loading'" class="progress" max="100" :value="progress"></progress>
      <button
        v-if="state !== 'error'"
        type="button"
        class="talk"
        aria-label="Hold to talk"
        :aria-pressed="state === 'recording'"
        :disabled="state !== 'ready' && state !== 'recording'"
        @pointerdown="onPointerDown"
        @pointerup="release"
        @pointercancel="release"
        @contextmenu.prevent
      >
        <svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <rect x="9" y="3" width="6" height="11" rx="3" />
          <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.dictation {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 60vh;
  color: var(--vp-c-text-1);
}

/* Empty: the hint is the page, large and centered. */
.dictation:not(.has-text) {
  justify-content: center;
}
.dictation:not(.has-text) .status {
  position: static;
  background: none;
}
.dictation:not(.has-text) .hint {
  font-size: 1.7rem;
  color: var(--vp-c-text-1);
}

.actions {
  position: sticky;
  top: 12px;
  z-index: 1;
  align-self: flex-end;
  display: flex;
  gap: 4px;
  margin: 12px 24px 0;
}
@media (min-width: 960px) {
  .actions {
    top: calc(var(--vp-nav-height) + 12px);
  }
}

.actions button {
  font-family: inherit;
  font-size: 14px;
  font-weight: 500;
  line-height: 1;
  color: var(--vp-c-text-2);
  background: none;
  border: 0;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
}
.actions button:hover {
  color: var(--vp-c-text-1);
  background: var(--vp-c-default-soft);
}
.actions button:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}

.transcript {
  width: 100%;
  max-width: calc(62ch + 48px);
  margin: 0 auto;
  padding: 6vh 24px 20vh;      /* bottom room so the last line clears the hint */
  font-size: 1.125rem;
  line-height: 1.55;
}

.transcript p {
  display: grid;
  grid-template-columns: 7ch 1fr;
  gap: 0 1.2ch;
  margin: 0 0 1.1em;
}

.transcript time {
  color: var(--vp-c-text-3);
  white-space: nowrap;
  font-size: 0.72em;
  line-height: 2.15;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.transcript [contenteditable] {
  cursor: text;
  border-radius: 3px;
}
.transcript [contenteditable]:hover {
  background: var(--vp-c-default-soft);
}
.transcript [contenteditable]:focus {
  outline: none;
  background: var(--vp-c-brand-soft);
}

/* With text, the hint is a small line pinned to the bottom of the viewport. */
.status {
  position: sticky;
  bottom: 0;
  margin-top: auto;
  padding: 20px 24px calc(28px + env(safe-area-inset-bottom));
  text-align: center;
  background: linear-gradient(to top, var(--vp-c-bg) 60%, transparent);
  pointer-events: none;
}

.hint {
  margin: 0;
  color: var(--vp-c-text-2);
  transition: color 0.2s ease;
}

.hint::before {
  content: '';
  display: none;
  width: 0.55em;
  height: 0.55em;
  margin: 0 0.55em 0 0;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  vertical-align: 6%;
}
.is-recording .hint {
  color: var(--vp-c-brand-1);
}
.is-recording .hint::before {
  display: inline-block;
  animation: breathe 1.4s ease-in-out infinite;
}
@keyframes breathe {
  50% { opacity: 0.45; }
}

.is-transcribing .hint {
  color: var(--vp-c-text-3);
}
.is-error .hint {
  color: var(--vp-c-danger-1);
  max-width: 40ch;
  margin-inline: auto;
}

.talk {
  display: grid;
  place-items: center;
  width: 76px;
  height: 76px;
  margin: 18px auto 0;
  padding: 0;
  border: 2px solid var(--vp-c-brand-1);
  border-radius: 50%;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-bg);
  cursor: pointer;
  pointer-events: auto;          /* .status ignores the pointer, the button must not */
  touch-action: none;            /* a hold is not a scroll or a zoom */
  user-select: none;
  -webkit-user-select: none;
  -webkit-touch-callout: none;
  -webkit-tap-highlight-color: transparent;
  transition: transform 0.15s ease, background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
.talk:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 4px;
}
.talk:disabled {
  color: var(--vp-c-text-3);
  border-color: var(--vp-c-divider);
  cursor: default;
}
.is-recording .talk {
  color: var(--vp-c-bg);
  background: var(--vp-c-brand-1);
  transform: scale(1.08);
  animation: ring 1.4s ease-out infinite;
}
@keyframes ring {
  from { box-shadow: 0 0 0 0 var(--vp-c-brand-soft); }
  to   { box-shadow: 0 0 0 22px transparent; }
}

.progress {
  display: block;
  width: min(320px, 70vw);
  height: 3px;
  margin: 14px auto 0;
  appearance: none;
  border: 0;
  background: var(--vp-c-default-soft);
  border-radius: 2px;
  overflow: hidden;
}
.progress::-webkit-progress-bar {
  background: transparent;
}
.progress::-webkit-progress-value {
  background: var(--vp-c-brand-1);
  transition: width 0.15s ease;
}
.progress::-moz-progress-bar {
  background: var(--vp-c-brand-1);
}

@media (prefers-reduced-motion: reduce) {
  .hint::before { animation: none; }
  .hint { transition: none; }
  .talk { transition: none; }
  .is-recording .talk { animation: none; }
  .progress::-webkit-progress-value { transition: none; }
}

@media (max-width: 600px) {
  .dictation:not(.has-text) .hint { font-size: 1.35rem; }
  .dictation.is-error .hint { font-size: 1.05rem; }
  .transcript { font-size: 1rem; padding: 4vh 20px 8vh; }
  .transcript p { grid-template-columns: 1fr; }
  .transcript time { display: none; }
}
</style>
