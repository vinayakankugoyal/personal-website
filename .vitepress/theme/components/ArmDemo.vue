<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const canvasRef = ref(null)

let rafId = 0
let resizeObserver = null

onMounted(() => {
  const canvas = canvasRef.value
  const ctx = canvas.getContext('2d')

  // arm geometry (px); L1/L2 are the standard link-length names
  const L1 = 100
  const L2 = 50

  const EASING = 0.05 // fraction of the remaining angle covered per frame

  let baseX = 0
  let baseY = 0

  // current joint angles (radians)
  let theta1 = 0
  let theta2 = Math.PI / 2

  // joint angles the arm is easing toward
  let goalTheta1 = 0
  let goalTheta2 = 0

  // last clicked point, relative to the shoulder base
  let targetX = 0
  let targetY = 0

  function resize() {
    canvas.width = canvas.clientWidth
    canvas.height = canvas.clientHeight
    baseX = canvas.width / 2
    baseY = canvas.height / 2
  }

  // joint positions in canvas coordinates for a given pose
  function forwardKinematics(theta1, theta2) {
    const elbow = {
      x: baseX + L1 * Math.cos(theta1),
      y: baseY + L1 * Math.sin(theta1),
    }
    const hand = {
      x: elbow.x + L2 * Math.cos(theta1 + theta2),
      y: elbow.y + L2 * Math.sin(theta1 + theta2),
    }

    return { elbow, hand }
  }

  // joint angles that put the hand at (x, y), base-relative;
  // unreachable targets are clamped to full extension / full fold
  function solveIK(x, y) {
    const d = Math.hypot(x, y)

    if (d > L1 + L2) {
      return { theta1: Math.atan2(y, x), theta2: 0 }
    }

    if (d < Math.abs(L1 - L2)) {
      return { theta1: Math.atan2(y, x), theta2: Math.PI }
    }

    // law of cosines at the elbow; theta2 = 0 when the arm is straight
    const theta2 = Math.acos((d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2))

    // angle to the target, minus the angle between link 1 and the
    // base->target line
    const theta1 =
      Math.atan2(y, x) -
      Math.atan2(L2 * Math.sin(theta2), L1 + L2 * Math.cos(theta2))

    return { theta1, theta2 }
  }

  // shortest signed rotation from angle a to angle b, in (-PI, PI]
  function angleDiff(a, b) {
    const diff = b - a
    return diff - 2 * Math.PI * Math.round(diff / (2 * Math.PI))
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const { elbow, hand } = forwardKinematics(theta1, theta2)

    // x/y axes through the base
    ctx.beginPath()
    ctx.moveTo(0, baseY)
    ctx.lineTo(canvas.width, baseY)
    ctx.moveTo(baseX, 0)
    ctx.lineTo(baseX, canvas.height)
    ctx.setLineDash([6, 6])
    ctx.lineWidth = 1
    ctx.strokeStyle = '#bbb'
    ctx.stroke()
    ctx.setLineDash([])

    // arm links
    ctx.beginPath()
    ctx.moveTo(baseX, baseY)
    ctx.lineTo(elbow.x, elbow.y)
    ctx.lineTo(hand.x, hand.y)
    ctx.lineWidth = 7
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = '#444'
    ctx.stroke()

    // base and elbow joints
    ctx.beginPath()
    ctx.arc(baseX, baseY, 8, 0, 2 * Math.PI)
    ctx.arc(elbow.x, elbow.y, 6, 0, 2 * Math.PI)
    ctx.fillStyle = '#444'
    ctx.fill()

    // hand
    ctx.beginPath()
    ctx.arc(hand.x, hand.y, 6, 0, 2 * Math.PI)
    ctx.fillStyle = '#d33'
    ctx.fill()

    // target ring
    ctx.beginPath()
    ctx.arc(targetX + baseX, targetY + baseY, 10, 0, 2 * Math.PI)
    ctx.lineWidth = 2
    ctx.strokeStyle = '#d33'
    ctx.stroke()
  }

  function animate() {
    theta1 += angleDiff(theta1, goalTheta1) * EASING
    theta2 += angleDiff(theta2, goalTheta2) * EASING

    draw()

    rafId = requestAnimationFrame(animate)
  }

  canvas.addEventListener('pointermove', function (event) {
    const rect = canvas.getBoundingClientRect()
    targetX = event.clientX - rect.left - baseX
    targetY = event.clientY - rect.top - baseY

    const goal = solveIK(targetX, targetY)
    goalTheta1 = goal.theta1
    goalTheta2 = goal.theta2
  })

  resize()
  resizeObserver = new ResizeObserver(resize)
  resizeObserver.observe(canvas)

  animate()
})

onUnmounted(() => {
  cancelAnimationFrame(rafId)
  resizeObserver?.disconnect()
})
</script>

<template>
  <canvas ref="canvasRef" class="arm-demo" aria-label="Interactive 2-joint robot arm demo"></canvas>
</template>

<style scoped>
.arm-demo {
  display: block;
  width: 100%;
  height: 420px;
  margin: 1.5rem 0;
  border-radius: 8px;
  background: #fafafa;
  border: 1px solid var(--vp-c-divider);
  cursor: crosshair;
  touch-action: none;
}
</style>
