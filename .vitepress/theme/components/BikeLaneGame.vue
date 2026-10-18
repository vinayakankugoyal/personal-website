<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const wrapRef = ref(null)
const canvasRef = ref(null)

// 'ready' | 'playing' | 'crashed'; drives the HTML overlays
const phase = ref('ready')
const blocks = ref('0.0')
const bagels = ref(0)
const best = ref('0.0')
const lesson = ref(null)
// hazards you got past this run, most first
const dodged = ref([])

// what each obstacle teaches when it takes you out
const LESSONS = {
  door: {
    title: 'Doored!',
    text: 'A car door swung open right in front of you. Painted bike lanes often run through the "door zone" of parked cars, so ride a door\'s width away when you can and watch for people sitting in parked cars. New York law (VTL §1214) says nobody may open a car door into traffic unless it\'s safe. Drivers and passengers should use the Dutch reach: open the door with the far hand so you turn and look back.',
  },
  police: {
    title: 'Blocked by a parked police car',
    text: 'Police and other official vehicles regularly park in bike lanes, forcing riders out into traffic. NYC rules let you leave the bike lane when it\'s blocked: look over your shoulder, signal, and merge early instead of swerving at the last second. You can report blocked bike lanes to 311.',
  },
  truck: {
    title: 'Double-parked truck',
    text: 'Delivery trucks double-park in the bike lane all day. They hide what\'s ahead and push you into traffic, and big trucks have huge blind spots: if you can\'t see the driver in their mirror, they can\'t see you. Never squeeze along the right side of a truck that might turn.',
  },
  taxi: {
    title: 'Cut off by a cab',
    text: 'Taxis and rideshares dive into the bike lane to pick up and drop off passengers, often without signaling. Watch for blinkers, brake lights, and anyone waving from the curb, and keep your hands covering the brakes.',
  },
  ped: {
    title: 'Pedestrian on a phone',
    text: 'People step out from between parked cars, usually looking at their phones instead of the bike lane. Slow down near crosswalks and bus stops, ring your bell early, and always yield to pedestrians in the crosswalk.',
  },
  ebike: {
    title: 'Wrong-way e-bike',
    text: 'Riders going the wrong way in the bike lane close the gap much faster than you expect. Always ride with traffic, keep right when someone comes at you, and use your bell or voice early.',
  },
  pothole: {
    title: 'Pothole!',
    text: 'Potholes, steel plates, and sunken grates can send you over the handlebars. Scan the pavement ahead, not just the traffic, keep a firm grip, and report potholes to 311.',
  },
  car: {
    title: 'Stuck in traffic',
    text: 'The general traffic lane isn\'t an escape hatch: it\'s full of cars and buses that stop without warning. If you have to leave the bike lane, look over your shoulder, signal, and merge with plenty of space, then get back in the bike lane as soon as it\'s clear.',
  },
}

const BEST_KEY = 'bike-lane-dash-best'

// [one, many]
const DODGE_LABELS = {
  door: ['car door', 'car doors'],
  police: ['cop car', 'cop cars'],
  truck: ['truck', 'trucks'],
  taxi: ['cab', 'cabs'],
  ped: ['pedestrian', 'pedestrians'],
  ebike: ['wrong-way rider', 'wrong-way riders'],
  pothole: ['pothole', 'potholes'],
  car: ['stopped car', 'stopped cars'],
}

// shown on the crash screen; every figure links to where it came from
const STATS = [
  {
    value: '30',
    label: 'people killed riding bikes in NYC in 2023, the most in over two decades',
    source: 'https://www.fox5ny.com/news/nyc-cyclist-deaths-hit-record-high-in-2023',
  },
  {
    value: '23',
    label: 'of those 30 were riding e-bikes',
    source: 'https://www.cbsnews.com/newyork/news/in-response-to-numerous-fatalities-new-york-city-launches-e-bike-safety-education-campaign',
  },
  {
    value: '~400',
    label: 'bike riders seriously injured in NYC that same year',
    source: 'https://bronx.news12.com/record-levels-of-cyclist-deaths-for-nyc-in-2023',
  },
  {
    value: '1.9%',
    label: 'of 76,000+ complaints to 311 about blocked bike lanes ended in a ticket (2016 to 2023)',
    source: 'https://usa.streetsblog.org/2023/04/06/cops-in-nyc-ticket-only-2-of-blocked-bike-lanes-analysis',
  },
]

const RULES = [
  {
    cite: 'NY VTL §1214',
    text: 'No one may open a car door on the traffic side until it is reasonably safe to do so.',
    source: 'https://newyork.public.law/laws/n.y._vehicle_and_traffic_law_section_1214',
  },
  {
    cite: '34 RCNY §4-08(e)',
    text: 'Stopping, standing, or parking in a bike lane is banned, with a $115 fine.',
    source: 'https://usa.streetsblog.org/2023/04/06/cops-in-nyc-ticket-only-2-of-blocked-bike-lanes-analysis',
  },
  {
    cite: '34 RCNY §4-12(p)',
    text: 'Ride in the bike lane when there is one, but you may leave it to turn or when it is blocked or unsafe.',
    source: 'https://nyc.gov/html/dot/downloads/pdf/bicyclerules-english.pdf',
  },
  {
    cite: 'NY VTL §1236',
    text: 'At night you need a white front light visible from 500 ft and a red or amber rear light visible from 300 ft.',
    source: 'https://newyork.public.law/laws/n.y._vehicle_&_traffic_law_section_1236',
  },
  {
    cite: 'NY VTL §1238',
    text: 'Riders and passengers under 14 must wear a helmet.',
    source: 'https://healthweb-back.health.ny.gov/publications/0667.pdf',
  },
  {
    cite: 'NYC e-bike limit',
    text: 'Since October 24, 2025, e-bikes and e-scooters are capped at 15 mph citywide.',
    source: 'https://ny1.com/nyc/all-boroughs/news/2025/10/24/new-15-mile-per-hour-speed-limit-for-e-bikes-goes-into-effect',
  },
]

// set up in onMounted; the template calls these
let startGame = () => {}
let onKey = () => {}

let cleanup = () => {}

function start() {
  startGame()
}

function handleKey(e) {
  onKey(e)
}

onUnmounted(() => {
  cleanup()
})

onMounted(() => {
  const canvas = canvasRef.value
  const wrap = wrapRef.value
  const ctx = canvas.getContext('2d')

  // world units are roughly meters; +z points down the street, away from you
  const LW = 1.3 // lane width
  const LANES = [-LW, 0, LW] // 0: door zone, 1: bike lane, 2: traffic lane
  const PARK_X = -2 * LW // parked car centerline
  const CURB_L = PARK_X - 0.75
  const CURB_R = 2.6 * LW
  const SIDEWALK = 2.4
  const D0 = 4 // camera distance behind the rider
  const Z_NEAR = -D0 + 0.7
  const Z_FAR = 85
  const SPAWN_Z = 78

  const START_SPEED = 12
  const MAX_SPEED = 30
  const ACCEL = 0.22 // units/s per second
  const BLOCK_LEN = 80 // a Manhattan street block is ~80m
  const DOOR_LEN = 0.95

  let W = 0
  let H = 0
  let focal = 1
  let horizonY = 0
  let camH = 2.4
  let camX = 0

  let rafId = 0
  let last = 0
  let time = 0
  let onScreen = true

  // 'ready' | 'playing' | 'crashing' | 'over'
  let mode = 'ready'
  let crashTimer = 0
  let speed = START_SPEED
  let dist = 0
  let bagelCount = 0
  let rowAccum = 0
  let nextGap = 0
  let dodgeCounts = {}

  const player = { lane: 1, x: 0, y: 0, vy: 0, lean: 0, pedal: 0 }

  let parked = []
  let buildings = []
  let lamps = []
  let obstacles = []
  let treats = []
  let skyline = []

  // scenery generators: z where the next item starts
  let parkFront = 0
  let lampFront = 0
  const buildFront = { '-1': 0, '1': 0 }

  let bestDist = 0
  try {
    bestDist = Number(localStorage.getItem(BEST_KEY)) || 0
  } catch {}
  best.value = (bestDist / BLOCK_LEN).toFixed(1)

  const rand = (a, b) => a + Math.random() * (b - a)
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)]
  const chance = (p) => Math.random() < p
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v))

  const shadeCache = new Map()
  // scale a #rrggbb color's brightness by k
  function shade(hex, k) {
    const key = hex + k
    let out = shadeCache.get(key)
    if (!out) {
      const n = parseInt(hex.slice(1), 16)
      const c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => clamp(Math.round(v * k), 0, 255))
      out = `rgb(${c[0]},${c[1]},${c[2]})`
      shadeCache.set(key, out)
    }
    return out
  }

  function resize() {
    const rect = canvas.getBoundingClientRect()
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    W = rect.width
    H = rect.height
    canvas.width = Math.round(W * dpr)
    canvas.height = Math.round(H * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    // one lane spans ~1/4.2 of the view at the rider; ground at the rider sits at 86% height
    focal = (Math.min(W, H * 1.2) * D0) / (LW * 4.2)
    horizonY = H * 0.34
    camH = ((H * 0.86 - horizonY) * D0) / focal
  }

  // perspective projection; s is pixels per world unit at that depth
  function proj(x, y, z) {
    const s = focal / (z + D0)
    return { x: W / 2 + (x - camX) * s, y: horizonY + (camH - y) * s, s }
  }

  function quad(pts, fill) {
    ctx.beginPath()
    ctx.moveTo(pts[0].x, pts[0].y)
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
    ctx.closePath()
    ctx.fillStyle = fill
    ctx.fill()
  }

  // axis-aligned box; returns the projected near face for decorating
  function drawBox(x, y0, z, w, h, d, front, side, top) {
    const zf = z + d
    if (zf <= Z_NEAR) return null
    const zn = Math.max(z, Z_NEAR)
    const l = x - w / 2
    const r = x + w / 2
    const t = y0 + h
    const nlb = proj(l, y0, zn)
    const nrb = proj(r, y0, zn)
    const nlt = proj(l, t, zn)
    const nrt = proj(r, t, zn)
    const flb = proj(l, y0, zf)
    const frb = proj(r, y0, zf)
    const flt = proj(l, t, zf)
    const frt = proj(r, t, zf)
    if (top && t < camH) quad([nlt, nrt, frt, flt], top)
    if (camX > r) quad([nrb, nrt, frt, frb], side)
    else if (camX < l) quad([nlb, nlt, flt, flb], side)
    quad([nlb, nrb, nrt, nlt], front)
    return { x0: nlb.x, x1: nrb.x, y0: nlt.y, y1: nlb.y, s: nlb.s, cut: zn !== z }
  }

  // a ground-level rectangle from x0..x1, z0..z1
  function groundRect(x0, x1, z0, z1, fill) {
    z0 = Math.max(z0, Z_NEAR)
    if (z1 <= z0) return
    quad([proj(x0, 0, z0), proj(x1, 0, z0), proj(x1, 0, z1), proj(x0, 0, z1)], fill)
  }

  function fogAlpha(z) {
    return clamp((Z_FAR - z) / 18, 0, 1)
  }

  // ---------- scenery ----------

  const CAR_COLORS = ['#b8bec6', '#2e3640', '#8c1c1c', '#1f3f73', '#e9e9e9', '#4a5a3a', '#6b6f76', '#121418']
  const BRICK = ['#8b4a3a', '#a0583f', '#6d5a4f', '#7a3f32', '#5f6a73', '#8a7a64', '#4e4a52']
  const AWNINGS = ['#1f6f4a', '#9b2226', '#264653', '#bc6c25', '#3a0ca3']

  function addParked(z, door) {
    parked.push({ z, color: pick(CAR_COLORS), door, open: 0, triggered: false })
  }

  function addBuilding(side, z) {
    const d = rand(4, 9)
    const h = rand(5.5, 14)
    const cols = Math.max(1, Math.floor((d - 0.6) / 1.3))
    const rows = Math.max(1, Math.floor((h - 2.2) / 1.5))
    const lit = []
    for (let i = 0; i < rows * cols; i++) lit.push(chance(0.45))
    const w = 6
    const face = side < 0 ? CURB_L - SIDEWALK : CURB_R + SIDEWALK
    buildings.push({
      side,
      z,
      d,
      h,
      w,
      x: face + (side * w) / 2,
      face,
      rows,
      cols,
      lit,
      color: pick(BRICK),
      awning: pick(AWNINGS),
    })
    return d
  }

  function fillScenery() {
    while (parkFront < Z_FAR + 5) {
      addParked(parkFront, false)
      parkFront += 2.2 + rand(0.4, 1.3) + (chance(0.12) ? 3 : 0)
    }
    while (lampFront < Z_FAR + 5) {
      lamps.push({ z: lampFront })
      lampFront += 14
    }
    for (const side of [-1, 1]) {
      while (buildFront[side] < Z_FAR + 10) {
        buildFront[side] += addBuilding(side, buildFront[side]) + (chance(0.15) ? 3 : 0.05)
      }
    }
  }

  function makeSkyline() {
    skyline = []
    let x = -0.1
    while (x < 1.1) {
      const w = rand(0.02, 0.06)
      skyline.push({ x, w, h: rand(0.03, 0.14) })
      x += w * rand(0.7, 1.1)
    }
    // a couple of landmark-ish spires
    skyline.push({ x: 0.46, w: 0.03, h: 0.26, spire: true })
    skyline.push({ x: 0.71, w: 0.025, h: 0.2, spire: true })
  }

  // ---------- obstacles ----------

  function addObstacle(kind, lane, z, extra = {}) {
    const o = { kind, lane, x: LANES[lane], z, ...extra }
    switch (kind) {
      case 'police':
      case 'car':
      case 'taxi':
        o.w = 1.05
        o.d = 2.2
        break
      case 'truck':
        o.w = 1.15
        o.d = 4
        break
      case 'ebike':
        o.w = 0.5
        o.d = 1.4
        break
      case 'ped':
        o.w = 0.5
        o.d = 0.4
        break
      case 'pothole':
        o.w = 0.9
        o.d = 0.8
        o.jumpable = true
        break
    }
    if (kind === 'car') o.color = pick(CAR_COLORS)
    if (kind === 'ped') {
      o.shirt = pick(['#e76f51', '#2a9d8f', '#e9c46a', '#264653', '#f4a261', '#8d99ae'])
      o.walk = 0
    }
    obstacles.push(o)
    return o
  }

  function clearParked(z0, z1) {
    parked = parked.filter((c) => c.z + 2.2 < z0 || c.z > z1)
  }

  function addTreats(lane, z) {
    for (let i = 0; i < 4; i++) treats.push({ x: LANES[lane], z: z + i * 1.8, got: false })
  }

  function freeLane(blocked) {
    return pick([0, 1, 2].filter((l) => !blocked.includes(l)))
  }

  // one "row" of trouble; always leaves at least one lane clear without jumping
  function spawnRow(z) {
    const r = Math.random()
    let blocked = []
    if (r < 0.18) {
      clearParked(z - 0.8, z + 2.6)
      parked.push({ z, color: pick(CAR_COLORS), door: true, open: 0, triggered: false })
      blocked = [0]
      if (chance(0.45)) {
        addObstacle('car', 2, z - 0.5)
        blocked.push(2)
      }
      addTreats(1, z - 4)
      return
    }
    if (r < 0.34 || r < 0.48) {
      const kind = r < 0.34 ? 'police' : 'truck'
      const lane = pick([0, 1])
      addObstacle(kind, lane, z)
      blocked = [lane]
      if (lane === 0 && chance(0.4)) {
        addObstacle('car', 2, z + 0.5)
        blocked.push(2)
      } else if (lane === 1 && chance(0.4)) {
        addObstacle('pothole', pick([0, 2]), z + 1)
      }
      addTreats(freeLane(blocked), z - 3)
      return
    }
    if (r < 0.58) {
      addObstacle('taxi', 2, z, { merge: 0 })
      addTreats(0, z - 2)
      return
    }
    if (r < 0.7) {
      const target = pick([0, 1])
      clearParked(z - 1.4, z + 1.4)
      const o = addObstacle('ped', target, z, { targetX: LANES[target] })
      o.x = PARK_X
      addTreats(2, z - 3)
      return
    }
    if (r < 0.8) {
      const lane = pick([1, 2])
      // close in on you a bit, but never fast enough to catch the row in front of it
      const gap = Math.max(9, speed * 1.0)
      const extra = Math.min(6, (0.8 * gap * speed) / (SPAWN_Z - gap))
      addObstacle('ebike', lane, z, { extra })
      return
    }
    if (r < 0.9) {
      const lanes = [0, 1, 2].sort(() => Math.random() - 0.5).slice(0, 2)
      for (const l of lanes) addObstacle('pothole', l, z + rand(-0.5, 0.5))
      addTreats(freeLane(lanes), z - 2)
      return
    }
    addObstacle('car', 2, z)
    addObstacle(pick(['police', 'pothole']), 0, z + 0.3)
    addTreats(1, z - 2)
  }

  // ---------- game flow ----------

  function reset() {
    parked = []
    buildings = []
    lamps = []
    obstacles = []
    treats = []
    parkFront = Z_NEAR - 3
    lampFront = Z_NEAR + 2
    buildFront[-1] = Z_NEAR - 5
    buildFront[1] = Z_NEAR - 8
    fillScenery()
    player.lane = 1
    player.x = 0
    player.y = 0
    player.vy = 0
    player.lean = 0
    camX = 0
    speed = START_SPEED
    dist = 0
    bagelCount = 0
    rowAccum = 0
    nextGap = speed * 2.5 // a couple of calm seconds to start
    dodgeCounts = {}
    blocks.value = '0.0'
    bagels.value = 0
  }

  startGame = () => {
    reset()
    mode = 'playing'
    phase.value = 'playing'
    lesson.value = null
    wrap.focus({ preventScroll: true })
  }

  function crash(kind) {
    mode = 'crashing'
    crashTimer = 0.8
    lesson.value = { kind, ...LESSONS[kind] }
    dodged.value = Object.entries(dodgeCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([k, count]) => ({ label: DODGE_LABELS[k][count === 1 ? 0 : 1], count }))
    if (dist > bestDist) {
      bestDist = dist
      try {
        localStorage.setItem(BEST_KEY, String(Math.floor(bestDist)))
      } catch {}
    }
    best.value = (bestDist / BLOCK_LEN).toFixed(1)
  }

  function moveLane(dir) {
    if (mode !== 'playing') return
    player.lane = clamp(player.lane + dir, 0, 2)
  }

  function jump() {
    if (mode !== 'playing' || player.y > 0.01) return
    player.vy = 7.2
  }

  onKey = (e) => {
    const k = e.key
    if (mode === 'playing') {
      if (k === 'ArrowLeft' || k === 'a' || k === 'A') moveLane(-1)
      else if (k === 'ArrowRight' || k === 'd' || k === 'D') moveLane(1)
      else if (k === 'ArrowUp' || k === 'w' || k === 'W' || k === ' ') jump()
      else return
      e.preventDefault()
    } else if ((mode === 'ready' || mode === 'over') && (k === ' ' || k === 'Enter') && e.target === wrap) {
      e.preventDefault()
      startGame()
    }
  }

  // swipes, or taps on the left/middle/right of the screen
  let down = null
  function onPointerDown(e) {
    if (mode !== 'playing') return
    down = { x: e.clientX, y: e.clientY }
  }
  function onPointerUp(e) {
    if (mode !== 'playing' || !down) return
    const dx = e.clientX - down.x
    const dy = e.clientY - down.y
    down = null
    if (Math.abs(dx) > 30 && Math.abs(dx) > Math.abs(dy)) moveLane(dx > 0 ? 1 : -1)
    else if (dy < -30) jump()
    else if (Math.abs(dx) < 15 && Math.abs(dy) < 15) {
      const rect = canvas.getBoundingClientRect()
      const fx = (e.clientX - rect.left) / rect.width
      if (fx < 0.33) moveLane(-1)
      else if (fx > 0.67) moveLane(1)
      else jump()
    }
  }

  // ---------- update ----------

  function overlaps(a0, a1, b0, b1) {
    return a0 < b1 && b0 < a1
  }

  function update(dt) {
    time += dt

    if (mode === 'crashing') {
      crashTimer -= dt
      if (crashTimer <= 0) {
        mode = 'over'
        phase.value = 'crashed'
      }
      return
    }
    if (mode === 'over') return

    const v = mode === 'playing' ? speed : 6
    const dz = v * dt
    if (mode === 'playing') {
      speed = Math.min(MAX_SPEED, speed + ACCEL * dt)
      dist += dz
    }

    for (const list of [parked, buildings, lamps, obstacles, treats]) {
      for (const o of list) o.z -= dz
    }
    parkFront -= dz
    lampFront -= dz
    buildFront[-1] -= dz
    buildFront[1] -= dz
    if (mode === 'playing') {
      for (const o of obstacles) {
        if (!o.passed && o.z + o.d < -0.5) {
          o.passed = true
          dodgeCounts[o.kind] = (dodgeCounts[o.kind] || 0) + 1
        }
      }
      for (const c of parked) {
        if (c.door && !c.passed && c.z + 2.2 < -0.5) {
          c.passed = true
          dodgeCounts.door = (dodgeCounts.door || 0) + 1
        }
      }
    }
    parked = parked.filter((c) => c.z + 2.2 > Z_NEAR - 1)
    buildings = buildings.filter((b) => b.z + b.d > Z_NEAR - 1)
    lamps = lamps.filter((l) => l.z > Z_NEAR - 1)
    obstacles = obstacles.filter((o) => o.z + o.d > Z_NEAR - 1)
    treats = treats.filter((t) => t.z > Z_NEAR - 1 && !t.got)
    fillScenery()

    if (mode === 'playing') {
      rowAccum += dz
      if (rowAccum >= nextGap) {
        rowAccum = 0
        nextGap = Math.max(9, speed * rand(1.0, 1.5))
        spawnRow(SPAWN_Z)
      }
    }

    // obstacles that move or react as you get close
    for (const o of obstacles) {
      if (o.kind === 'ebike') o.z -= o.extra * dt
      if (o.kind === 'taxi' && o.z < speed * 1.7) {
        o.merge = Math.min(1, o.merge + dt / 0.7)
        o.x = LANES[2] + (LANES[1] - LANES[2]) * (1 - (1 - o.merge) ** 2)
      }
      if (o.kind === 'ped' && o.z < speed * 2.3 && o.x < o.targetX) {
        o.x = Math.min(o.targetX, o.x + 2.1 * dt)
        o.walk += dt * 8
      }
    }
    for (const c of parked) {
      if (c.door && c.z < speed * 0.85) c.open = Math.min(1, c.open + dt / 0.2)
    }

    // rider
    const tx = LANES[player.lane]
    const step = 12 * dt
    player.x += clamp(tx - player.x, -step, step)
    player.lean += (clamp((tx - player.x) * -0.35, -0.3, 0.3) - player.lean) * Math.min(1, dt * 12)
    if (player.y > 0 || player.vy > 0) {
      player.vy -= 24 * dt
      player.y = Math.max(0, player.y + player.vy * dt)
      if (player.y === 0) player.vy = 0
    }
    player.pedal += dz * 1.4
    camX += (player.x * 0.6 - camX) * Math.min(1, dt * 6)

    if (mode !== 'playing') return

    // collisions: rider occupies x±0.28, z -0.5..0.5
    const px0 = player.x - 0.28
    const px1 = player.x + 0.28
    for (const o of obstacles) {
      if (o.jumpable && player.y > 0.25) continue
      if (overlaps(o.z, o.z + o.d, -0.5, 0.5) && overlaps(o.x - o.w / 2, o.x + o.w / 2, px0, px1)) {
        crash(o.kind)
        return
      }
    }
    for (const c of parked) {
      if (!c.door || c.open < 0.3) continue
      const { hx, hz, ex, ez } = doorGeom(c)
      if (overlaps(ez, hz, -0.5, 0.5) && overlaps(hx, ex, px0, px1)) {
        crash('door')
        return
      }
    }
    for (const t of treats) {
      if (!t.got && Math.abs(t.z) < 0.6 && Math.abs(t.x - player.x) < 0.6) {
        t.got = true
        bagelCount++
        bagels.value = bagelCount
      }
    }
    blocks.value = (dist / BLOCK_LEN).toFixed(1)
  }

  // the passenger door hinges at its front edge and swings out into the door zone
  function doorGeom(c) {
    const a = c.open * 1.2
    const hx = PARK_X + 0.525
    const hz = c.z + 1.45
    return { hx, hz, ex: hx + DOOR_LEN * Math.sin(a), ez: hz - DOOR_LEN * Math.cos(a) }
  }

  // ---------- drawing ----------

  function drawSky() {
    const g = ctx.createLinearGradient(0, 0, 0, horizonY)
    g.addColorStop(0, '#272a5a')
    g.addColorStop(0.6, '#8a4f7d')
    g.addColorStop(1, '#f2a46d')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, W, horizonY + 1)

    ctx.fillStyle = '#4a3f6b'
    for (const b of skyline) {
      const x = b.x * W
      const h = b.h * H
      ctx.fillRect(x, horizonY - h, b.w * W, h + 1)
      if (b.spire) ctx.fillRect(x + b.w * W * 0.4, horizonY - h - H * 0.04, b.w * W * 0.2, H * 0.04)
    }
  }

  function drawGround() {
    const far = Z_FAR + 10
    groundRect(-60, CURB_L, Z_NEAR, far, '#8d8a86') // left sidewalk
    groundRect(CURB_R, 60, Z_NEAR, far, '#8d8a86') // right sidewalk
    groundRect(CURB_L, CURB_R, Z_NEAR, far, '#3b3d42') // asphalt
    groundRect(-1.5 * LW, 0.5 * LW, Z_NEAR, far, '#2f7d4f') // green bike lane
    groundRect(CURB_L - 0.12, CURB_L, Z_NEAR, far, '#b5b1aa') // curbs
    groundRect(CURB_R, CURB_R + 0.12, Z_NEAR, far, '#b5b1aa')
    groundRect(-1.5 * LW - 0.06, -1.5 * LW + 0.06, Z_NEAR, far, '#e8e8e8') // bike lane edges
    groundRect(0.5 * LW - 0.06, 0.5 * LW + 0.06, Z_NEAR, far, '#e8e8e8')

    // dashes split the bike lane and the traffic lanes; they scroll with distance
    const off = (dist + time * (mode === 'ready' ? 6 : 0)) % 6
    for (let z = -off - 6; z < far; z += 6) {
      groundRect(-0.5 * LW - 0.03, -0.5 * LW + 0.03, z, z + 2, 'rgba(255,255,255,0.45)')
      groundRect(1.5 * LW - 0.05, 1.5 * LW + 0.05, z, z + 3, '#d8d8d8')
    }
    // bike stencils every 24m in the bike lane
    const soff = (dist + time * (mode === 'ready' ? 6 : 0)) % 24
    for (let z = -soff; z < far; z += 24) drawStencil(0, z + 10)

    // haze at the horizon hides things popping in
    const g = ctx.createLinearGradient(0, horizonY, 0, horizonY + H * 0.12)
    g.addColorStop(0, 'rgba(242,164,109,0.9)')
    g.addColorStop(1, 'rgba(242,164,109,0)')
    ctx.fillStyle = g
    ctx.fillRect(0, horizonY, W, H * 0.12)
  }

  function drawStencil(x, z) {
    if (z < Z_NEAR + 1) return
    const a = proj(x, 0, z)
    const b = proj(x, 0, z + 1.4)
    const rx = 0.3 * a.s
    const ry = Math.max(1, (a.y - b.y) / 4)
    ctx.strokeStyle = 'rgba(255,255,255,0.75)'
    ctx.lineWidth = Math.max(1, 0.05 * a.s)
    ctx.beginPath()
    ctx.ellipse(a.x - rx, (a.y + b.y) / 2, rx * 0.6, ry, 0, 0, Math.PI * 2)
    ctx.ellipse(a.x + rx, (a.y + b.y) / 2, rx * 0.6, ry, 0, 0, Math.PI * 2)
    ctx.stroke()
  }

  function drawBuilding(b) {
    if (b.z > Z_FAR + 5) return
    ctx.globalAlpha = fogAlpha(b.z)
    drawBox(b.x, 0, b.z, b.w, b.h, b.d, shade(b.color, 0.7), b.color, shade(b.color, 0.55))
    const fx = b.face
    // awning along the storefronts
    const z0 = Math.max(b.z + 0.2, Z_NEAR)
    const z1 = b.z + b.d - 0.2
    if (z1 > z0) {
      quad([proj(fx, 0, z0), proj(fx, 1.1, z0), proj(fx, 1.1, z1), proj(fx, 0, z1)], '#2a3442')
      quad([proj(fx, 1.1, z0), proj(fx, 1.5, z0), proj(fx, 1.5, z1), proj(fx, 1.1, z1)], b.awning)
    }
    if (b.z < 60) {
      for (let row = 0; row < b.rows; row++) {
        for (let col = 0; col < b.cols; col++) {
          const wz0 = b.z + 0.5 + col * 1.3
          const wz1 = wz0 + 0.7
          if (wz1 <= Z_NEAR) continue
          const y = 2.2 + row * 1.5
          const zz0 = Math.max(wz0, Z_NEAR)
          quad(
            [proj(fx, y, zz0), proj(fx, y + 0.9, zz0), proj(fx, y + 0.9, wz1), proj(fx, y, wz1)],
            b.lit[row * b.cols + col] ? '#ffd27a' : '#2b2f3a',
          )
        }
      }
    }
    ctx.globalAlpha = 1
  }

  function drawLamp(l) {
    const x = CURB_L - 0.4
    drawBox(x, 0, l.z, 0.12, 3.4, 0.12, '#2d3138', '#24272c', '#2d3138')
    if (l.z > Z_NEAR + 0.5) {
      const p = proj(x + 0.35, 3.35, l.z)
      const r = Math.max(2, 0.5 * p.s)
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r)
      g.addColorStop(0, 'rgba(255,230,160,0.95)')
      g.addColorStop(1, 'rgba(255,230,160,0)')
      ctx.fillStyle = g
      ctx.fillRect(p.x - r, p.y - r, r * 2, r * 2)
    }
  }

  function wheelPair(x, z, w) {
    if (z <= Z_NEAR) return
    ctx.fillStyle = '#111'
    for (const sx of [-1, 1]) {
      const cx = x + sx * (w / 2 - 0.12)
      const a = proj(cx - 0.1, 0.38, z)
      const b = proj(cx + 0.1, 0, z)
      ctx.fillRect(a.x, a.y, b.x - a.x, b.y - a.y)
    }
  }

  // tail light rectangles on a car's rear face
  function tailLights(f, color) {
    if (!f || f.cut) return
    const s = f.s
    ctx.fillStyle = color
    const h = 0.1 * s
    const y = f.y1 - 0.42 * s
    ctx.fillRect(f.x0 + 0.04 * s, y, 0.16 * s, h)
    ctx.fillRect(f.x1 - 0.2 * s, y, 0.16 * s, h)
  }

  function drawCar(x, z, body, opts = {}) {
    wheelPair(x, z + 0.05, 1.05)
    const f = drawBox(x, 0.28, z, 1.05, 0.55, 2.2, shade(body, 0.92), shade(body, 0.72), shade(body, 1.08))
    const c = drawBox(x, 0.83, z + 0.55, 0.9, 0.45, 1.15, '#33414f', shade(body, 0.6), shade(body, 1.02))
    tailLights(f, opts.brake ? '#ff2a1a' : '#8a1d16')
    if (opts.passenger && c && !c.cut) {
      // someone sitting in the car: a cue that a door might open
      const s = c.s
      ctx.fillStyle = '#121820'
      ctx.beginPath()
      ctx.arc(c.x1 - 0.22 * s, c.y1 - 0.18 * s, 0.11 * s, 0, Math.PI * 2)
      ctx.fill()
    }
    return { f, c }
  }

  function drawParked(c) {
    ctx.globalAlpha = fogAlpha(c.z)
    drawCar(PARK_X, c.z, c.color, { passenger: c.door })
    if (c.door && c.open > 0) {
      const { hx, hz, ex, ez } = doorGeom(c)
      if (ez > Z_NEAR) {
        quad([proj(hx, 0.3, hz), proj(ex, 0.3, ez), proj(ex, 0.85, ez), proj(hx, 0.85, hz)], shade(c.color, 1.05))
        quad([proj(hx, 0.85, hz), proj(ex, 0.85, ez), proj(ex, 1.2, ez), proj(hx, 1.2, hz)], '#55697d')
      }
    }
    ctx.globalAlpha = 1
  }

  function flashing(rate) {
    return Math.floor(time * rate) % 2 === 0
  }

  function glow(x, y, r, color) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r)
    g.addColorStop(0, color)
    g.addColorStop(1, 'rgba(0,0,0,0)')
    ctx.fillStyle = g
    ctx.fillRect(x - r, y - r, r * 2, r * 2)
  }

  function drawPolice(o) {
    const { f } = drawCar(o.x, o.z, '#f1f3f5', { brake: false })
    const bar = drawBox(o.x, 1.28, o.z + 1.0, 0.7, 0.1, 0.25, '#222', '#222', '#333')
    if (f && !f.cut) {
      const s = f.s
      ctx.fillStyle = '#1d3f8f'
      ctx.fillRect(f.x0, f.y1 - 0.25 * s, f.x1 - f.x0, 0.07 * s)
      ctx.fillStyle = '#1d3f8f'
      ctx.font = `bold ${Math.max(6, 0.15 * s)}px system-ui, sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('NYPD', (f.x0 + f.x1) / 2, f.y0 + 0.2 * s)
    }
    if (bar && !bar.cut) {
      const s = bar.s
      const on = flashing(6)
      const mid = (bar.x0 + bar.x1) / 2
      const y = (bar.y0 + bar.y1) / 2
      ctx.fillStyle = on ? '#ff2a2a' : '#5a1010'
      ctx.fillRect(bar.x0, bar.y0, mid - bar.x0, bar.y1 - bar.y0)
      ctx.fillStyle = on ? '#14204a' : '#3a7bff'
      ctx.fillRect(mid, bar.y0, bar.x1 - mid, bar.y1 - bar.y0)
      glow(on ? bar.x0 + (mid - bar.x0) / 2 : mid + (bar.x1 - mid) / 2, y, 0.9 * s, on ? 'rgba(255,40,40,0.55)' : 'rgba(60,120,255,0.55)')
    }
  }

  function drawTaxi(o) {
    const { f } = drawCar(o.x, o.z, '#f4c20d', { brake: o.merge > 0 })
    const sign = drawBox(o.x, 1.28, o.z + 1.0, 0.4, 0.14, 0.2, '#fff6cc', '#e0d9b0', '#fff')
    if (f && !f.cut && o.z < speed * 1.9 && flashing(4)) {
      // left blinker: it's coming into the bike lane
      const s = f.s
      ctx.fillStyle = '#ffb000'
      ctx.fillRect(f.x0 + 0.04 * s, f.y1 - 0.3 * s, 0.16 * s, 0.07 * s)
      glow(f.x0 + 0.12 * s, f.y1 - 0.27 * s, 0.4 * s, 'rgba(255,176,0,0.6)')
    }
    if (sign && !sign.cut) glow((sign.x0 + sign.x1) / 2, sign.y0, 0.4 * sign.s, 'rgba(255,240,180,0.4)')
  }

  function drawTruck(o) {
    wheelPair(o.x, o.z + 0.2, 1.15)
    drawBox(o.x, 0.3, o.z + 3.2, 1.1, 1.5, 1.2, '#d9d9d9', '#b5b5b5', '#e5e5e5') // cab
    const f = drawBox(o.x, 0.45, o.z, 1.15, 2.2, 3.3, '#ececec', '#c9c9c9', '#f5f5f5')
    if (f && !f.cut) {
      const s = f.s
      // roll-up door
      ctx.strokeStyle = 'rgba(0,0,0,0.15)'
      ctx.lineWidth = Math.max(1, 0.02 * s)
      for (let i = 1; i < 8; i++) {
        const y = f.y0 + ((f.y1 - f.y0) * i) / 8
        ctx.beginPath()
        ctx.moveTo(f.x0 + 0.08 * s, y)
        ctx.lineTo(f.x1 - 0.08 * s, y)
        ctx.stroke()
      }
      ctx.fillStyle = '#c0392b'
      ctx.fillRect(f.x0, f.y1 - 0.1 * s, f.x1 - f.x0, 0.1 * s)
      const on = flashing(3)
      ctx.fillStyle = on ? '#ffb000' : '#6b4a00'
      ctx.fillRect(f.x0 + 0.04 * s, f.y1 - 0.3 * s, 0.14 * s, 0.12 * s)
      ctx.fillRect(f.x1 - 0.18 * s, f.y1 - 0.3 * s, 0.14 * s, 0.12 * s)
      if (on) {
        glow(f.x0 + 0.11 * s, f.y1 - 0.24 * s, 0.45 * s, 'rgba(255,176,0,0.5)')
        glow(f.x1 - 0.11 * s, f.y1 - 0.24 * s, 0.45 * s, 'rgba(255,176,0,0.5)')
      }
    }
  }

  function drawPothole(o) {
    if (o.z + o.d <= Z_NEAR + 0.2) return
    const a = proj(o.x, 0, Math.max(o.z, Z_NEAR))
    const b = proj(o.x, 0, o.z + o.d)
    const cy = (a.y + b.y) / 2
    const ry = Math.max(1, (a.y - b.y) / 2)
    const rx = (o.w / 2) * ((a.s + b.s) / 2)
    ctx.fillStyle = '#5c5e63'
    ctx.beginPath()
    ctx.ellipse(a.x, cy, rx * 1.1, ry * 1.15, 0, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#141416'
    ctx.beginPath()
    ctx.ellipse(a.x, cy + ry * 0.1, rx * 0.85, ry * 0.8, 0, 0, Math.PI * 2)
    ctx.fill()
  }

  // person facing the camera, drawn as a flat billboard
  function drawPed(o) {
    if (o.z <= Z_NEAR + 0.3) return
    const p = proj(o.x, 0, o.z)
    const s = p.s
    const swing = Math.sin(o.walk) * 0.12
    ctx.strokeStyle = '#2b2d42'
    ctx.lineCap = 'round'
    ctx.lineWidth = 0.12 * s
    ctx.beginPath()
    ctx.moveTo(p.x - 0.08 * s, p.y - 0.85 * s)
    ctx.lineTo(p.x - 0.1 * s + swing * s, p.y)
    ctx.moveTo(p.x + 0.08 * s, p.y - 0.85 * s)
    ctx.lineTo(p.x + 0.1 * s - swing * s, p.y)
    ctx.stroke()
    ctx.fillStyle = o.shirt
    ctx.fillRect(p.x - 0.2 * s, p.y - 1.45 * s, 0.4 * s, 0.65 * s)
    ctx.fillStyle = '#c68e6b'
    ctx.beginPath()
    ctx.arc(p.x, p.y - 1.58 * s, 0.13 * s, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#2b1d14'
    ctx.beginPath()
    ctx.arc(p.x, p.y - 1.63 * s, 0.13 * s, Math.PI, 0)
    ctx.fill()
    // head down, staring at a phone
    ctx.fillStyle = '#dff3ff'
    ctx.fillRect(p.x + 0.04 * s, p.y - 1.32 * s, 0.1 * s, 0.14 * s)
    glow(p.x + 0.09 * s, p.y - 1.27 * s, 0.25 * s, 'rgba(200,235,255,0.5)')
  }

  // delivery rider coming straight at you
  function drawEbike(o) {
    if (o.z <= Z_NEAR + 0.3) return
    const p = proj(o.x, 0, o.z)
    const s = p.s
    ctx.fillStyle = '#111'
    ctx.fillRect(p.x - 0.05 * s, p.y - 0.68 * s, 0.1 * s, 0.68 * s)
    ctx.strokeStyle = '#222'
    ctx.lineWidth = 0.05 * s
    ctx.beginPath()
    ctx.moveTo(p.x - 0.3 * s, p.y - 1.1 * s)
    ctx.lineTo(p.x + 0.3 * s, p.y - 1.1 * s)
    ctx.stroke()
    // insulated delivery bag
    ctx.fillStyle = '#1aa38a'
    ctx.fillRect(p.x - 0.28 * s, p.y - 2.05 * s, 0.56 * s, 0.5 * s)
    ctx.fillStyle = '#324a5f'
    ctx.fillRect(p.x - 0.2 * s, p.y - 1.55 * s, 0.4 * s, 0.5 * s)
    ctx.fillStyle = '#9c6644'
    ctx.beginPath()
    ctx.arc(p.x, p.y - 1.66 * s, 0.12 * s, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#e63946'
    ctx.beginPath()
    ctx.arc(p.x, p.y - 1.7 * s, 0.13 * s, Math.PI, 0)
    ctx.fill()
    ctx.fillStyle = '#fffbe6'
    ctx.beginPath()
    ctx.arc(p.x, p.y - 0.8 * s, 0.07 * s, 0, Math.PI * 2)
    ctx.fill()
    glow(p.x, p.y - 0.8 * s, 0.7 * s, 'rgba(255,250,220,0.6)')
  }

  function drawTreat(t) {
    if (t.z <= Z_NEAR + 0.3 || t.got) return
    const p = proj(t.x, 0.55 + Math.sin(time * 4 + t.z) * 0.06, t.z)
    const s = p.s
    ctx.strokeStyle = '#c98a45'
    ctx.lineWidth = 0.11 * s
    ctx.beginPath()
    ctx.ellipse(p.x, p.y, 0.2 * s, 0.15 * s, 0, 0, Math.PI * 2)
    ctx.stroke()
    ctx.strokeStyle = '#e6b77a'
    ctx.lineWidth = 0.04 * s
    ctx.beginPath()
    ctx.ellipse(p.x, p.y - 0.02 * s, 0.2 * s, 0.14 * s, 0, Math.PI * 1.1, Math.PI * 1.9)
    ctx.stroke()
  }

  // you, from behind
  function drawPlayer() {
    const g = proj(player.x, 0, 0)
    const p = proj(player.x, player.y, 0)
    const s = p.s
    ctx.fillStyle = 'rgba(0,0,0,0.35)'
    ctx.beginPath()
    ctx.ellipse(g.x, g.y, 0.32 * s, 0.07 * s, 0, 0, Math.PI * 2)
    ctx.fill()

    ctx.save()
    ctx.translate(p.x, p.y)
    if (mode === 'crashing' || mode === 'over') ctx.rotate(0.9)
    else ctx.rotate(player.lean)

    ctx.fillStyle = '#151515'
    ctx.fillRect(-0.045 * s, -0.7 * s, 0.09 * s, 0.7 * s)
    ctx.fillStyle = flashing(4) ? '#ff3b30' : '#6e1a16'
    ctx.fillRect(-0.06 * s, -0.66 * s, 0.12 * s, 0.07 * s)

    const k = Math.sin(player.pedal) * 0.12
    ctx.strokeStyle = '#1f2a37'
    ctx.lineCap = 'round'
    ctx.lineWidth = 0.11 * s
    ctx.beginPath()
    ctx.moveTo(-0.1 * s, -1.02 * s)
    ctx.lineTo(-0.13 * s, (-0.42 + k) * s)
    ctx.moveTo(0.1 * s, -1.02 * s)
    ctx.lineTo(0.13 * s, (-0.42 - k) * s)
    ctx.stroke()

    ctx.strokeStyle = '#222'
    ctx.lineWidth = 0.05 * s
    ctx.beginPath()
    ctx.moveTo(-0.32 * s, -1.12 * s)
    ctx.lineTo(0.32 * s, -1.12 * s)
    ctx.stroke()

    ctx.fillStyle = '#e4572e'
    ctx.beginPath()
    ctx.moveTo(-0.18 * s, -0.98 * s)
    ctx.lineTo(0.18 * s, -0.98 * s)
    ctx.lineTo(0.24 * s, -1.55 * s)
    ctx.lineTo(-0.24 * s, -1.55 * s)
    ctx.closePath()
    ctx.fill()

    ctx.strokeStyle = '#e4572e'
    ctx.lineWidth = 0.08 * s
    ctx.beginPath()
    ctx.moveTo(-0.22 * s, -1.5 * s)
    ctx.lineTo(-0.3 * s, -1.13 * s)
    ctx.moveTo(0.22 * s, -1.5 * s)
    ctx.lineTo(0.3 * s, -1.13 * s)
    ctx.stroke()

    ctx.fillStyle = '#2d3e50'
    ctx.fillRect(-0.16 * s, -1.5 * s, 0.32 * s, 0.36 * s)
    ctx.fillStyle = '#d9f99d'
    ctx.fillRect(-0.16 * s, -1.3 * s, 0.32 * s, 0.04 * s)

    ctx.fillStyle = '#f4f4f4'
    ctx.beginPath()
    ctx.ellipse(0, -1.68 * s, 0.15 * s, 0.13 * s, 0, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#1f2a37'
    ctx.fillRect(-0.15 * s, -1.66 * s, 0.3 * s, 0.03 * s)
    ctx.restore()
  }

  function drawObstacle(o) {
    if (o.z > Z_FAR) return
    ctx.globalAlpha = fogAlpha(o.z)
    switch (o.kind) {
      case 'police':
        drawPolice(o)
        break
      case 'taxi':
        drawTaxi(o)
        break
      case 'truck':
        drawTruck(o)
        break
      case 'car':
        drawCar(o.x, o.z, o.color, { brake: true })
        break
      case 'pothole':
        drawPothole(o)
        break
      case 'ped':
        drawPed(o)
        break
      case 'ebike':
        drawEbike(o)
        break
    }
    ctx.globalAlpha = 1
  }

  function draw() {
    ctx.save()
    if (mode === 'crashing') {
      const k = crashTimer * 10
      ctx.translate(rand(-k, k), rand(-k, k))
    }
    drawSky()
    drawGround()

    // potholes lie flat on the road, so draw them before anything standing up
    for (const o of obstacles) if (o.kind === 'pothole') drawObstacle(o)

    for (const side of [-1, 1]) {
      buildings
        .filter((b) => b.side === side)
        .sort((a, b) => b.z - a.z)
        .forEach(drawBuilding)
    }

    // painter's algorithm, far to near
    const items = []
    for (const l of lamps) items.push({ z: l.z, fn: () => drawLamp(l) })
    for (const c of parked) items.push({ z: c.z + 1.1, fn: () => drawParked(c) })
    for (const o of obstacles) if (o.kind !== 'pothole') items.push({ z: o.z + o.d / 2, fn: () => drawObstacle(o) })
    for (const t of treats) items.push({ z: t.z, fn: () => drawTreat(t) })
    items.push({ z: 0, fn: drawPlayer })
    items.sort((a, b) => b.z - a.z)
    for (const it of items) it.fn()

    if (mode === 'crashing') {
      ctx.fillStyle = `rgba(255,60,40,${clamp(crashTimer, 0, 0.5) * 0.6})`
      ctx.fillRect(-20, -20, W + 40, H + 40)
    }
    ctx.restore()
  }

  function frame(t) {
    const dt = Math.min(0.05, (t - last) / 1000 || 0)
    last = t
    if (onScreen && W > 0) {
      update(dt)
      draw()
    }
    rafId = requestAnimationFrame(frame)
  }

  makeSkyline()
  resize()
  reset()

  const resizeObserver = new ResizeObserver(resize)
  resizeObserver.observe(canvas)
  const visObserver = new IntersectionObserver(([entry]) => {
    onScreen = entry.isIntersecting
  })
  visObserver.observe(canvas)

  canvas.addEventListener('pointerdown', onPointerDown)
  canvas.addEventListener('pointerup', onPointerUp)

  rafId = requestAnimationFrame(frame)

  cleanup = () => {
    cancelAnimationFrame(rafId)
    resizeObserver.disconnect()
    visObserver.disconnect()
    canvas.removeEventListener('pointerdown', onPointerDown)
    canvas.removeEventListener('pointerup', onPointerUp)
  }
})
</script>

<template>
  <div
    ref="wrapRef"
    class="bike-game"
    tabindex="0"
    aria-label="Bike Lane Dash game. Use the left and right arrow keys to change lanes and up or space to hop."
    @keydown="handleKey"
  >
    <canvas ref="canvasRef"></canvas>

    <div v-if="phase === 'playing'" class="hud">
      <span>{{ blocks }} blocks</span>
      <span>🥯 {{ bagels }}</span>
    </div>

    <div v-if="phase === 'ready'" class="overlay">
      <div class="card">
        <h2>Bike Lane Dash</h2>
        <p>
          Ride a Manhattan bike lane at rush hour. Dodge cop cars, double-parked trucks, opening car doors, and
          everyone else who thinks the bike lane is theirs. Grab bagels along the way.
        </p>
        <ul class="controls">
          <li><kbd>←</kbd> <kbd>→</kbd> or swipe to change lanes</li>
          <li><kbd>↑</kbd> / <kbd>space</kbd> or swipe up to hop potholes</li>
        </ul>
        <button type="button" @click="start">Start riding</button>
      </div>
    </div>

    <div v-if="phase === 'crashed' && lesson" class="overlay">
      <div class="card card-wide">
        <p class="eyebrow">You crashed</p>
        <h2>{{ lesson.title }}</h2>
        <p>{{ lesson.text }}</p>

        <h3>Your ride</h3>
        <dl class="run">
          <div><dt>Blocks</dt><dd>{{ blocks }}</dd></div>
          <div><dt>Bagels</dt><dd>{{ bagels }}</dd></div>
          <div><dt>Best</dt><dd>{{ best }}</dd></div>
        </dl>
        <p v-if="dodged.length" class="dodged">
          You got past
          <template v-for="(d, i) in dodged" :key="d.label">
            <strong>{{ d.count }} {{ d.label }}</strong>{{ i < dodged.length - 2 ? ', ' : i === dodged.length - 2 ? ' and ' : '' }}
          </template>
          before this one.
        </p>

        <h3>By the numbers</h3>
        <ul class="facts">
          <li v-for="f in STATS" :key="f.value">
            <span class="fact-value">{{ f.value }}</span>
            <span>{{ f.label }} <a :href="f.source" target="_blank" rel="noopener">source</a></span>
          </li>
        </ul>

        <h3>The rules</h3>
        <ul class="rules">
          <li v-for="r in RULES" :key="r.cite">
            <a class="cite" :href="r.source" target="_blank" rel="noopener">{{ r.cite }}</a>
            <span>{{ r.text }}</span>
          </li>
        </ul>

        <div class="again">
          <button type="button" @click="start">Ride again</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bike-game {
  position: relative;
  width: 100%;
  height: min(640px, 78vh);
  min-height: 420px;
  margin: 1.5rem 0;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  background: #272a5a;
  outline: none;
  user-select: none;
  -webkit-user-select: none;
}

.bike-game:focus-visible {
  box-shadow: 0 0 0 2px var(--vp-c-brand-1);
}

canvas {
  display: block;
  width: 100%;
  height: 100%;
  touch-action: none;
}

.hud {
  position: absolute;
  top: 10px;
  left: 12px;
  right: 12px;
  display: flex;
  justify-content: space-between;
  font: 600 15px/1 system-ui, sans-serif;
  color: #fff;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.7);
  pointer-events: none;
}

.overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  background: rgba(10, 12, 30, 0.45);
}

.card {
  max-width: 420px;
  max-height: 100%;
  overflow-y: auto;
  padding: 20px 22px;
  border-radius: 10px;
  background: rgba(18, 20, 40, 0.88);
  color: #f2f2f2;
  font-size: 15px;
  line-height: 1.5;
}

.card-wide {
  max-width: 540px;
  padding-bottom: 0;
}

.card h3 {
  margin: 18px 0 8px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #9fd9b4;
}

.card a {
  color: #9fd9b4;
}

.run {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin: 0 0 10px;
}

.run div {
  padding: 8px 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.07);
}

.run dt {
  font-size: 12px;
  color: #b9bbd6;
}

.run dd {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.dodged {
  font-size: 14px;
}

.facts,
.rules {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 10px;
}

.facts li,
.rules li {
  margin: 0;
  display: grid;
  grid-template-columns: 4.2rem 1fr;
  gap: 12px;
  align-items: baseline;
  font-size: 14px;
  line-height: 1.45;
}

.fact-value {
  font-size: 22px;
  font-weight: 700;
  color: #ff8a73;
  font-variant-numeric: tabular-nums;
}

.rules li {
  grid-template-columns: 7.5rem 1fr;
}

.rules .cite {
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.facts a {
  font-size: 12px;
  white-space: nowrap;
}

.again {
  position: sticky;
  bottom: 0;
  margin: 16px -22px 0;
  padding: 12px 22px 16px;
  background: rgba(18, 20, 40, 0.97);
}

@media (max-width: 480px) {
  .rules li {
    grid-template-columns: 1fr;
    gap: 2px;
  }
}

.card h2 {
  margin: 0 0 8px;
  padding: 0;
  border: none;
  font-size: 22px;
  line-height: 1.25;
  color: #fff;
}

.card p {
  margin: 0 0 12px;
}

.eyebrow {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #ff8a73;
}

.stats {
  font-weight: 600;
}

.controls {
  margin: 0 0 14px;
  padding-left: 18px;
}

.controls li {
  margin: 2px 0;
}

kbd {
  display: inline-block;
  padding: 0 5px;
  border-radius: 4px;
  border: 1px solid rgba(255, 255, 255, 0.35);
  font: 12px/1.6 system-ui, sans-serif;
}

button {
  padding: 8px 18px;
  border-radius: 6px;
  background: #2f9e5b;
  color: #fff;
  font-weight: 600;
  cursor: pointer;
}

button:hover {
  background: #37b169;
}
</style>
