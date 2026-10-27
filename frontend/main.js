/**
 * Particles — gesture-controlled 3D particle system (frontend).
 *
 * Data flow:
 *   WebSocket (hand openness, hand count, hand distance @ ~30 Hz)
 *     -> control state (smoothed again at render rate)
 *     -> per-frame CPU update of particle targets (template * scale + spread)
 *     -> custom glow shader with additive blending (GPU)
 *
 * No build step: Three.js is loaded from a CDN through the import map in index.html.
 * URL options:  ?ws=ws://host:8765   ?n=24000 (particle count)
 */
import * as THREE from 'three';

// ===========================================================================
// Configuration
// ===========================================================================
const params = new URLSearchParams(location.search);
const PARTICLE_COUNT = clampInt(params.get('n'), 5000, 60000, 24000);
const WS_URL = params.get('ws') || `ws://${location.hostname || 'localhost'}:8765`;

const BREATH_PHASES = [            // 4-4-6 breathing pattern (seconds)
  { name: 'Inhale', duration: 4 },
  { name: 'Hold', duration: 4 },
  { name: 'Exhale', duration: 6 },
];
const BREATH_CYCLE = BREATH_PHASES.reduce((s, p) => s + p.duration, 0);

const BACKEND_LABELS = {
  cpu: 'CPU · MediaPipe',
  hailo: 'UGen300 · Hailo-10H',
  mock: 'Mock · synthetic',
};

// ===========================================================================
// Renderer, scene, camera
// ===========================================================================
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 300);
camera.position.set(0, 0, 6.5);

// ===========================================================================
// Background: faint, slowly rotating starfield
// ===========================================================================
function createStarfield(count = 2000) {
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const r = 50 + Math.random() * 60;
    const [x, y, z] = randomUnitVector();
    pos.set([x * r, y * r, z * r], i * 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({
    color: 0xffffff, size: 0.16, sizeAttenuation: true,
    transparent: true, opacity: 0.35, depthWrite: false,
  });
  return new THREE.Points(geo, mat);
}
const stars = createStarfield();
scene.add(stars);

// ===========================================================================
// Shape templates — each returns N target positions (x, y, z)
// ===========================================================================

/** 3D "puffy" heart: classic parametric outline, inflated along z. */
function makeHeart(n) {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const t = Math.random() * Math.PI * 2;
    const s = Math.sqrt(Math.random());           // uniform over the 2D area
    const x = 16 * Math.sin(t) ** 3;
    const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
    const depth = 0.55 * Math.sqrt(Math.max(0, 1 - s * s));  // thick center, thin edge
    // 75% on the surface (crisp shape), 25% inside the volume (soft glow).
    const z = Math.random() < 0.75
      ? (Math.random() < 0.5 ? -1 : 1) * depth + gauss() * 0.02
      : (Math.random() * 2 - 1) * depth;
    out[i * 3] = (x * s / 17) * 1.7;
    out[i * 3 + 1] = ((y * s + 2.5) / 17) * 1.7;
    out[i * 3 + 2] = z;
  }
  return out;
}

/** Flower: two layers of cupped petals, a seed head, a stem and a leaf. */
function makeFlower(n) {
  const out = new Float32Array(n * 3);
  const headY = 0.75;
  const petal = (R, k, rot, cup) => {
    const th = Math.random() * Math.PI * 2;
    const env = Math.sqrt(Math.abs(Math.cos((k / 2) * (th + rot))));  // k rounded petals
    const r = R * env * Math.sqrt(Math.random());
    const rr = r / R;
    return [r * Math.cos(th), headY + r * Math.sin(th), cup * rr * rr + gauss() * 0.025];
  };
  for (let i = 0; i < n; i++) {
    const pick = Math.random();
    let p;
    if (pick < 0.42) {
      p = petal(1.45, 5, 0, 0.45);                       // outer petals
    } else if (pick < 0.66) {
      p = petal(0.9, 5, Math.PI / 5, 0.7);               // inner petals, offset
      p[2] += 0.05;
    } else if (pick < 0.76) {
      const [x, y, z] = randomUnitVector();              // seed head (dome)
      p = [x * 0.22, headY + y * 0.22, Math.abs(z) * 0.18 + 0.12];
    } else if (pick < 0.92) {
      const u = Math.random();                           // stem
      p = [0.18 * Math.sin(u * 2.4) + gauss() * 0.025, headY - 0.15 - u * 2.55, -0.1 + gauss() * 0.025];
    } else {
      const l = Math.random();                           // leaf on the stem
      const w = Math.sin(Math.PI * l) * 0.24 * (Math.random() * 2 - 1);
      const a = 0.6;
      const bx = 0.18 * Math.sin(0.55 * 2.4), by = headY - 0.15 - 0.55 * 2.55;
      p = [bx + l * 0.9 * Math.cos(a) - w * Math.sin(a), by + l * 0.9 * Math.sin(a) + w * Math.cos(a), -0.1 + gauss() * 0.02];
    }
    out[i * 3] = p[0] * 0.92;
    out[i * 3 + 1] = (p[1] + 0.35) * 0.92;
    out[i * 3 + 2] = p[2] * 0.92;
  }
  return out;
}

/** Saturn: a planet sphere plus a tilted ring with a Cassini-like gap. */
function makeSaturn(n) {
  const out = new Float32Array(n * 3);
  const tiltX = 0.42, tiltZ = -0.28;
  const cx = Math.cos(tiltX), sx = Math.sin(tiltX), cz = Math.cos(tiltZ), sz = Math.sin(tiltZ);
  for (let i = 0; i < n; i++) {
    let x, y, z;
    if (Math.random() < 0.55) {
      const [ux, uy, uz] = randomUnitVector();
      const r = 0.85 * (Math.random() < 0.85 ? 1 - Math.random() * 0.03 : Math.cbrt(Math.random()));
      x = ux * r; y = uy * r; z = uz * r;
    } else {
      let r = 1.2 + Math.sqrt(Math.random()) * 0.85;
      if (r > 1.6 && r < 1.68) r += 0.08;                  // ring gap
      const th = Math.random() * Math.PI * 2;
      x = r * Math.cos(th); y = gauss() * 0.012; z = r * Math.sin(th);
    }
    // Tilt the whole planet (rotate around X, then Z) so the ring reads in 3D.
    const y1 = y * cx - z * sx, z1 = y * sx + z * cx;
    out[i * 3] = x * cz - y1 * sz;
    out[i * 3 + 1] = x * sz + y1 * cz;
    out[i * 3 + 2] = z1;
  }
  return out;
}

// Fireworks are animated, so their targets are computed every frame.
const FIREWORKS = { shells: 7, period: 3.6, launch: 0.18 };

const TEMPLATES = {
  heart: { positions: makeHeart(PARTICLE_COUNT), followRate: 0.055 },
  flower: { positions: makeFlower(PARTICLE_COUNT), followRate: 0.055 },
  saturn: { positions: makeSaturn(PARTICLE_COUNT), followRate: 0.055 },
  fireworks: { positions: null, followRate: 0.4 },
};

// ===========================================================================
// Particle system: BufferGeometry + custom glow shader
// ===========================================================================
const N = PARTICLE_COUNT;
const positions = new Float32Array(N * 3);   // current (rendered) positions
const colors = new Float32Array(N * 3);      // current colors
const targetColors = new Float32Array(N * 3);
const alphas = new Float32Array(N).fill(1);
const sizes = new Float32Array(N);
const seed = new Float32Array(N);            // per-particle random 0..1
const seed2 = new Float32Array(N);
const dirs = new Float32Array(N * 3);        // per-particle random unit vector

for (let i = 0; i < N; i++) {
  const [x, y, z] = randomUnitVector();
  dirs.set([x, y, z], i * 3);
  const r = 4 + Math.random() * 3;           // start as a loose cloud
  positions.set([x * r, y * r, z * r], i * 3);
  seed[i] = Math.random();
  seed2[i] = Math.random();
  sizes[i] = 0.5 + Math.random() ** 2 * 1.3; // mostly small, a few bright ones
}

const geometry = new THREE.BufferGeometry();
const posAttr = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
const colorAttr = new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage);
const alphaAttr = new THREE.BufferAttribute(alphas, 1).setUsage(THREE.DynamicDrawUsage);
geometry.setAttribute('position', posAttr);
geometry.setAttribute('aColor', colorAttr);
geometry.setAttribute('aAlpha', alphaAttr);
geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 20); // skip per-frame recompute

const material = new THREE.ShaderMaterial({
  uniforms: {
    uSize: { value: 46 },
    uPixelRatio: { value: renderer.getPixelRatio() },
    uOpacity: { value: 0.55 },
  },
  vertexShader: /* glsl */ `
    uniform float uSize;
    uniform float uPixelRatio;
    attribute vec3 aColor;
    attribute float aAlpha;
    attribute float aSize;
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = uSize * aSize * uPixelRatio / -mv.z;  // perspective size
      vColor = aColor;
      vAlpha = aAlpha;
    }
  `,
  fragmentShader: /* glsl */ `
    uniform float uOpacity;
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
      // Soft round sprite: bright core + gaussian-like halo.
      float d = length(gl_PointCoord - 0.5) * 2.0;
      if (d > 1.0) discard;
      float halo = pow(1.0 - d, 2.2);
      float core = smoothstep(0.35, 0.0, d);
      float a = (halo * 0.7 + core * 0.5) * vAlpha * uOpacity;
      gl_FragColor = vec4(vColor * a, a);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
});

const particles = new THREE.Points(geometry, material);
const group = new THREE.Group();
group.add(particles);
scene.add(group);

// ===========================================================================
// Colors
// ===========================================================================
const state = {
  template: 'heart',
  templateSince: 0,
  baseColor: new THREE.Color(document.getElementById('color').value),
  colorAnimUntil: 0,
  breathing: false,
  breathStart: 0,
};

/** Compute per-particle target colors from the base color (small hue/lightness jitter). */
function updateTargetColors(now) {
  const hsl = { h: 0, s: 0, l: 0 };
  state.baseColor.getHSL(hsl);
  const c = new THREE.Color();
  const fireworks = state.template === 'fireworks';
  for (let i = 0; i < N; i++) {
    let h = hsl.h + (seed[i] - 0.5) * 0.06;
    if (fireworks) h += (i % FIREWORKS.shells) * 0.14;   // each shell its own hue
    const l = THREE.MathUtils.clamp(hsl.l + (seed2[i] - 0.5) * 0.3, 0.25, 0.85);
    c.setHSL(((h % 1) + 1) % 1, hsl.s, l);
    targetColors[i * 3] = c.r;
    targetColors[i * 3 + 1] = c.g;
    targetColors[i * 3 + 2] = c.b;
  }
  state.colorAnimUntil = now + 2.5;
}

// ===========================================================================
// Hand input over WebSocket (auto-reconnect; app stays usable when offline)
// ===========================================================================
const input = {
  connected: false,
  lastMessage: 0,
  numHands: 0,
  openness: 0,
  handDistance: null,
};

const ui = {
  cam: document.getElementById('cam'),
  camPlaceholder: document.getElementById('cam-placeholder'),
  wsDot: document.getElementById('ws-dot'),
  wsState: document.getElementById('ws-state'),
  backend: document.getElementById('backend'),
  fpsRender: document.getElementById('fps-render'),
  fpsInfer: document.getElementById('fps-infer'),
  hands: document.getElementById('hands'),
  meter: document.getElementById('open-meter'),
  breath: document.getElementById('breath'),
  breathWord: document.getElementById('breath-word'),
  breathCount: document.getElementById('breath-count'),
};

function setConnectionState(kind, text) {
  ui.wsDot.className = `dot ${kind}`;
  ui.wsState.textContent = text;
}

function connect(delay = 500) {
  setConnectionState('warn', 'Connecting');
  let ws;
  try {
    ws = new WebSocket(WS_URL);
  } catch (err) {
    console.warn('WebSocket error', err);
    setTimeout(() => connect(Math.min(delay * 2, 5000)), delay);
    return;
  }
  ws.onopen = () => {
    input.connected = true;
    delay = 500;
    setConnectionState('ok', 'Connected');
  };
  ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
  ws.onclose = () => {
    input.connected = false;
    input.numHands = 0;                    // fall back to idle mode
    setConnectionState('', 'Offline · retrying');
    ui.fpsInfer.textContent = '—';
    ui.hands.textContent = '0';
    ui.meter.style.width = '0%';
    ui.cam.style.display = 'none';
    ui.camPlaceholder.style.display = '';
    ui.camPlaceholder.textContent = 'Inference server offline';
    setTimeout(() => connect(Math.min(delay * 2, 5000)), delay);
  };
  ws.onerror = () => ws.close();
}

function handleMessage(msg) {
  if (msg.backend) ui.backend.textContent = BACKEND_LABELS[msg.backend] || msg.backend_label || msg.backend;
  if (msg.type !== 'hands') return;

  input.lastMessage = performance.now() / 1000;
  input.numHands = msg.num_hands;
  input.openness = msg.openness;
  input.handDistance = msg.hand_distance;

  if (msg.image) {
    ui.cam.src = msg.image;
    ui.cam.style.display = 'block';
    ui.camPlaceholder.style.display = 'none';
  } else {
    ui.cam.style.display = 'none';
    ui.camPlaceholder.style.display = '';
    ui.camPlaceholder.textContent = msg.camera ? 'Preview disabled' : 'No camera detected';
  }
  ui.fpsInfer.textContent = `${msg.fps.toFixed(0)} fps · ${msg.infer_ms.toFixed(0)} ms`;
  ui.hands.textContent = msg.num_hands;
  ui.meter.style.width = `${(msg.num_hands ? msg.openness : 0) * 100}%`;
}

// ===========================================================================
// Breathing guide (4 s inhale, 4 s hold, 6 s exhale)
// ===========================================================================
/** Returns { value 0..1 (lung fullness), phase name, seconds left in phase }. */
function breathAt(elapsed) {
  let t = elapsed % BREATH_CYCLE;
  for (const phase of BREATH_PHASES) {
    if (t < phase.duration) {
      const p = t / phase.duration;
      const ease = 0.5 - 0.5 * Math.cos(Math.PI * p);
      const value = phase.name === 'Inhale' ? ease : phase.name === 'Hold' ? 1 : 1 - ease;
      return { value, phase: phase.name, left: Math.ceil(phase.duration - t) };
    }
    t -= phase.duration;
  }
  return { value: 0, phase: 'Inhale', left: 4 };
}

let lastBreathLabel = '';
function updateBreathUI(b) {
  const label = `${b.phase}|${b.left}`;
  if (label === lastBreathLabel) return;
  lastBreathLabel = label;
  ui.breathWord.textContent = b.phase;
  ui.breathCount.textContent = b.left;
}

// ===========================================================================
// Per-frame simulation
// ===========================================================================
// Render-side smoothing (backend sends ~30 Hz, we render at 60+ Hz).
const smooth = { presence: 0, open: 0.45, distFactor: 1, breath: 0 };
const shellCenters = new Float32Array(FIREWORKS.shells * 3);
const shellTau = new Float32Array(FIREWORKS.shells);

/** Compute burst center and normalized time (0..1) of each firework shell. */
function updateFireworkShells(t) {
  const { shells, period } = FIREWORKS;
  for (let s = 0; s < shells; s++) {
    const local = t + (s / shells) * period;
    const cycle = Math.floor(local / period);
    shellTau[s] = (local - cycle * period) / period;
    shellCenters[s * 3] = (hash(s * 17.1 + cycle * 3.7) - 0.5) * 3.4;
    shellCenters[s * 3 + 1] = 0.35 + (hash(s * 9.3 + cycle * 7.9) - 0.5) * 1.6;
    shellCenters[s * 3 + 2] = (hash(s * 4.7 + cycle * 1.3) - 0.5) * 1.6;
  }
}

/** Firework target for particle i; writes into out[0..2] and returns alpha. */
function fireworkTarget(i, out) {
  const s = i % FIREWORKS.shells;
  const tau = shellTau[s];
  const cx = shellCenters[s * 3], cy = shellCenters[s * 3 + 1], cz = shellCenters[s * 3 + 2];
  if (tau < FIREWORKS.launch) {
    // Rising rocket with a short fading trail.
    const p = tau / FIREWORKS.launch;
    const head = -3.4 + (cy + 3.4) * (1 - (1 - p) ** 2);
    out[0] = cx + gaussFrom(seed2[i]) * 0.02;
    out[1] = head - seed[i] * 0.7 * p;
    out[2] = cz;
    return 0.5 * (1 - seed[i]) * Math.min(1, p * 4);
  }
  // Burst: particles fly out radially, slow down, droop with gravity and fade.
  // Most particles sit on the expanding shell; the rest form streaks behind it.
  const p = (tau - FIREWORKS.launch) / (1 - FIREWORKS.launch);
  const streak = seed[i] ** 3 * 0.45;           // 0 = shell front, >0 = trailing
  const r = 1.3 * (0.92 + 0.08 * seed2[i]) * (1 - Math.exp(-5 * p)) * (1 - streak);
  out[0] = cx + dirs[i * 3] * r;
  out[1] = cy + dirs[i * 3 + 1] * r - 0.9 * p * p;
  out[2] = cz + dirs[i * 3 + 2] * r;
  return 1.8 * (1 - p) ** 1.6 * (1 - streak * 1.6);  // >1 = extra bright right after the burst
}

const tmp = new Float32Array(3);

function updateParticles(t, dt) {
  // ---- 1. Control signals ------------------------------------------------
  const live = input.connected && t - input.lastMessage < 1.5;
  const hands = live ? input.numHands : 0;
  const k = (rate) => 1 - Math.pow(1 - rate, dt * 60);  // frame-rate independent lerp

  smooth.presence += ((hands > 0 ? 1 : 0) - smooth.presence) * k(0.05);
  smooth.open += ((hands > 0 ? input.openness : 0.45) - smooth.open) * k(0.15);
  const distTarget = hands === 2 && input.handDistance != null
    ? THREE.MathUtils.mapLinear(THREE.MathUtils.clamp(input.handDistance, 0.2, 0.9), 0.2, 0.9, 0.85, 1.25)
    : 1;
  smooth.distFactor += (distTarget - smooth.distFactor) * k(0.08);

  // Gesture: fist -> gather & shrink (0.55x); open hands -> expand & diffuse (1.35x).
  // Spread stays moderate so the shape remains readable while it "breathes out".
  const gestureScale = THREE.MathUtils.lerp(0.55, 1.35, smooth.open) * smooth.distFactor;
  let scale = THREE.MathUtils.lerp(1, gestureScale, smooth.presence);
  let spread = smooth.presence * smooth.open ** 2 * 0.22;

  // Breathing mode drives the scale; hands still modulate it gently.
  const breathTarget = state.breathing ? 1 : 0;
  smooth.breath += (breathTarget - smooth.breath) * k(0.03);
  if (smooth.breath > 0.001) {
    const b = breathAt(t - state.breathStart);
    if (state.breathing) updateBreathUI(b);
    const handMod = THREE.MathUtils.lerp(1, THREE.MathUtils.lerp(0.85, 1.15, smooth.open), smooth.presence);
    const breathScale = THREE.MathUtils.lerp(0.72, 1.28, b.value) * handMod;
    scale = THREE.MathUtils.lerp(scale, breathScale, smooth.breath);
    spread += smooth.breath * b.value * 0.15;
  }

  // Idle mode (no hands): slow rotation and floating.
  const idle = 1 - smooth.presence;
  group.rotation.y += dt * (0.06 + 0.18 * idle);
  group.rotation.x = Math.sin(t * 0.23) * 0.12 * idle;
  group.position.y = Math.sin(t * 0.4) * 0.12 * idle;
  const floatAmp = 0.025 + 0.06 * idle;

  // ---- 2. Particle integration --------------------------------------------
  const tpl = TEMPLATES[state.template];
  const fireworks = tpl.positions === null;
  if (fireworks) updateFireworkShells(t);

  // Ease into a new template's follow rate (fireworks need to be snappy).
  const sinceSwitch = t - state.templateSince;
  const rate = THREE.MathUtils.lerp(0.05, tpl.followRate, THREE.MathUtils.clamp(sinceSwitch / 2, 0, 1));
  const follow = k(rate);
  const alphaFollow = k(fireworks ? rate : 0.05);

  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    let alpha = 1;
    if (fireworks) {
      alpha = fireworkTarget(i, tmp);
    } else {
      tmp[0] = tpl.positions[i3];
      tmp[1] = tpl.positions[i3 + 1];
      tmp[2] = tpl.positions[i3 + 2];
    }
    // target = template * scale + random outward spread + gentle floating
    const sp = spread * (0.3 + seed[i]);
    const ph = seed2[i] * 6.2832;
    const tx = tmp[0] * scale + dirs[i3] * sp + Math.sin(t * 0.7 + ph) * floatAmp;
    const ty = tmp[1] * scale + dirs[i3 + 1] * sp + Math.cos(t * 0.6 + ph * 1.3) * floatAmp;
    const tz = tmp[2] * scale + dirs[i3 + 2] * sp + Math.sin(t * 0.5 + ph * 0.7) * floatAmp;

    positions[i3] += (tx - positions[i3]) * follow;
    positions[i3 + 1] += (ty - positions[i3 + 1]) * follow;
    positions[i3 + 2] += (tz - positions[i3 + 2]) * follow;
    alphas[i] += (alpha - alphas[i]) * alphaFollow;
  }
  posAttr.needsUpdate = true;
  alphaAttr.needsUpdate = true;

  // ---- 3. Color transitions (only while animating) ------------------------
  if (t < state.colorAnimUntil) {
    const cf = k(0.06);
    for (let i = 0; i < N * 3; i++) colors[i] += (targetColors[i] - colors[i]) * cf;
    colorAttr.needsUpdate = true;
  }
}

// ===========================================================================
// UI wiring
// ===========================================================================
const templateButtons = [...document.querySelectorAll('[data-template]')];

function setTemplate(name) {
  if (!TEMPLATES[name] || name === state.template) return;
  const wasFireworks = state.template === 'fireworks';
  state.template = name;
  state.templateSince = clock.getElapsedTime();
  templateButtons.forEach((b) => b.classList.toggle('active', b.dataset.template === name));
  if (wasFireworks || name === 'fireworks') updateTargetColors(state.templateSince);
}

function setBreathing(on) {
  state.breathing = on;
  if (on) {
    state.breathStart = clock.getElapsedTime();
    lastBreathLabel = '';
  }
  ui.breath.classList.toggle('on', on);
  const btn = document.getElementById('breath-toggle');
  btn.classList.toggle('active', on);
  btn.setAttribute('aria-pressed', String(on));
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

templateButtons.forEach((b) => b.addEventListener('click', () => setTemplate(b.dataset.template)));
document.getElementById('color').addEventListener('input', (e) => {
  state.baseColor.set(e.target.value);
  updateTargetColors(clock.getElapsedTime());
});
document.getElementById('breath-toggle').addEventListener('click', () => setBreathing(!state.breathing));
document.getElementById('fullscreen').addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', () => {
  document.getElementById('fullscreen').textContent = document.fullscreenElement ? 'Exit' : 'Fullscreen';
});
window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  const names = ['heart', 'flower', 'saturn', 'fireworks'];
  if (e.key >= '1' && e.key <= '4') setTemplate(names[Number(e.key) - 1]);
  else if (e.key === 'b' || e.key === 'B') setBreathing(!state.breathing);
  else if (e.key === 'f' || e.key === 'F') toggleFullscreen();
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  material.uniforms.uPixelRatio.value = renderer.getPixelRatio();
});

// ===========================================================================
// Main loop
// ===========================================================================
const clock = new THREE.Clock();
let fpsFrames = 0;
let fpsTime = 0;

function animate() {
  const dt = Math.min(clock.getDelta(), 0.1);  // clamp after tab switches
  const t = clock.getElapsedTime();

  updateParticles(t, dt);
  stars.rotation.y += dt * 0.004;
  renderer.render(scene, camera);

  fpsFrames++;
  fpsTime += dt;
  if (fpsTime >= 0.5) {
    ui.fpsRender.textContent = `${Math.round(fpsFrames / fpsTime)} fps`;
    fpsFrames = 0;
    fpsTime = 0;
  }
  requestAnimationFrame(animate);
}

updateTargetColors(0);
colors.set(targetColors);
connect();
requestAnimationFrame(animate);

// Expose a tiny debug handle (handy for demos and automated checks).
window.particles = { state, input, smooth, setTemplate, setBreathing, N };

// ===========================================================================
// Helpers
// ===========================================================================
function randomUnitVector() {
  const u = Math.random() * 2 - 1;
  const th = Math.random() * Math.PI * 2;
  const r = Math.sqrt(1 - u * u);
  return [r * Math.cos(th), r * Math.sin(th), u];
}

/** Approximate standard normal sample (sum of uniforms). */
function gauss() {
  return (Math.random() + Math.random() + Math.random() - 1.5) * 1.15;
}

/** Deterministic pseudo-normal from a uniform seed (no allocation per frame). */
function gaussFrom(u) {
  return (u - 0.5) * 2.3;
}

/** Cheap deterministic hash -> [0, 1). */
function hash(x) {
  const s = Math.sin(x * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
}

function clampInt(value, min, max, fallback) {
  const n = parseInt(value, 10);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
}
