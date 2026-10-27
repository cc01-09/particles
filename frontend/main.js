/**
 * Particles — gesture-controlled cosmic particle system (frontend).
 *
 * Data flow:
 *   WebSocket (per-hand openness + palm center @ ~30 Hz)
 *     -> control state (smoothed at render rate)
 *     -> per-particle spring-damper physics on the CPU:
 *          target = lerp(formed, scattered, openness)      (gather <-> scatter)
 *     -> GPU shader: curl-noise drift, twinkle, Doppler beaming, glow sprite
 *     -> bloom post-processing
 *
 * The camera and the particle sizes never change with gestures: a gesture
 * moves each particle individually.
 *
 * Hand input is handled by HandTracker as messages arrive — independent of
 * rendering and of which UI panels are visible. All timing uses one clock:
 * performance.now().
 *
 * No build step: Three.js is loaded from a CDN through the import map.
 * URL options:  ?ws=ws://host:8765   ?n=32000 (fixed count, disables auto-scaling)
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { TEMPLATE_DEFS, buildTemplates } from './js/templates.js';
import { NATURAL, THEMES, cssGradient, paletteFromBase, samplePalette, toColors } from './js/palettes.js';
import { createSky } from './js/sky.js';
import { HandSocket } from './js/hands.js';
import { DebugPanel, loadThresholds } from './js/debug.js';
import { HandTracker } from './js/handTracker.js';
import { BRIGHT_STAR_FRACTION, WAKE_SECONDS, loadTuning } from './js/tuning.js';
import { mulberry32, unitVector } from './js/noise.js';

// ===========================================================================
// Configuration
// ===========================================================================
const params = new URLSearchParams(location.search);
const MAX_PARTICLES = clampInt(params.get('n'), 4000, 80000, 32000);
const MIN_PARTICLES = Math.min(8000, MAX_PARTICLES);
const ADAPTIVE = !params.has('n') && params.get('adaptive') !== '0';
const WS_URL = params.get('ws') || `ws://${location.hostname || 'localhost'}:8765`;

const CAMERA_Z = 6.5;
const FOV = 55;
const TAN_HALF_FOV = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
// Live-tunable constants (motion, hand stability, idle dimming): see js/tuning.js
const tuning = loadTuning();
const now = () => performance.now() / 1000;   // the single time base for everything

const BREATH_PHASES = [          // 4-4-6 breathing pattern (seconds)
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
// Renderer, camera, post-processing (bloom)
// ===========================================================================
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;   // soft highlight roll-off: glow without clipping
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV, window.innerWidth / window.innerHeight, 0.1, 400);
camera.position.set(0, 0, CAMERA_Z);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.42, 0.45, 0.32);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// Background sky (multi-layer stars + Milky Way band, with parallax).
const sky = createSky(renderer.getPixelRatio());
scene.add(sky.group);

// Opaque black sphere: the event horizon (Black Hole) or the planet body
// (Saturn) hides the particles and stars behind it.
const occluder = new THREE.Mesh(
  new THREE.SphereGeometry(1, 48, 32),
  new THREE.MeshBasicMaterial({ color: 0x000000 }),
);
occluder.scale.setScalar(0.001);
scene.add(occluder);

// ===========================================================================
// Templates and per-particle state
// ===========================================================================
const TEMPLATES = buildTemplates(MAX_PARTICLES);
const N = MAX_PARTICLES;
let activeCount = N;

const positions = new Float32Array(N * 3);    // simulated + rendered positions
const velocities = new Float32Array(N * 3);
const prevFormed = new Float32Array(N * 3);   // last formed target (feed-forward)
const colors = new Float32Array(N * 3);
const targetColors = new Float32Array(N * 3);
const sizes = new Float32Array(N);
const targetSizes = new Float32Array(N);
const doppler = new Float32Array(N);
const targetDoppler = new Float32Array(N);
const seeds = new Float32Array(N);            // 0..1, twinkle phase + bright-star pick
const mags = new Float32Array(N);             // 0..1 power law: most faint, few bright
const groups = new Float32Array(N);           // 0 / 1: which hand a stardust grain follows
const stiffness = new Float32Array(N);        // base spring frequency (rad/s), x tuning.springStiffness
const scatterDir = new Float32Array(N * 3);   // per-particle random scatter direction
const scatterDist = new Float32Array(N);

{
  const random = mulberry32(7);
  for (let i = 0; i < N; i++) {
    seeds[i] = random();
    mags[i] = random() ** 5;
    groups[i] = i % 2;
    stiffness[i] = 1.6 + random() * 2.6;      // varied -> particles arrive at different times
    const [x, y, z] = unitVector(random);
    scatterDir.set([x, y, z], i * 3);
    scatterDist[i] = 1.2 + 3.2 * random() ** 0.7;
  }
}

const geometry = new THREE.BufferGeometry();
const posAttr = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
const colorAttr = new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage);
const sizeAttr = new THREE.BufferAttribute(sizes, 1).setUsage(THREE.DynamicDrawUsage);
const dopplerAttr = new THREE.BufferAttribute(doppler, 1).setUsage(THREE.DynamicDrawUsage);
geometry.setAttribute('position', posAttr);
geometry.setAttribute('aColor', colorAttr);
geometry.setAttribute('aSize', sizeAttr);
geometry.setAttribute('aDoppler', dopplerAttr);
geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
geometry.setAttribute('aGroup', new THREE.BufferAttribute(groups, 1));
geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 50);

// GLSL 3D simplex noise (Ian McEwan / Ashima Arts, MIT license).
const GLSL_NOISE = /* glsl */ `
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
      i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 105.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
  }
  // Divergence-free flow: cross product of the gradients of two noise fields.
  vec3 curlNoise(vec3 p) {
    const float e = 0.12;
    float a = snoise(p), b = snoise(p + 31.7);
    vec3 ga = vec3(snoise(p + vec3(e, 0, 0)) - a, snoise(p + vec3(0, e, 0)) - a, snoise(p + vec3(0, 0, e)) - a);
    vec3 gb = vec3(snoise(p + vec3(31.7 + e, 31.7, 31.7)) - b, snoise(p + vec3(31.7, 31.7 + e, 31.7)) - b,
                   snoise(p + vec3(31.7, 31.7, 31.7 + e)) - b);
    return cross(ga, gb) / (e * e);
  }
`;

const material = new THREE.ShaderMaterial({
  uniforms: {
    uTime: { value: 0 },
    uSize: { value: 26 },
    uPixelRatio: { value: renderer.getPixelRatio() },
    uFlow: { value: 0.18 },          // ambient drift (gas flow, floating dust)
    uScatter: { value: new THREE.Vector2(1, 1) },  // scatter amount per hand group
    uScatterNoise: { value: 0.5 },   // extra curl-noise drift while scattered
    uDoppler: { value: 0 },
    uOpacity: { value: 0.85 },
    uDim: { value: 1 },              // idle dimming (1 = normal brightness)
    uIdle: { value: 0 },             // 0..1 dimming progress (boosts the few bright stars)
    uStarFraction: { value: BRIGHT_STAR_FRACTION },
  },
  vertexShader: /* glsl */ `
    uniform float uTime, uSize, uPixelRatio, uFlow, uScatterNoise, uDoppler, uDim, uIdle, uStarFraction;
    uniform vec2 uScatter;
    attribute vec3 aColor;
    attribute float aSize, aSeed, aGroup, aDoppler;
    varying vec3 vColor;
    ${GLSL_NOISE}
    void main() {
      vec3 p = position;
      // Curl-noise drift: gentle when formed, stronger while scattered ("blown stardust").
      float scatter = mix(uScatter.x, uScatter.y, aGroup);
      float amp = uFlow + scatter * uScatterNoise;
      if (amp > 0.002) {
        p += curlNoise(p * 0.32 + vec3(0.0, 0.0, uTime * 0.04)) * amp * 0.12;
      }
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = uSize * aSize * uPixelRatio / -mv.z;

      // A few percent of particles act as "bright stars": they stay lit and
      // twinkle more while everything else dims during idle time.
      float star = step(fract(aSeed * 97.31), uStarFraction);
      float dim = mix(uDim, 1.0 + 1.6 * uIdle, star);

      // Subtle twinkle, different rate and phase per particle.
      float twinkle = 1.0 + (0.14 + 0.3 * star * uIdle) * sin(uTime * (0.7 + aSeed * 2.6) + aSeed * 61.0);
      // Relativistic beaming: material moving toward us (right side) is brighter.
      float beam = 1.0 + uDoppler * aDoppler * 0.8 * clamp(p.x / max(length(p.xz), 0.3), -1.0, 1.0);
      vColor = aColor * twinkle * max(beam, 0.2) * dim;
    }
  `,
  fragmentShader: /* glsl */ `
    uniform float uOpacity;
    varying vec3 vColor;
    void main() {
      float d = length(gl_PointCoord - 0.5) * 2.0;
      if (d > 1.0) discard;
      float a = exp(-d * d * 5.5) + 0.35 * exp(-d * d * 22.0);   // soft halo + core
      gl_FragColor = vec4(vColor, a * uOpacity);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
});

const particles = new THREE.Points(geometry, material);
particles.frustumCulled = false;
scene.add(particles);

// ===========================================================================
// App state
// ===========================================================================
const state = {
  template: 'stardust',
  theme: 'natural',
  customColor: document.getElementById('color').value,
  breathing: false,
  breathStart: 0,
  transitionUntil: 0,
  lastUserActivity: 0,   // last key / pointer input (counts as activity for idle dimming)
  resetFeedForward: true,
};

const input = {
  connected: false,
  lastMessage: 0,
  hands: [],            // raw [{label, openness, ratio, center:[x,y]}] (x mirrored: selfie view)
  thresholds: null,     // as reported by the server
};

// Turns detections into stable per-group targets (stabilize / control / hold).
const tracker = new HandTracker(tuning);

/** Default scatter of the current mode: Stardust fills the screen, objects are formed. */
const defaultOpen = () => (state.template === 'stardust' ? 1 : 0);

// Smoothed control signals (what the particles actually follow).
const ctrl = {
  breathWeight: 0,      // fades the breathing influence in/out
  scatter: 0,           // template modes: mean of the two groups (0 = formed, 1 = scattered)
  groups: [             // one particle group per hand
    { x: 0, y: 0, open: 1, smooth: 1 },
    { x: 0, y: 0, open: 1, smooth: 1 },
  ],
  brightness: 1,        // idle dimming level currently applied
  sway: { x: 0, y: 0 },
};

// ===========================================================================
// Colors
// ===========================================================================
function currentPalette() {
  if (state.theme === 'custom') return toColors(paletteFromBase(state.customColor));
  if (state.theme === 'natural') return toColors(NATURAL[state.template]);
  return toColors(THEMES[state.theme].colors);
}

/** Recompute per-particle color / size / Doppler targets and start a smooth transition. */
function updateTargets(now) {
  const d = TEMPLATES[state.template].data;
  const palette = currentPalette();
  const rgb = [0, 0, 0];
  for (let i = 0; i < N; i++) {
    samplePalette(palette, d.c[i], rgb);
    // Brightness follows a power law: most particles faint, a few brilliant.
    const b = d.b[i] * (0.35 + 1.6 * mags[i]) * 0.75;
    targetColors[i * 3] = rgb[0] * b;
    targetColors[i * 3 + 1] = rgb[1] * b;
    targetColors[i * 3 + 2] = rgb[2] * b;
    targetSizes[i] = d.size[i] * (0.5 + 2.2 * mags[i]);
    targetDoppler[i] = d.doppler[i];
  }
  state.transitionUntil = now + 3;
}

// ===========================================================================
// Hand input
// ===========================================================================
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
  handDot: document.getElementById('hand-dot'),
};

const debugPanel = new DebugPanel(document.getElementById('debug'), tuning, (thresholds) => {
  socket.send({ type: 'config', ...thresholds });
});

const socket = new HandSocket(WS_URL, {
  onState(kind, text) {
    ui.wsDot.className = `dot ${kind}`;
    ui.wsState.textContent = text;
  },
  onOpen() {
    // Re-apply calibration saved in this browser.
    const saved = loadThresholds();
    if (saved) socket.send({ type: 'config', ...saved });
  },
  onClose(wasConnected) {
    input.connected = false;
    input.hands = [];
    input.thresholds = null;
    if (wasConnected) tracker.dropAll(now());   // controlling hands become held
    ui.cam.style.display = 'none';
    ui.camPlaceholder.style.display = '';
    ui.camPlaceholder.textContent = 'Inference server offline';
    ui.fpsInfer.textContent = '—';
    ui.hands.textContent = '0';
    ui.meter.style.width = '0%';
  },
  onMessage(msg) {
    if (msg.backend) ui.backend.textContent = BACKEND_LABELS[msg.backend] || msg.backend_label || msg.backend;
    if (msg.type !== 'hands') return;
    // Gesture state is updated here, on arrival, before any UI work.
    input.connected = true;
    input.lastMessage = now();
    input.hands = msg.hands || [];
    input.thresholds = msg.thresholds || null;
    tracker.ingest(input.hands, input.lastMessage, msg.stale === true);

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
  },
});

/** Palm center (normalized, mirrored) -> world position on the z = 0 plane. */
function handToWorld(center) {
  const halfH = CAMERA_Z * TAN_HALF_FOV;
  const halfW = halfH * camera.aspect;
  // A little gain so reaching the screen edges doesn't need the frame edges.
  const u = THREE.MathUtils.clamp((center[0] - 0.5) * 2 * 1.25, -1, 1);
  const v = THREE.MathUtils.clamp(-(center[1] - 0.5) * 2 * 1.25, -1, 1);
  return [u * halfW * 0.85, v * halfH * 0.85];
}

// ===========================================================================
// Breathing guide (4 s inhale, 4 s hold, 6 s exhale)
// ===========================================================================
/** Returns { value 0..1 (lung fullness), phase name, seconds left in phase }. */
function breathAt(elapsed) {
  let t = elapsed % BREATH_CYCLE;
  for (const phase of BREATH_PHASES) {
    if (t < phase.duration) {
      const ease = 0.5 - 0.5 * Math.cos((Math.PI * t) / phase.duration);
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
// Control signals (per frame)
// ===========================================================================
const lerpRate = (rate, dt) => 1 - Math.pow(1 - rate, dt * 60);  // frame-rate independent

function updateControls(t, dt) {
  tracker.update(t);

  // Breathing: inhale -> scatter, exhale -> gather.
  ctrl.breathWeight += ((state.breathing ? 1 : 0) - ctrl.breathWeight) * lerpRate(0.03, dt);
  let breath = 0;
  if (state.breathing || ctrl.breathWeight > 0.001) {
    const b = breathAt(t - state.breathStart);
    breath = b.value;
    if (state.breathing) updateBreathUI(b);
  }
  const bw = ctrl.breathWeight;

  // --- Groups follow their targets (a controlling hand, or the held state) ---
  // Openness maps linearly to scatter. The follower is rate-limited, so the
  // particles move as fast as the fingers do, up to maxScatterSpeed.
  const smoothF = 1 - Math.exp(-dt / tuning.scatterSmoothing);
  const maxStep = tuning.maxScatterSpeed * dt;
  const followF = 1 - Math.exp(-dt / tuning.followDelay);
  ctrl.groups.forEach((g, idx) => {
    const target = tracker.groups[idx].target;
    const open = THREE.MathUtils.lerp(target.open, 0.08 + 0.82 * breath, bw);
    g.smooth += (open - g.smooth) * smoothF;
    g.open += THREE.MathUtils.clamp(g.smooth - g.open, -maxStep, maxStep);
    const [x, y] = handToWorld([target.x, target.y]);
    g.x += (x - g.x) * followF;
    g.y += (y - g.y) * followF;
  });
  ctrl.scatter = (ctrl.groups[0].open + ctrl.groups[1].open) / 2;
  if (state.template !== 'stardust') {
    // Objects: inhale scatters gently (35%) instead of fully.
    ctrl.scatter = THREE.MathUtils.lerp(ctrl.scatter, 0.35 * breath, bw);
  }


  // Background parallax: slow drift plus a hint of the groups' position.
  const hx = (tracker.groups[0].target.x + tracker.groups[1].target.x) / 2 - 0.5;
  const hy = (tracker.groups[0].target.y + tracker.groups[1].target.y) / 2 - 0.5;
  const sx = Math.sin(t * 0.05) * 0.25 + hx * 0.6;
  const sy = Math.cos(t * 0.037) * 0.15 - hy * 0.4;
  ctrl.sway.x += (sx - ctrl.sway.x) * lerpRate(0.02, dt);
  ctrl.sway.y += (sy - ctrl.sway.y) * lerpRate(0.02, dt);
}

/**
 * Idle dimming: after `idleSeconds` without a hand or without a noticeable
 * gesture change, fade to `dimLevel` over `dimDuration`. Any activity
 * (a hand appearing, a gesture change, a key or pointer) wakes it up
 * within WAKE_SECONDS. Background stars are never dimmed.
 */
function updateDimming(t, dt) {
  const idleFor = t - Math.max(tracker.lastActivity, state.lastUserActivity);
  const p = THREE.MathUtils.clamp((idleFor - tuning.idleSeconds) / tuning.dimDuration, 0, 1);
  const level = 1 - (1 - tuning.dimLevel) * (p * p * (3 - 2 * p));
  if (level > ctrl.brightness) {
    ctrl.brightness += (level - ctrl.brightness) * (1 - Math.exp(-dt / (WAKE_SECONDS / 3)));
  } else {
    ctrl.brightness = level;     // the fade itself is already slow
  }
  material.uniforms.uDim.value = ctrl.brightness;
  material.uniforms.uIdle.value = THREE.MathUtils.clamp((1 - ctrl.brightness) / (1 - tuning.dimLevel + 1e-6), 0, 1);
}

// ===========================================================================
// Particle physics (per frame, CPU)
// ===========================================================================
function simulate(t, dt) {
  const tpl = TEMPLATES[state.template];
  const d = tpl.data;
  const m = d.tilt;
  const stardust = state.template === 'stardust';
  const aspect = camera.aspect;
  const [g0, g1] = ctrl.groups;
  const resetFF = state.resetFeedForward;
  state.resetFeedForward = false;
  const invDt = 1 / dt;
  const stiffScale = tuning.springStiffness;
  const zeta = tuning.springDamping;

  for (let i = 0; i < activeCount; i++) {
    const i3 = i * 3;
    let fx, fy, fz;     // formed position
    let sx, sy, sz;     // scattered position
    let o;              // 0 = formed, 1 = scattered

    if (stardust) {
      // Formed = dense cloud around the hand; scattered = spread over the screen.
      const g = groups[i] < 0.5 ? g0 : g1;
      fx = g.x + d.pos[i3];
      fy = g.y + d.pos[i3 + 1];
      fz = d.pos[i3 + 2];
      const z = d.screen[i3 + 2];
      const halfH = (CAMERA_Z - z) * TAN_HALF_FOV * 1.04;
      const ph = seeds[i] * 40;
      sx = (d.screen[i3] + 0.05 * Math.sin(t * 0.07 + ph)) * halfH * aspect;
      sy = (d.screen[i3 + 1] + 0.05 * Math.cos(t * 0.06 + ph)) * halfH;
      sz = z;
      o = g.open;         // linear: half-open hand = half scattered
    } else {
      // Formed = template point, spun around its local axis, then tilted.
      let lx = d.pos[i3];
      const ly = d.pos[i3 + 1];
      let lz = d.pos[i3 + 2];
      const w = d.omega[i];
      if (w !== 0) {
        const c = Math.cos(w * t);
        const s = Math.sin(w * t);
        const x = lx * c - lz * s;
        lz = lx * s + lz * c;
        lx = x;
      }
      if (d.fixed[i]) {
        fx = lx; fy = ly; fz = lz;
      } else {
        fx = m[0] * lx + m[1] * ly + m[2] * lz;
        fy = m[3] * lx + m[4] * ly + m[5] * lz;
        fz = m[6] * lx + m[7] * ly + m[8] * lz;
      }
      // Scattered = blown away along a personal random direction.
      const dist = scatterDist[i];
      sx = fx + scatterDir[i3] * dist;
      sy = fy + scatterDir[i3 + 1] * dist;
      sz = fz + scatterDir[i3 + 2] * dist;
      o = ctrl.scatter;
    }

    const tx = fx + (sx - fx) * o;
    const ty = fy + (sy - fy) * o;
    const tz = fz + (sz - fz) * o;

    // Feed-forward: velocity of the formed shape itself (rotation), so that
    // spinning disks are tracked exactly. Stardust keeps the lag on purpose
    // (the dust "drags" behind the hand).
    let vfx = 0, vfy = 0, vfz = 0;
    if (!stardust && !resetFF) {
      vfx = (fx - prevFormed[i3]) * invDt;
      vfy = (fy - prevFormed[i3 + 1]) * invDt;
      vfz = (fz - prevFormed[i3 + 2]) * invDt;
    }
    prevFormed[i3] = fx;
    prevFormed[i3 + 1] = fy;
    prevFormed[i3 + 2] = fz;

    // Spring-damper: a = w^2 (target - p) - 2 zeta w (v - v_target)
    // (per-particle frequencies differ, so grains arrive at different times)
    const w = stiffness[i] * stiffScale;
    const k = w * w;
    const c = 2 * zeta * w;
    let vx = velocities[i3];
    let vy = velocities[i3 + 1];
    let vz = velocities[i3 + 2];
    vx += (k * (tx - positions[i3]) - c * (vx - vfx)) * dt;
    vy += (k * (ty - positions[i3 + 1]) - c * (vy - vfy)) * dt;
    vz += (k * (tz - positions[i3 + 2]) - c * (vz - vfz)) * dt;
    velocities[i3] = vx;
    velocities[i3 + 1] = vy;
    velocities[i3 + 2] = vz;
    positions[i3] += vx * dt;
    positions[i3 + 1] += vy * dt;
    positions[i3 + 2] += vz * dt;
  }
  posAttr.needsUpdate = true;

  // Smooth color / size transitions after a template or theme change.
  if (t < state.transitionUntil) {
    const f = lerpRate(0.045, dt);
    for (let i = 0; i < N * 3; i++) colors[i] += (targetColors[i] - colors[i]) * f;
    for (let i = 0; i < N; i++) {
      sizes[i] += (targetSizes[i] - sizes[i]) * f;
      doppler[i] += (targetDoppler[i] - doppler[i]) * f;
    }
    colorAttr.needsUpdate = true;
    sizeAttr.needsUpdate = true;
    dopplerAttr.needsUpdate = true;
  }

  // Per-template uniforms (eased).
  const u = material.uniforms;
  const e = lerpRate(0.04, dt);
  u.uFlow.value += (tpl.flow - u.uFlow.value) * e;
  u.uDoppler.value += (tpl.doppler - u.uDoppler.value) * e;
  if (stardust) u.uScatter.value.set(g0.open, g1.open);
  else u.uScatter.value.set(ctrl.scatter, ctrl.scatter);
  const occ = occluder.scale.x + (Math.max(tpl.occluder, 0.001) - occluder.scale.x) * lerpRate(0.05, dt);
  occluder.scale.setScalar(occ);
}

// ===========================================================================
// UI wiring
// ===========================================================================
const templateButtons = [...document.querySelectorAll('[data-template]')];
const themeRow = document.getElementById('theme-row');
const customTheme = document.getElementById('custom-theme');

// Theme chips (inserted before the custom color picker).
const themeButtons = Object.entries(THEMES).map(([key, theme]) => {
  const btn = document.createElement('button');
  btn.className = 'theme';
  btn.dataset.theme = key;
  btn.title = `${theme.label} theme`;
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = cssGradient(theme.colors || NATURAL.galaxy);
  const label = document.createElement('span');
  label.className = 'label-text';
  label.textContent = theme.label;
  btn.append(swatch, label);
  btn.addEventListener('click', () => setTheme(key));
  themeRow.insertBefore(btn, customTheme);
  return btn;
});

function setTemplate(key) {
  if (!TEMPLATES[key]) return;
  if (key !== state.template) {
    state.template = key;
    state.resetFeedForward = true;
    updateTargets(now());
    // Morph into the new mode's default shape. Groups driven by a hand stay
    // under hand control; held groups take the new default immediately
    // (the springs make the morph itself smooth).
    tracker.resetGroups(defaultOpen());
    tracker.groups.forEach((tg, i) => {
      if (tg.trackId === null) ctrl.groups[i].open = ctrl.groups[i].smooth = defaultOpen();
    });
  }
  templateButtons.forEach((b) => b.classList.toggle('active', b.dataset.template === key));
}

/** R / Reset: smoothly return held particles to this mode's default shape. */
function resetShape() {
  tracker.resetGroups(defaultOpen());
}

function setTheme(key) {
  state.theme = key;
  themeButtons.forEach((b) => b.classList.toggle('active', b.dataset.theme === key));
  customTheme.classList.toggle('active', key === 'custom');
  updateTargets(now());
}

function setBreathing(on) {
  state.breathing = on;
  if (on) {
    state.breathStart = now();
    lastBreathLabel = '';
  }
  ui.breath.classList.toggle('on', on);
  const btn = document.getElementById('breath-toggle');
  btn.classList.toggle('active', on);
  btn.setAttribute('aria-pressed', String(on));
}

function setCollapsed(on) {
  document.body.classList.toggle('collapsed', on);
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

templateButtons.forEach((b) => b.addEventListener('click', () => setTemplate(b.dataset.template)));
document.getElementById('color').addEventListener('input', (e) => {
  state.customColor = e.target.value;
  setTheme('custom');
});
document.getElementById('breath-toggle').addEventListener('click', () => setBreathing(!state.breathing));
document.getElementById('fullscreen').addEventListener('click', toggleFullscreen);
document.getElementById('collapse').addEventListener('click', () => setCollapsed(true));
document.getElementById('reset').addEventListener('click', resetShape);
document.getElementById('expand').addEventListener('click', () => setCollapsed(false));
document.addEventListener('fullscreenchange', () => {
  document.getElementById('fullscreen').textContent = document.fullscreenElement ? 'Exit' : 'Fullscreen';
});

// Any key or pointer input counts as activity (wakes the display from idle dimming).
const markUserActivity = () => { state.lastUserActivity = now(); };
window.addEventListener('pointerdown', markUserActivity);
window.addEventListener('pointermove', markUserActivity, { passive: true });

window.addEventListener('keydown', (e) => {
  markUserActivity();
  // Ignore shortcuts only while typing into a text field (sliders / color are fine).
  if (e.target instanceof HTMLInputElement && !['range', 'color', 'checkbox'].includes(e.target.type)) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key >= '1' && key <= String(TEMPLATE_DEFS.length)) setTemplate(TEMPLATE_DEFS[Number(key) - 1].key);
  else if (key === 't') {
    const keys = Object.keys(THEMES);
    setTheme(keys[(keys.indexOf(state.theme) + 1) % keys.length]);
  } else if (key === 'b') setBreathing(!state.breathing);
  else if (key === 'f') toggleFullscreen();
  else if (key === 'h') setCollapsed(!document.body.classList.contains('collapsed'));
  else if (key === 'd') debugPanel.toggle();
  else if (key === 'r') resetShape();
});

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(window.innerWidth, window.innerHeight);
  material.uniforms.uPixelRatio.value = renderer.getPixelRatio();
  sky.material.uniforms.uPixelRatio.value = renderer.getPixelRatio();
}
window.addEventListener('resize', onResize);

// ===========================================================================
// Adaptive quality: scale the particle count to hold ~60 fps
// ===========================================================================
const perf = { frames: 0, time: 0, fps: 60, warmup: 3, goodWindows: 0 };

function adaptQuality(dt) {
  perf.frames++;
  perf.time += dt;
  if (perf.time < 1) return;
  perf.fps = perf.frames / perf.time;
  perf.frames = 0;
  perf.time = 0;
  ui.fpsRender.textContent = `${Math.round(perf.fps)} fps`;
  if (!ADAPTIVE || (perf.warmup -= 1) > 0) return;

  if (perf.fps < 48 && activeCount > MIN_PARTICLES) {
    activeCount = Math.max(MIN_PARTICLES, Math.floor(activeCount * 0.8));
    perf.goodWindows = 0;
  } else if (perf.fps > 58 && activeCount < N) {
    // Grow back slowly, only after sustained headroom.
    if (++perf.goodWindows >= 3) {
      activeCount = Math.min(N, Math.floor(activeCount * 1.1));
      perf.goodWindows = 0;
    }
  } else {
    perf.goodWindows = 0;
  }
  geometry.setDrawRange(0, activeCount);
}

// ===========================================================================
// Main loop
// ===========================================================================
let debugTimer = 0;
let lastFrame = now();

function animate() {
  const t = now();
  const rawDt = t - lastFrame;
  const dt = Math.min(rawDt, 0.05);   // physics step, clamped after tab switches / slow frames
  lastFrame = t;

  updateControls(t, dt);
  updateDimming(t, Math.min(rawDt, 0.5));   // wall-clock timing (fade / 1.5 s wake-up)
  simulate(t, Math.max(dt, 1e-3));
  material.uniforms.uTime.value = t;
  sky.update(t, ctrl.sway);
  composer.render();

  adaptQuality(dt);
  if ((debugTimer += dt) > 0.2) {
    debugTimer = 0;
    const idleFor = t - Math.max(tracker.lastActivity, state.lastUserActivity);
    debugPanel.update({
      tracker,
      serverThresholds: input.thresholds,
      stats: `particles ${activeCount.toLocaleString()} / ${N.toLocaleString()} · ${Math.round(perf.fps)} fps · `
        + `scatter A ${ctrl.groups[0].open.toFixed(2)} / B ${ctrl.groups[1].open.toFixed(2)} · `
        + `idle ${Math.floor(idleFor)} s · brightness ${(ctrl.brightness * 100).toFixed(0)}%`,
    });
    // Tiny hand indicator, visible even with every panel hidden.
    ui.handDot.className = `dot ${tracker.anyControlling ? 'ok' : tracker.anyStabilizing ? 'warn' : ''}`;
  }
  requestAnimationFrame(animate);
}

// Start: Stardust spread across the screen, Natural colors.
{
  const d = TEMPLATES.stardust.data;
  for (let i = 0; i < N; i++) {
    const z = d.screen[i * 3 + 2];
    const halfH = (CAMERA_Z - z) * TAN_HALF_FOV;
    positions[i * 3] = d.screen[i * 3] * halfH * camera.aspect;
    positions[i * 3 + 1] = d.screen[i * 3 + 1] * halfH;
    positions[i * 3 + 2] = z;
  }
}
setTemplate('stardust');
setTheme('natural');
colors.set(targetColors);
sizes.set(targetSizes);
doppler.set(targetDoppler);
socket.connect();
requestAnimationFrame(animate);

// Debug / automation handle (used by the headless tests).
window.particles = {
  state, input, ctrl, tracker, tuning, setTemplate, setTheme, setBreathing, setCollapsed, resetShape,
  get activeCount() { return activeCount; },
  /** Mean screen-space x/y (NDC, -1..1) of the particles in a hand group. */
  centroid(group) {
    const v = new THREE.Vector3();
    let sx = 0, sy = 0, n = 0;
    for (let i = 0; i < activeCount; i++) {
      if (group != null && groups[i] !== group) continue;
      v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).project(camera);
      sx += v.x; sy += v.y; n++;
    }
    return { x: sx / n, y: sy / n };
  },
  /** RMS distance of a group's particles from its centroid (world units). */
  spread(group) {
    let cx = 0, cy = 0, n = 0;
    for (let i = 0; i < activeCount; i++) {
      if (group != null && groups[i] !== group) continue;
      cx += positions[i * 3]; cy += positions[i * 3 + 1]; n++;
    }
    cx /= n; cy /= n;
    let s = 0;
    for (let i = 0; i < activeCount; i++) {
      if (group != null && groups[i] !== group) continue;
      s += (positions[i * 3] - cx) ** 2 + (positions[i * 3 + 1] - cy) ** 2;
    }
    return Math.sqrt(s / n);
  },
};

// ===========================================================================
// Helpers
// ===========================================================================
function clampInt(value, min, max, fallback) {
  const n = parseInt(value, 10);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
}
