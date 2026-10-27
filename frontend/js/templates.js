/**
 * Celestial templates: procedural point distributions for each mode.
 *
 * Each builder fills a `TemplateData` for n particles:
 *   pos      Float32Array(n*3)  local positions (disks lie in the X-Z plane)
 *   c        Float32Array(n)    palette coordinate 0..1 (see palettes.js)
 *   b        Float32Array(n)    brightness multiplier
 *   size     Float32Array(n)    size multiplier
 *   omega    Float32Array(n)    angular speed around the local Y axis (rad/s)
 *   fixed    Uint8Array(n)      1 = not rotated/tilted (camera-facing parts)
 *   doppler  Float32Array(n)    0..1 weight of relativistic (Doppler) beaming
 *
 * Particle index order is random within every template, so any prefix
 * [0, k) is a representative subset — this is what lets the renderer drop
 * particles on slow devices without leaving holes in the shapes.
 */
import { createNoise3D, fbm, gauss, mulberry32, unitVector } from './noise.js';

function allocate(n) {
  return {
    pos: new Float32Array(n * 3),
    c: new Float32Array(n),
    b: new Float32Array(n).fill(1),
    size: new Float32Array(n).fill(1),
    omega: new Float32Array(n),
    fixed: new Uint8Array(n),
    doppler: new Float32Array(n),
  };
}

const expRandom = (random, scale) => -Math.log(1 - random()) * scale;

// ---------------------------------------------------------------------------
// Stardust: soft ivory dust that fills the screen and gathers around the hand.
// `pos` holds the offset inside a gathered cluster; `screen` holds a
// normalized position (u, v in [-1, 1], z depth) used when fully scattered.
// ---------------------------------------------------------------------------
function buildStardust(n, random) {
  const d = allocate(n);
  d.screen = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    // Cluster: dense gaussian core with a few loose outliers.
    const r = 0.34 * (0.35 + random() ** 2 * 1.9);
    d.pos[i * 3] = gauss(random) * r;
    d.pos[i * 3 + 1] = gauss(random) * r;
    d.pos[i * 3 + 2] = gauss(random) * r;
    d.screen[i * 3] = random() * 2 - 1;
    d.screen[i * 3 + 1] = random() * 2 - 1;
    d.screen[i * 3 + 2] = -2.5 + random() * 4.5;
    d.c[i] = 0.2 + random() * 0.7;
    d.b[i] = 0.3 + random() * 0.3;
    d.size[i] = 0.7;
  }
  return d;
}

// ---------------------------------------------------------------------------
// Spiral galaxy: logarithmic arms, bright bulge, dust lanes, blue clusters.
// ---------------------------------------------------------------------------
function buildGalaxy(n, random) {
  const d = allocate(n);
  const a = 0.16;        // spiral scale
  const pitch = 0.27;    // tan(pitch angle) -> r = a * e^(pitch * theta)
  const armAngle = (r, arm) => Math.log(r / a) / pitch + arm * Math.PI;

  // Pre-place young blue star clusters along the arms.
  const clusters = Array.from({ length: 42 }, () => {
    const r = 0.7 + random() * 1.9;
    const th = armAngle(r, random() < 0.5 ? 0 : 1) + gauss(random) * 0.12;
    return [r * Math.cos(th), r * Math.sin(th)];
  });

  for (let i = 0; i < n; i++) {
    const u = random();
    let x, y, z;
    if (u < 0.17) {
      // Bulge: flattened gaussian, warm and bright.
      x = gauss(random) * 0.3; y = gauss(random) * 0.17; z = gauss(random) * 0.3;
      d.c[i] = 0.22 + random() * 0.18;
      d.b[i] = 0.35 + random() * 0.3;
    } else if (u < 0.6) {
      // Arm stars: exponential radial profile, scattered around the arm.
      const r = Math.min(0.28 + expRandom(random, 0.8), 2.9);
      const th = armAngle(r, random() < 0.5 ? 0 : 1) + gauss(random) * (0.11 + 0.08 / r);
      const rr = r * (1 + gauss(random) * 0.05);
      x = rr * Math.cos(th); z = rr * Math.sin(th); y = gauss(random) * 0.035;
      d.c[i] = 0.5 + 0.45 * Math.min(1, r / 2.6) + gauss(random) * 0.05;
      d.b[i] = 0.55 + random() * 0.4;
    } else if (u < 0.74) {
      // Dust lanes: thin, dim and reddish, on the inner edge of each arm.
      const r = Math.min(0.4 + expRandom(random, 0.75), 2.7);
      const th = armAngle(r, random() < 0.5 ? 0 : 1) - 0.24 + gauss(random) * 0.06;
      x = r * Math.cos(th); z = r * Math.sin(th); y = gauss(random) * 0.015;
      d.c[i] = 0.02 + random() * 0.06;
      d.b[i] = 0.45;
      d.size[i] = 1.0;
    } else if (u < 0.79) {
      // Blue clusters: tight knots of hot young stars.
      const [cx, cz] = clusters[Math.floor(random() * clusters.length)];
      x = cx + gauss(random) * 0.05; z = cz + gauss(random) * 0.05; y = gauss(random) * 0.02;
      d.c[i] = 0.93 + random() * 0.07;
      d.b[i] = 1.3 + random() * 0.8;
      d.size[i] = 1.1;
    } else {
      // Diffuse disk.
      const r = Math.min(expRandom(random, 0.9), 3.1);
      const th = random() * Math.PI * 2;
      x = r * Math.cos(th); z = r * Math.sin(th); y = gauss(random) * 0.06;
      d.c[i] = 0.45 + 0.4 * Math.min(1, r / 3);
      d.b[i] = 0.3;
    }
    const scale = 0.8;   // fit the whole disk on screen
    d.pos.set([x * scale, y * scale, z * scale], i * 3);
    d.omega[i] = 0.045;  // rigid pattern rotation (keeps the arms from winding up)
  }
  return d;
}

// ---------------------------------------------------------------------------
// Saturn: banded oblate planet + razor-thin rings with the Cassini division.
// ---------------------------------------------------------------------------
const SATURN_R = 0.8;
const SATURN_TILT = [0.42, -0.2];

/** Rotate a point around X then Z (same convention as the template tilt). */
function tiltPoint(x, y, z, [ax, az]) {
  const y1 = y * Math.cos(ax) - z * Math.sin(ax);
  const z1 = y * Math.sin(ax) + z * Math.cos(ax);
  return [x * Math.cos(az) - y1 * Math.sin(az), x * Math.sin(az) + y1 * Math.cos(az), z1];
}

function buildSaturn(n, random) {
  const d = allocate(n);
  const R = SATURN_R;
  // Ring regions in planet radii: [inner, outer, weight, brightness, color].
  const rings = [
    [1.24, 1.53, 0.16, 0.28, 0.12],   // C ring: faint, dusky
    [1.53, 1.95, 0.5, 0.95, 0.5],     // B ring: brightest
    [1.95, 2.03, 0.012, 0.12, 0.1],   // Cassini division: almost empty
    [2.03, 2.27, 0.33, 0.62, 0.62],   // A ring
  ];
  const totalW = rings.reduce((s, r) => s + r[2], 0);

  for (let i = 0; i < n; i++) {
    let x, y, z;
    if (random() < 0.42) {
      // Planet: thin shell on the (slightly oblate) surface. The back half is
      // hidden by the black occluder sphere; the front half gets limb
      // darkening from its final, already-tilted orientation.
      const [ux, uy, uz] = unitVector(random);
      const r = R * (1 - random() * 0.02);
      const lat = uy;
      const band = 0.5 + 0.5 * Math.sin(lat * 15 + 1.4 * Math.sin(lat * 5));
      [x, y, z] = tiltPoint(ux * r, uy * r * 0.92, uz * r, SATURN_TILT);
      const facing = Math.max(0, z / R);            // 1 at disk center, 0 at the limb
      d.c[i] = Math.abs(lat) > 0.8 ? 0.9 + random() * 0.1 : 0.3 + 0.35 * band + gauss(random) * 0.03;
      d.b[i] = (0.55 + 0.5 * band) * (0.25 + 0.75 * facing ** 0.6);
      d.size[i] = 0.75;
      d.fixed[i] = 1;                                // tilt already applied
    } else {
      let pick = random() * totalW;
      let ring = rings[0];
      for (const r of rings) { if ((pick -= r[2]) <= 0) { ring = r; break; } }
      const [r0, r1, , bright, color] = ring;
      let rr = Math.sqrt(r0 * r0 + random() * (r1 * r1 - r0 * r0));
      if (rr > 2.205 && rr < 2.215) rr += 0.012;               // Encke gap
      const th = random() * Math.PI * 2;
      x = rr * R * Math.cos(th); z = rr * R * Math.sin(th); y = gauss(random) * 0.004;
      const inner = (rr - r0) / (r1 - r0);                     // brightness gradient across a ring
      d.c[i] = color + inner * 0.08;
      d.b[i] = bright * (0.8 + 0.3 * inner);
      d.size[i] = 0.7;
      d.omega[i] = 0.3 / rr ** 1.5;                             // Keplerian
    }
    d.pos.set([x, y, z], i * 3);
  }
  return d;
}

// ---------------------------------------------------------------------------
// Nebula: irregular gas sampled from 3D fractal noise, with embedded stars.
// ---------------------------------------------------------------------------
function buildNebula(n, random) {
  const d = allocate(n);
  const noise = createNoise3D(random);
  let i = 0;
  let guard = 0;
  while (i < n && guard++ < n * 80) {
    const x = (random() * 2 - 1) * 2.8;
    const y = (random() * 2 - 1) * 1.9;
    const z = (random() * 2 - 1) * 1.3;

    if (random() < 0.012) {
      // Embedded young stars: sparse, bright, slightly blue-white.
      const r2 = (x / 2.2) ** 2 + (y / 1.5) ** 2 + (z / 1.0) ** 2;
      if (r2 > 1) continue;
      d.pos.set([x, y, z], i * 3);
      d.c[i] = 0.5 + random() * 0.25;
      d.b[i] = 1.6 + random() * 1.8;
      d.size[i] = 1.0 + random() * 1.0;
      i++;
      continue;
    }

    // Gas density: fractal noise shaped by a soft ellipsoidal falloff.
    const falloff = Math.exp(-((x / 2.4) ** 2 + (y / 1.6) ** 2 + (z / 1.1) ** 2) * 1.4);
    const ridge = 1 - Math.abs(fbm(noise, x * 0.6, y * 0.6, z * 0.6, 5));     // filaments
    const blob = 0.5 + 0.5 * fbm(noise, x * 0.3 + 11, y * 0.3, z * 0.3, 2);    // large-scale clumps
    const density = Math.max(0, ridge ** 4 * 1.3 * blob - 0.08) * falloff;
    if (random() > density) continue;

    // Color: a second noise field mixes H-alpha (red/pink) and OIII (teal) regions.
    const mix = 0.5 + 0.5 * fbm(noise, x * 0.35 + 40, y * 0.35, z * 0.35, 3);
    d.pos.set([x, y, z], i * 3);
    d.c[i] = mix < 0.45 ? 0.08 + mix * 0.5 : 0.55 + (mix - 0.45) * 0.8;
    d.b[i] = 0.07 + density * 0.32;
    d.size[i] = 1.0 + random() * 0.9;   // soft, larger "gas" sprites
    i++;
  }
  for (; i < n; i++) d.pos.set(unitVector(random).map((v) => v * 0.5), i * 3);  // fallback (unlikely)
  for (let k = 0; k < n; k++) d.omega[k] = 0.012;
  return d;
}

// ---------------------------------------------------------------------------
// Black hole: dark shadow, thin photon ring, hot rotating accretion disk,
// lensed image of the far side of the disk above and below the shadow, and
// Doppler beaming (the approaching side is brighter).
// ---------------------------------------------------------------------------
export const BLACK_HOLE_R = 0.6;
function buildBlackHole(n, random) {
  const d = allocate(n);
  const rIn = 1.05;
  const rOut = 3.2;
  for (let i = 0; i < n; i++) {
    const u = random();
    let x, y, z;
    if (u < 0.07) {
      // Photon ring: thin, bright circle facing the camera.
      const th = random() * Math.PI * 2;
      const r = BLACK_HOLE_R * 1.13 + gauss(random) * 0.006;
      x = r * Math.cos(th); y = r * Math.sin(th); z = 0;
      d.c[i] = 0.5; d.b[i] = 2.4; d.size[i] = 0.5;
      d.fixed[i] = 1; d.doppler[i] = 0.6;
    } else if (u < 0.3) {
      // Gravitational lensing: the far side of the disk appears bent over
      // the top (large arc) and under the bottom (thin arc) of the shadow.
      const top = random() < 0.7;
      const th = top ? random() * Math.PI : Math.PI + random() * Math.PI;
      const r = top ? BLACK_HOLE_R * 1.28 + Math.abs(gauss(random)) * 0.22
        : BLACK_HOLE_R * 1.2 + Math.abs(gauss(random)) * 0.07;
      x = r * Math.cos(th) * 1.05; y = r * Math.sin(th); z = 0.01;
      const heat = Math.min(1, (r - BLACK_HOLE_R) / 0.6);
      d.c[i] = 0.5 + heat * 0.35;
      d.b[i] = (top ? 1.5 : 1.0) * Math.abs(Math.sin(th)) ** 0.5 * (1.3 - heat * 0.7);
      d.size[i] = 0.6;
      d.fixed[i] = 1; d.doppler[i] = 1;
    } else {
      // Accretion disk: hotter and brighter towards the center.
      const r = rIn + (rOut - rIn) * random() ** 1.7;
      const th = random() * Math.PI * 2;
      x = r * Math.cos(th); z = r * Math.sin(th); y = gauss(random) * 0.004 * r;
      const t = (r - rIn) / (rOut - rIn);
      const streak = 0.75 + 0.25 * Math.sin(th * 7 + r * 9);   // turbulent streaks
      d.c[i] = 0.47 + t * 0.53;
      d.b[i] = (1.5 * (rIn / r) ** 1.6 + 0.08) * streak;
      d.size[i] = 0.6;
      d.omega[i] = 0.9 * (rIn / r) ** 1.5;                      // Keplerian: inner orbits faster
      d.doppler[i] = 1;
    }
    d.pos.set([x, y, z], i * 3);
  }
  return d;
}

// ---------------------------------------------------------------------------
// Globular star cluster: Plummer profile (very dense core, sparse halo).
// ---------------------------------------------------------------------------
function buildCluster(n, random) {
  const d = allocate(n);
  const a = 0.26;  // Plummer scale radius
  for (let i = 0; i < n; i++) {
    let r;
    do { r = a / Math.sqrt(random() ** (-2 / 3) - 1); } while (!(r < 3.0));
    const [ux, uy, uz] = unitVector(random);
    d.pos.set([ux * r, uy * r, uz * r], i * 3);
    const u = random();
    if (u < 0.12) {                // red giants: big, bright, orange
      d.c[i] = 0.16 + random() * 0.12; d.b[i] = 1.6; d.size[i] = 1.5;
    } else if (u < 0.17) {         // blue stragglers
      d.c[i] = 0.9 + random() * 0.1; d.b[i] = 1.3; d.size[i] = 1.1;
    } else {                       // main sequence: yellow-white, mostly faint
      d.c[i] = 0.42 + gauss(random) * 0.08; d.b[i] = 0.6;
    }
    d.b[i] *= 0.2 + 0.8 * Math.min(1, r / 0.6);      // keep the dense core from saturating
    d.omega[i] = 0.02;
  }
  return d;
}

/**
 * Template catalogue. Metadata:
 *   tilt     : Euler rotation (x, z) applied after the spin around local Y
 *   flow     : GPU curl-noise amplitude while formed (gas drift)
 *   occluder : radius of an opaque black sphere (planet body / event horizon)
 *   doppler  : enables Doppler beaming in the shader
 */
export const TEMPLATE_DEFS = [
  { key: 'stardust', label: 'Stardust', build: buildStardust, tilt: [0, 0], flow: 0.18, occluder: 0, doppler: 0 },
  { key: 'galaxy', label: 'Spiral Galaxy', build: buildGalaxy, tilt: [1.0, 0.35], flow: 0.02, occluder: 0, doppler: 0 },
  { key: 'saturn', label: 'Saturn', build: buildSaturn, tilt: SATURN_TILT, flow: 0.0, occluder: SATURN_R * 0.95, doppler: 0 },
  { key: 'nebula', label: 'Nebula', build: buildNebula, tilt: [0.2, 0.1], flow: 0.12, occluder: 0, doppler: 0 },
  { key: 'blackhole', label: 'Black Hole', build: buildBlackHole, tilt: [0.13, -0.05], flow: 0.0, occluder: BLACK_HOLE_R, doppler: 1 },
  { key: 'cluster', label: 'Star Cluster', build: buildCluster, tilt: [0, 0], flow: 0.01, occluder: 0, doppler: 0 },
];

/** Build every template for n particles (deterministic per template). */
export function buildTemplates(n) {
  const out = {};
  TEMPLATE_DEFS.forEach((def, idx) => {
    const data = def.build(n, mulberry32(1000 + idx * 7919));
    // Pre-compute the tilt matrix (rotate around X, then Z).
    const [ax, az] = def.tilt;
    const cx = Math.cos(ax), sx = Math.sin(ax), cz = Math.cos(az), sz = Math.sin(az);
    data.tilt = [cz, -sz * cx, sz * sx, sz, cz * cx, -cz * sx, 0, sx, cx];  // row-major 3x3
    out[def.key] = { ...def, data };
  });
  return out;
}
