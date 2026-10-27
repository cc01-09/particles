/**
 * Background sky: several layers of distant stars at different depths plus a
 * faint Milky Way band. Layers shift by different amounts (parallax) as the
 * view gently sways, which gives the scene depth without moving the camera.
 */
import * as THREE from 'three';
import { createNoise3D, gauss, mulberry32, unitVector } from './noise.js';

const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uPixelRatio;
  attribute vec3 aColor;
  attribute float aSize;
  attribute float aSeed;
  varying vec3 vColor;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = aSize * uPixelRatio;  // distant stars: constant pixel size
    float twinkle = 1.0 + 0.25 * sin(uTime * (0.6 + aSeed * 2.2) + aSeed * 80.0);
    vColor = aColor * twinkle;
  }
`;

const fragmentShader = /* glsl */ `
  varying vec3 vColor;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    float a = exp(-d * d * 5.0);
    gl_FragColor = vec4(vColor, a);
  }
`;

function makeLayer({ count, radius, size, brightness, random, band = null }) {
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const seeds = new Float32Array(count);
  const c = new THREE.Color();
  for (let i = 0; i < count; i++) {
    let v;
    if (band) {
      // Milky Way: stars concentrated around a tilted great circle,
      // denser and wider toward the galactic center direction.
      const lon = random() * Math.PI * 2;
      const core = Math.exp(-((lon - Math.PI) ** 2) * 0.8);
      const lat = gauss(random) * (0.07 + 0.12 * core) * (1 + 0.6 * band.noise(Math.cos(lon) * 2, Math.sin(lon) * 2, 0));
      v = [Math.cos(lon) * Math.cos(lat), Math.sin(lat), Math.sin(lon) * Math.cos(lat)];
      // tilt the band across the sky
      const [x, y, z] = v;
      const t = 0.9;
      v = [x, y * Math.cos(t) - z * Math.sin(t), y * Math.sin(t) + z * Math.cos(t)];
    } else {
      v = unitVector(random);
    }
    const r = radius * (0.85 + random() * 0.3);
    pos.set([v[0] * r, v[1] * r, v[2] * r], i * 3);
    // Star temperature: mostly white, some warm, some blue.
    const temp = random();
    c.setHSL(temp < 0.2 ? 0.08 : temp > 0.85 ? 0.6 : 0.12, temp < 0.2 || temp > 0.85 ? 0.5 : 0.15, 0.85);
    const mag = random() ** (band ? 2 : 4);
    const b = brightness * (0.35 + 0.65 * mag);
    col.set([c.r * b, c.g * b, c.b * b], i * 3);
    sizes[i] = size * (0.6 + 1.4 * mag);
    seeds[i] = random();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
  geo.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
  return geo;
}

export function createSky(pixelRatio) {
  const random = mulberry32(42);
  const material = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uPixelRatio: { value: pixelRatio } },
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const group = new THREE.Group();
  // [config, parallax factor] — nearer layers move more.
  const layers = [
    [{ count: 2600, radius: 160, size: 1.3, brightness: 0.35, random }, 0.15],
    [{ count: 1100, radius: 100, size: 1.8, brightness: 0.55, random }, 0.45],
    [{ count: 260, radius: 60, size: 2.4, brightness: 0.8, random }, 1.0],
    [{ count: 9000, radius: 150, size: 1.1, brightness: 0.16, random, band: { noise: createNoise3D(random) } }, 0.1],
  ].map(([cfg, parallax]) => {
    const points = new THREE.Points(makeLayer(cfg), material);
    points.frustumCulled = false;
    group.add(points);
    return { points, parallax };
  });

  return {
    group,
    material,
    /** sway: small {x, y} offset (world units) driving the parallax. */
    update(t, sway) {
      material.uniforms.uTime.value = t;
      for (const { points, parallax } of layers) {
        points.position.set(sway.x * parallax * 3, sway.y * parallax * 3, 0);
      }
      group.rotation.y = t * 0.003;
    },
  };
}
