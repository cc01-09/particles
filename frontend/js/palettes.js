/**
 * Color system.
 *
 * Every palette has 5 stops ordered by *role*, so one palette can color any
 * template. Templates assign each particle a palette coordinate c in [0, 1]:
 *
 *   c = 0.00  dust / shadow      (dark, reddish)
 *   c = 0.25  warm / hot          (cores, giants)
 *   c = 0.50  bright core         (near white)
 *   c = 0.75  body / mid
 *   c = 1.00  cool / outer        (arms, halos)
 *
 * "Natural" uses a palette per template, inspired by real astronomical images.
 * Other themes use one palette for everything; "Custom" derives one from a
 * user-picked base color.
 */
import * as THREE from 'three';

/** Natural palettes (per template), approximating true-color astrophotography. */
export const NATURAL = {
  stardust: ['#8f857c', '#efe3cf', '#fbf8f1', '#e4e9f1', '#c7d3e3'],   // ivory -> silver
  galaxy: ['#4a2016', '#ffc98a', '#fff4e0', '#d4e1ff', '#7da6ff'],     // dust, warm bulge, blue arms
  saturn: ['#4d4032', '#d6b07a', '#f4e6c4', '#c9ae86', '#9aa0a8'],     // ammonia bands, grey-blue poles
  nebula: ['#3a0b22', '#ff5f7e', '#ffd8e2', '#6fd6c8', '#3d68d8'],     // H-alpha red, OIII teal
  blackhole: ['#2a0800', '#ff8a2a', '#fff3d6', '#ffb45e', '#a8360f'],  // white-hot inner -> red outer
  cluster: ['#6e3f22', '#ffc690', '#fff7ec', '#e2eaff', '#a8c3ff'],    // red giants -> blue stragglers
};

/** Preset themes (one 5-stop palette each). `colors: null` = Natural. */
export const THEMES = {
  natural: { label: 'Natural', colors: null },
  stellar: { label: 'Stellar White', colors: ['#3b3b46', '#f3eadc', '#ffffff', '#e2e8f2', '#b6c3d6'] },
  rose: { label: 'Nebula Rose', colors: ['#3d1028', '#ff8aa6', '#ffe2eb', '#df78b6', '#8a5cc7'] },
  teal: { label: 'Cosmic Teal', colors: ['#0b2a30', '#a2f2da', '#e9fffb', '#5ccfe0', '#2f7ec0'] },
  gold: { label: 'Solar Gold', colors: ['#3a1a05', '#ffb048', '#fff2cf', '#ffd27a', '#d9772b'] },
  violet: { label: 'Deep Violet', colors: ['#1c0f3a', '#c69cff', '#f2e9ff', '#8a7bff', '#4b3fc4'] },
};

/**
 * Build a palette around a user-picked base color: hue within about ±20°,
 * with varied lightness and saturation, following the role order above.
 */
export function paletteFromBase(hex) {
  const hsl = { h: 0, s: 0, l: 0 };
  new THREE.Color(hex).getHSL(hsl);
  const s = Math.max(hsl.s, 0.35);
  const make = (dh, sat, light) =>
    `#${new THREE.Color().setHSL((hsl.h + dh / 360 + 1) % 1, sat, light).getHexString()}`;
  return [
    make(-20, s * 0.8, 0.14),          // dust: darker, hue shifted
    make(+20, s, 0.62),                // warm side
    make(0, s * 0.35, 0.93),           // bright core (desaturated)
    make(0, s, 0.68),                  // the picked hue itself
    make(-20, s * 0.9, 0.52),          // cool side
  ];
}

/** Convert hex strings to THREE.Color (converted to linear working space). */
export function toColors(hexList) {
  return hexList.map((h) => new THREE.Color(h));
}

/** Sample a palette at c in [0, 1] with linear interpolation; writes into out[0..2]. */
export function samplePalette(colors, c, out) {
  const x = Math.min(Math.max(c, 0), 1) * (colors.length - 1);
  const i = Math.min(Math.floor(x), colors.length - 2);
  const f = x - i;
  const a = colors[i];
  const b = colors[i + 1];
  out[0] = a.r + (b.r - a.r) * f;
  out[1] = a.g + (b.g - a.g) * f;
  out[2] = a.b + (b.b - a.b) * f;
  return out;
}

/** CSS gradient preview of a palette (for the theme chips). */
export function cssGradient(hexList) {
  return `linear-gradient(135deg, ${hexList.slice(1).join(', ')})`;
}
