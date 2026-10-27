/**
 * Tunable constants (motion feel, hand stability, idle dimming).
 *
 * Every value here has a default constant and a slider in the D-key debug
 * panel. Adjusted values are stored in localStorage (wrapped in try/catch)
 * and loaded on the next visit.
 */

/** Fixed constants (not exposed as sliders). */
export const HISTORY_SECONDS = 1.2;        // per-hand state history kept for exit look-back
export const LOST_TIMEOUT = 0.2;           // s without a fresh detection before a hand counts as gone
export const RETRO_MARGIN = 0.15;          // s subtracted before the start of a fast exit motion
export const WAKE_SECONDS = 1.5;           // brightness recovery time when activity resumes
export const BRIGHT_STAR_FRACTION = 0.04;  // share of particles that stay bright while dimmed
export const ACTIVITY_OPEN_DELTA = 0.08;   // openness change that counts as "activity"
export const ACTIVITY_SPEED = 0.15;        // palm speed (frame widths / s) that counts as activity

/** Slider-backed constants, grouped for the debug panel. */
export const TUNING_SCHEMA = [
  // --- Motion feel (about 55% of the previous speed) ---
  { key: 'springStiffness', group: 'Motion', label: 'Spring stiffness', min: 0.2, max: 1.5, step: 0.05, def: 0.55,
    help: 'Multiplier on each particle\'s spring frequency (1.0 = previous, faster version).' },
  { key: 'springDamping', group: 'Motion', label: 'Spring damping ζ', min: 0.4, max: 1.2, step: 0.02, def: 0.85,
    help: '< 1 overshoots slightly, 1 = critically damped.' },
  { key: 'followDelay', group: 'Motion', label: 'Palm follow delay (s)', min: 0.02, max: 1.5, step: 0.02, def: 0.12,
    help: 'Time constant for the dust cloud center to follow the palm.' },
  { key: 'maxScatterSpeed', group: 'Motion', label: 'Max scatter speed (/s)', min: 0.1, max: 3, step: 0.05, def: 0.7,
    help: 'Upper limit on how fast the scatter amount can change (full range = 1).' },
  { key: 'scatterSmoothing', group: 'Motion', label: 'Scatter smoothing (s)', min: 0.02, max: 1, step: 0.02, def: 0.15,
    help: 'Low-pass on the hand openness before the speed limit.' },

  // --- Hand stability (raise / lower without side effects) ---
  { key: 'activationTime', group: 'Hand stability', label: 'Activation time (s)', min: 0, max: 1.5, step: 0.05, def: 0.4,
    help: 'A new hand must stay steady this long before it controls particles.' },
  { key: 'stableSpeed', group: 'Hand stability', label: 'Steady below (frame/s)', min: 0.05, max: 1.5, step: 0.05, def: 0.35,
    help: 'Palm speed under which a new hand counts as steady.' },
  { key: 'fastSpeed', group: 'Hand stability', label: 'Fast move above (frame/s)', min: 0.3, max: 3, step: 0.05, def: 1.0,
    help: 'Above this palm speed, openness updates pause.' },
  { key: 'retroLookback', group: 'Hand stability', label: 'Exit look-back (s)', min: 0.2, max: 1.0, step: 0.05, def: 0.8,
    help: 'Window before a hand disappears that is checked for a fast exit motion.' },

  // --- Idle dimming ---
  { key: 'idleSeconds', group: 'Idle dimming', label: 'Idle before dimming (s)', min: 5, max: 300, step: 5, def: 60,
    help: 'No hand, or no noticeable gesture change, for this long starts dimming.' },
  { key: 'dimDuration', group: 'Idle dimming', label: 'Dimming duration (s)', min: 5, max: 120, step: 5, def: 45,
    help: 'Time to fade from full brightness to the dimmed level.' },
  { key: 'dimLevel', group: 'Idle dimming', label: 'Dimmed brightness', min: 0.05, max: 1, step: 0.01, def: 0.18,
    help: 'Brightness while idle (fraction of normal). Background stars are not dimmed.' },
];

export const DEFAULT_TUNING = Object.fromEntries(TUNING_SCHEMA.map((s) => [s.key, s.def]));

const STORAGE_KEY = 'particles.tuning';

/** Load saved values merged over the defaults (ignores unknown / invalid keys). */
export function loadTuning() {
  const tuning = { ...DEFAULT_TUNING };
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    for (const s of TUNING_SCHEMA) {
      const v = saved[s.key];
      if (Number.isFinite(v)) tuning[s.key] = Math.min(s.max, Math.max(s.min, v));
    }
  } catch {
    // storage unavailable or corrupted: defaults
  }
  return tuning;
}

export function saveTuning(tuning) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tuning));
  } catch {
    // ignore: values still apply for this session
  }
}

export function clearTuning() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
