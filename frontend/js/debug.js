/**
 * Debug / calibration panel (toggle with the D key).
 *
 * - Live per-hand state (Controlling / Stabilizing), openness, raw tip
 *   ratio, palm coordinates and speed; per-group state (controlled / Held)
 * - FIST_RATIO / OPEN_RATIO sliders: sent to the inference server
 * - Sliders for every tuning constant (motion, hand stability, idle dimming)
 * All values are saved in localStorage (wrapped in try/catch).
 */
import { DEFAULT_TUNING, TUNING_SCHEMA, clearTuning, saveTuning } from './tuning.js';

const THRESHOLD_KEY = 'particles.thresholds';
export const DEFAULT_THRESHOLDS = { fist_ratio: 1.0, open_ratio: 1.75 };

export function loadThresholds() {
  try {
    const v = JSON.parse(localStorage.getItem(THRESHOLD_KEY) || 'null');
    if (v && Number.isFinite(v.fist_ratio) && Number.isFinite(v.open_ratio)) return v;
  } catch {
    // storage unavailable (private mode, blocked site data): use defaults
  }
  return null;
}

function store(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore: the values still apply for this session
  }
}

function el(tag, attrs = {}, text = '') {
  const node = document.createElement(tag);
  Object.assign(node, attrs);
  if (text) node.textContent = text;
  return node;
}

function slider({ id, label, min, max, step, value, help }) {
  const wrap = el('div', { className: 'slider' });
  const lab = el('label', { htmlFor: id }, label);
  if (help) lab.title = help;
  const out = el('output', { id: `${id}-value` });
  const input = el('input', { id, type: 'range', min, max, step });
  input.value = value;
  out.textContent = Number(value).toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0);
  wrap.append(lab, out, input);
  return { wrap, input, out };
}

export class DebugPanel {
  /**
   * @param {HTMLElement} root
   * @param {object} tuning live tuning object (mutated in place)
   * @param {(t:{fist_ratio:number, open_ratio:number}) => void} onThresholds
   */
  constructor(root, tuning, onThresholds) {
    this.root = root;
    this.tuning = tuning;
    this.visible = false;
    this.handsEl = root.querySelector('[data-debug="hands"]');
    this.statsEl = root.querySelector('[data-debug="stats"]');
    this.serverEl = root.querySelector('[data-debug="server"]');
    const thresholdsEl = root.querySelector('[data-debug="thresholds"]');
    const tuningEl = root.querySelector('[data-debug="tuning"]');

    // --- FIST_RATIO / OPEN_RATIO (server-side calibration) ------------------
    const initial = loadThresholds() || DEFAULT_THRESHOLDS;
    this.fist = slider({ id: 'fist-ratio', label: 'FIST_RATIO', min: 0.5, max: 1.6, step: 0.01, value: initial.fist_ratio,
      help: 'Tip ratio at (or below) which the hand counts as a closed fist.' });
    this.open = slider({ id: 'open-ratio', label: 'OPEN_RATIO', min: 1.1, max: 2.6, step: 0.01, value: initial.open_ratio,
      help: 'Tip ratio at (or above) which the hand counts as fully open.' });
    thresholdsEl.append(this.fist.wrap, this.open.wrap);
    const thresholdsChanged = () => {
      const f = parseFloat(this.fist.input.value);
      const o = Math.max(parseFloat(this.open.input.value), f + 0.05);
      const v = { fist_ratio: +f.toFixed(2), open_ratio: +o.toFixed(2) };
      this.setThresholdSliders(v);
      store(THRESHOLD_KEY, v);
      onThresholds(v);
    };
    this.fist.input.addEventListener('input', thresholdsChanged);
    this.open.input.addEventListener('input', thresholdsChanged);

    // --- Tuning constants, grouped ---------------------------------------
    this.tuningSliders = {};
    let currentGroup = null;
    for (const s of TUNING_SCHEMA) {
      if (s.group !== currentGroup) {
        currentGroup = s.group;
        tuningEl.append(el('h3', {}, s.group));
      }
      const sl = slider({ id: `tune-${s.key}`, label: s.label, min: s.min, max: s.max, step: s.step,
        value: tuning[s.key], help: s.help });
      sl.input.addEventListener('input', () => {
        tuning[s.key] = parseFloat(sl.input.value);
        sl.out.textContent = Number(tuning[s.key]).toFixed(s.step < 0.1 ? 2 : s.step < 1 ? 1 : 0);
        saveTuning(tuning);
      });
      this.tuningSliders[s.key] = { ...sl, schema: s };
      tuningEl.append(sl.wrap);
    }

    root.querySelector('#debug-reset').addEventListener('click', () => {
      store(THRESHOLD_KEY, null);
      this.setThresholdSliders(DEFAULT_THRESHOLDS);
      onThresholds({ ...DEFAULT_THRESHOLDS });
      clearTuning();
      Object.assign(tuning, DEFAULT_TUNING);
      for (const { input, out, schema } of Object.values(this.tuningSliders)) {
        input.value = tuning[schema.key];
        out.textContent = Number(tuning[schema.key]).toFixed(schema.step < 0.1 ? 2 : schema.step < 1 ? 1 : 0);
      }
    });
  }

  setThresholdSliders(v) {
    this.fist.input.value = v.fist_ratio;
    this.open.input.value = v.open_ratio;
    this.fist.out.textContent = Number(v.fist_ratio).toFixed(2);
    this.open.out.textContent = Number(v.open_ratio).toFixed(2);
  }

  toggle(force) {
    this.visible = force ?? !this.visible;
    this.root.hidden = !this.visible;
  }

  /** Refresh live values (called a few times per second). */
  update({ tracker, serverThresholds, stats }) {
    if (!this.visible) return;
    const { hands, groups } = tracker.describe();
    const handRows = hands.map((h) => `
      <div class="dbg-hand">
        <span>Hand #${h.id} · ${h.label}</span><span class="dbg-state">${h.state}</span>
        <span>open <b>${h.openness.toFixed(2)}</b>${h.applied != null ? ` → ${h.applied.toFixed(2)}` : ''}</span>
        <span>ratio ${h.ratio != null ? h.ratio.toFixed(2) : '—'}</span>
        <span>x ${h.center[0].toFixed(2)} · y ${h.center[1].toFixed(2)}</span>
        <span>speed ${h.speed.toFixed(2)}/s</span>
        <div class="dbg-bar"><div style="width:${(h.openness * 100).toFixed(0)}%"></div></div>
      </div>`).join('');
    const groupRows = groups.map((g) => `
      <div class="dbg-group"><span>Group ${'AB'[g.index]}</span><span class="dbg-state">${g.state}</span>
        <span>open ${g.target.open.toFixed(2)} · x ${g.target.x.toFixed(2)} · y ${g.target.y.toFixed(2)}</span></div>`).join('');
    this.handsEl.innerHTML = (handRows || '<div class="dbg-empty">No hands detected</div>') + groupRows;
    this.serverEl.textContent = serverThresholds
      ? `server: fist ${serverThresholds.fist_ratio.toFixed(2)} · open ${serverThresholds.open_ratio.toFixed(2)}`
      : 'server: not connected';
    this.statsEl.textContent = stats;
  }
}
