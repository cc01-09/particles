/**
 * Gesture calibration panel (toggle with the D key).
 *
 * Shows live per-hand openness, raw tip ratio and palm coordinates, and lets
 * you tune FIST_RATIO / OPEN_RATIO with sliders. New values are sent to the
 * inference server and saved in localStorage so they survive reloads.
 */
const STORAGE_KEY = 'particles.thresholds';
export const DEFAULT_THRESHOLDS = { fist_ratio: 1.0, open_ratio: 1.75 };

export function loadThresholds() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const v = JSON.parse(raw);
    if (Number.isFinite(v.fist_ratio) && Number.isFinite(v.open_ratio)) return v;
  } catch {
    // storage unavailable (private mode, blocked site data): use defaults
  }
  return null;
}

function saveThresholds(v) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(v));
  } catch {
    // ignore: the values still apply for this session
  }
}

function clearThresholds() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}

export class DebugPanel {
  /**
   * @param {HTMLElement} root   panel element
   * @param {(t: {fist_ratio:number, open_ratio:number}) => void} onChange
   */
  constructor(root, onChange) {
    this.root = root;
    this.onChange = onChange;
    this.visible = false;
    this.handsEl = root.querySelector('[data-debug="hands"]');
    this.statsEl = root.querySelector('[data-debug="stats"]');
    this.serverEl = root.querySelector('[data-debug="server"]');
    this.fist = root.querySelector('#fist-ratio');
    this.open = root.querySelector('#open-ratio');
    this.fistOut = root.querySelector('#fist-ratio-value');
    this.openOut = root.querySelector('#open-ratio-value');

    const initial = loadThresholds() || DEFAULT_THRESHOLDS;
    this.setSliders(initial);

    const changed = () => {
      // Keep OPEN above FIST.
      let f = parseFloat(this.fist.value);
      let o = parseFloat(this.open.value);
      if (o < f + 0.05) o = f + 0.05;
      const v = { fist_ratio: +f.toFixed(2), open_ratio: +o.toFixed(2) };
      this.setSliders(v);
      saveThresholds(v);
      this.onChange(v);
    };
    this.fist.addEventListener('input', changed);
    this.open.addEventListener('input', changed);
    root.querySelector('#thresholds-reset').addEventListener('click', () => {
      clearThresholds();
      this.setSliders(DEFAULT_THRESHOLDS);
      this.onChange({ ...DEFAULT_THRESHOLDS });
    });
  }

  get thresholds() {
    return { fist_ratio: parseFloat(this.fist.value), open_ratio: parseFloat(this.open.value) };
  }

  setSliders(v) {
    this.fist.value = v.fist_ratio;
    this.open.value = v.open_ratio;
    this.fistOut.textContent = Number(v.fist_ratio).toFixed(2);
    this.openOut.textContent = Number(v.open_ratio).toFixed(2);
  }

  toggle(force) {
    this.visible = force ?? !this.visible;
    this.root.hidden = !this.visible;
  }

  /** Refresh live values (called a few times per second). */
  update({ hands, serverThresholds, stats }) {
    if (!this.visible) return;
    this.handsEl.innerHTML = hands.length
      ? hands.map((h) => `
          <div class="dbg-hand">
            <span>${h.label}</span>
            <span>open <b>${h.openness.toFixed(2)}</b></span>
            <span>ratio ${h.ratio != null ? h.ratio.toFixed(2) : '—'}</span>
            <span>x ${h.center[0].toFixed(2)} · y ${h.center[1].toFixed(2)}</span>
            <div class="dbg-bar"><div style="width:${(h.openness * 100).toFixed(0)}%"></div></div>
          </div>`).join('')
      : '<div class="dbg-empty">No hands detected</div>';
    this.serverEl.textContent = serverThresholds
      ? `server: fist ${serverThresholds.fist_ratio.toFixed(2)} · open ${serverThresholds.open_ratio.toFixed(2)}`
      : 'server: not connected';
    this.statsEl.textContent = stats;
  }
}
