/**
 * Hand tracker: turns raw per-frame detections into stable control targets.
 *
 * It is driven only by incoming WebSocket messages (`ingest`) and the clock
 * (`update`). It never looks at the DOM, so gesture handling is completely
 * independent of which panels are visible, collapsed or fullscreen.
 *
 * Per hand (track) states:
 *   'stabilizing' — just appeared; ignored until it stays steady for
 *                   `activationTime` (filters out the raise-into-frame motion)
 *   'controlling' — drives a particle group. While the palm moves fast,
 *                   openness updates pause (position still follows).
 * When a controlling hand disappears, its group is *held*: the state is
 * rolled back to just before a fast exit motion (lowering the hand), then
 * kept as is until a hand takes over again or the user presses Reset.
 *
 * Two particle groups exist (one per hand). With one hand, both follow it.
 */
import {
  ACTIVITY_OPEN_DELTA, ACTIVITY_SPEED, HISTORY_SECONDS, LOST_TIMEOUT, RETRO_MARGIN,
} from './tuning.js';

const MATCH_DISTANCE = 0.3;   // max palm jump (frame units) to keep the same track between messages

export class HandTracker {
  /** @param {object} tuning live tuning object (read on every call) */
  constructor(tuning) {
    this.tuning = tuning;
    this.tracks = [];
    this.nextId = 1;
    // target: what the group should move toward; null position = screen center
    this.groups = [0, 1].map((index) => ({
      index,
      trackId: null,
      target: { open: 1, x: 0.5, y: 0.5 },
      heldSince: null,
    }));
    this.lastActivity = performance.now() / 1000;
  }

  // -------------------------------------------------------------------------
  // Input
  // -------------------------------------------------------------------------
  /**
   * Feed one message's hands: [{label, openness, ratio, center:[x,y]}].
   * `stale` messages (the server repeating the last detection for a moment
   * after the hand vanished) carry no new information and are skipped, so
   * a hand counts as gone as soon as it really leaves.
   */
  ingest(hands, now, stale = false) {
    if (stale) return;
    const unmatched = new Set(this.tracks);
    for (const h of hands) {
      let best = null;
      let bestDist = MATCH_DISTANCE;
      for (const tr of unmatched) {
        const d = Math.hypot(h.center[0] - tr.center[0], h.center[1] - tr.center[1]);
        if (d < bestDist) { bestDist = d; best = tr; }
      }
      if (best) {
        unmatched.delete(best);
        this.updateTrack(best, h, now);
      } else {
        this.tracks.push(this.createTrack(h, now));
        this.lastActivity = now;   // a hand was just detected -> wake up
      }
    }
  }

  createTrack(h, now) {
    return {
      id: this.nextId++,
      label: h.label,
      center: [...h.center],
      openness: h.openness,
      ratio: h.ratio,
      speed: 0,
      state: 'stabilizing',
      steadySince: null,
      firstSeen: now,
      lastSeen: now,
      lastSample: now,
      fast: false,
      applied: null,               // {open, x, y} once controlling
      history: [],                 // [{t, open, x, y, speed}]
      activityOpen: h.openness,
    };
  }

  updateTrack(tr, h, now) {
    const t = this.tuning;
    const dt = Math.max(now - tr.lastSample, 1e-3);
    const inst = Math.hypot(h.center[0] - tr.center[0], h.center[1] - tr.center[1]) / dt;
    tr.speed = tr.speed * 0.5 + Math.min(inst, 10) * 0.5;     // smoothed palm speed (frame / s)
    tr.center = [...h.center];
    tr.openness = h.openness;
    tr.ratio = h.ratio;
    tr.label = h.label;
    tr.lastSeen = now;
    tr.lastSample = now;

    if (tr.state === 'stabilizing') {
      // Start control only after the hand has been steady for a while.
      if (tr.speed < t.stableSpeed) {
        tr.steadySince ??= now;
        if (now - tr.steadySince >= t.activationTime) {
          tr.state = 'controlling';
          tr.applied = { open: h.openness, x: h.center[0], y: h.center[1] };
        }
      } else {
        tr.steadySince = null;
      }
      return;
    }

    // Controlling: position always follows; openness pauses during fast moves.
    tr.fast = tr.speed > t.fastSpeed;
    tr.applied.x = h.center[0];
    tr.applied.y = h.center[1];
    if (!tr.fast) tr.applied.open = h.openness;

    tr.history.push({ t: now, open: tr.applied.open, x: tr.applied.x, y: tr.applied.y, speed: tr.speed });
    while (tr.history.length && tr.history[0].t < now - HISTORY_SECONDS) tr.history.shift();

    if (Math.abs(tr.applied.open - tr.activityOpen) > ACTIVITY_OPEN_DELTA || tr.speed > ACTIVITY_SPEED) {
      tr.activityOpen = tr.applied.open;
      this.lastActivity = now;
    }
  }

  // -------------------------------------------------------------------------
  // Per-frame bookkeeping
  // -------------------------------------------------------------------------
  update(now) {
    // Drop hands that stopped appearing; a controlling one leaves its groups held.
    for (const tr of [...this.tracks]) {
      if (now - tr.lastSeen <= LOST_TIMEOUT) continue;
      this.tracks.splice(this.tracks.indexOf(tr), 1);
      if (tr.state === 'controlling') this.holdGroupsOf(tr, now);
    }

    const controlling = this.tracks.filter((tr) => tr.state === 'controlling')
      .sort((a, b) => a.applied.x - b.applied.x);

    if (controlling.length >= 2) {
      // Two hands: left-most drives group 0, right-most drives group 1.
      this.groups[0].trackId = controlling[0].id;
      this.groups[1].trackId = controlling[1].id;
    } else if (controlling.length === 1) {
      const id = controlling[0].id;
      if (!this.groups.some((g) => g.trackId === id)) {
        // A new hand takes over every group that is not driven by another hand.
        for (const g of this.groups) if (g.trackId === null) g.trackId = id;
      }
    }

    for (const g of this.groups) {
      const tr = g.trackId !== null && this.tracks.find((x) => x.id === g.trackId);
      if (!tr || tr.state !== 'controlling') {
        if (g.trackId !== null) g.trackId = null;
        continue;
      }
      g.heldSince = null;
      g.target.open = tr.applied.open;
      g.target.x = tr.applied.x;
      g.target.y = tr.applied.y;
    }
  }

  /** A controlling hand vanished: hold its groups at the pre-exit state. */
  holdGroupsOf(tr, now) {
    const pick = this.preExitSample(tr);
    for (const g of this.groups) {
      if (g.trackId !== tr.id) continue;
      g.trackId = null;
      g.heldSince = now;
      if (pick) {
        g.target.open = pick.open;
        g.target.x = pick.x;
        g.target.y = pick.y;
      }
    }
  }

  /**
   * Look back over the last moments before the hand vanished. If it was
   * moving fast (e.g. being lowered out of view), return the state from just
   * before that fast motion started; otherwise the last state.
   */
  preExitSample(tr) {
    const h = tr.history;
    if (!h.length) return tr.applied;
    const t = this.tuning;
    const windowStart = tr.lastSeen - t.retroLookback;
    let j = -1;
    for (let i = h.length - 1; i >= 0 && h[i].t >= windowStart; i--) {
      if (h[i].speed > t.fastSpeed) { j = i; break; }
    }
    if (j < 0) return h[h.length - 1];
    // Walk back to the start of this fast run (with hysteresis).
    let start = j;
    while (start > 0 && h[start - 1].speed > t.fastSpeed * 0.5) start--;
    const cutoff = h[start].t - RETRO_MARGIN;
    for (let i = start; i >= 0; i--) if (h[i].t <= cutoff) return h[i];
    return h[0];
  }

  // -------------------------------------------------------------------------
  // Commands
  // -------------------------------------------------------------------------
  /** Reset held groups to a default openness at the screen center. */
  resetGroups(defaultOpen) {
    for (const g of this.groups) {
      if (g.trackId !== null) continue;      // a hand is controlling it right now
      g.target.open = defaultOpen;
      g.target.x = 0.5;
      g.target.y = 0.5;
      g.heldSince = null;
    }
  }

  /** Clear every hand (e.g. on disconnect): controlling hands become held. */
  dropAll(now) {
    for (const tr of this.tracks) tr.lastSeen = -Infinity;
    this.update(now);
  }

  get anyControlling() {
    return this.tracks.some((tr) => tr.state === 'controlling');
  }

  get anyStabilizing() {
    return this.tracks.some((tr) => tr.state === 'stabilizing');
  }

  /** Human-readable status for the debug panel. */
  describe() {
    const hands = this.tracks.map((tr) => ({
      id: tr.id,
      label: tr.label,
      state: tr.state === 'controlling' ? (tr.fast ? 'Controlling (fast move: openness paused)' : 'Controlling') : 'Stabilizing',
      openness: tr.openness,
      applied: tr.applied ? tr.applied.open : null,
      ratio: tr.ratio,
      center: tr.center,
      speed: tr.speed,
    }));
    const groups = this.groups.map((g) => ({
      index: g.index,
      state: g.trackId !== null ? `Controlled by hand #${g.trackId}` : g.heldSince !== null ? 'Held' : 'Default',
      target: g.target,
    }));
    return { hands, groups };
  }
}
