/**
 * WebSocket client for the inference server.
 * Reconnects automatically with exponential backoff; while offline the app
 * simply keeps running in idle mode.
 */
export class HandSocket {
  /**
   * @param {string} url
   * @param {{onMessage: Function, onState: Function, onOpen?: Function}} handlers
   *   onState(kind, text): kind is 'ok' | 'warn' | '' (offline)
   */
  constructor(url, handlers) {
    this.url = url;
    this.handlers = handlers;
    this.ws = null;
    this.delay = 500;
    this.connected = false;
  }

  connect() {
    this.handlers.onState('warn', 'Connecting');
    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      console.warn('WebSocket error', err);
      this.retry();
      return;
    }
    this.ws.onopen = () => {
      this.connected = true;
      this.delay = 500;
      this.handlers.onState('ok', 'Connected');
      this.handlers.onOpen?.();
    };
    this.ws.onmessage = (event) => {
      try {
        this.handlers.onMessage(JSON.parse(event.data));
      } catch (err) {
        console.warn('Bad message', err);
      }
    };
    this.ws.onclose = () => {
      const was = this.connected;
      this.connected = false;
      this.handlers.onState('', 'Offline · retrying');
      this.handlers.onClose?.(was);
      this.retry();
    };
    this.ws.onerror = () => this.ws.close();
  }

  retry() {
    setTimeout(() => this.connect(), this.delay);
    this.delay = Math.min(this.delay * 2, 5000);
  }

  /** Send a JSON message if connected (returns false otherwise). */
  send(obj) {
    if (!this.connected || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(obj));
    return true;
  }
}
