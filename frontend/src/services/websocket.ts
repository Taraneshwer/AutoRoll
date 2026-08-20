import { WebSocketEventEnvelope, WebSocketStatus } from '../types';

type EventCallback = (event: WebSocketEventEnvelope) => void;
type StatusCallback = (status: WebSocketStatus) => void;

const getWsUrl = (): string => {
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/clients`;
  }
  return 'ws://localhost:8000/ws/clients';
};

export class WebSocketService {
  private static instance: WebSocketService | null = null;
  private ws: WebSocket | null = null;
  private status: WebSocketStatus = 'DISCONNECTED';
  private eventListeners: Map<string, Set<EventCallback>> = new Map();
  private statusListeners: Set<StatusCallback> = new Set();
  private reconnectTimer: any = null;
  private isExplicitClose = false;

  private constructor() {}

  public static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService();
    }
    return WebSocketService.instance;
  }

  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isExplicitClose = false;
    this.setStatus('CONNECTING');

    const wsUrl = getWsUrl();
    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.setStatus('CONNECTED');
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const envelope: WebSocketEventEnvelope = JSON.parse(event.data);
          this.notifyEventListeners(envelope);
        } catch (e) {
          console.warn('Failed to parse WebSocket event payload:', e);
        }
      };

      this.ws.onerror = (err) => {
        console.warn('WebSocket connection error:', err);
        this.setStatus('ERROR');
      };

      this.ws.onclose = () => {
        this.ws = null;
        if (!this.isExplicitClose) {
          this.setStatus('DISCONNECTED');
          this.scheduleReconnect();
        } else {
          this.setStatus('DISCONNECTED');
        }
      };
    } catch (e) {
      this.setStatus('ERROR');
      this.scheduleReconnect();
    }
  }

  public disconnect(): void {
    this.isExplicitClose = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('DISCONNECTED');
  }

  public getStatus(): WebSocketStatus {
    return this.status;
  }

  public onStatusChange(callback: StatusCallback): () => void {
    this.statusListeners.add(callback);
    callback(this.status);
    return () => {
      this.statusListeners.delete(callback);
    };
  }

  public subscribe(eventType: string, callback: EventCallback): () => void {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, new Set());
    }
    this.eventListeners.get(eventType)!.add(callback);

    return () => {
      const set = this.eventListeners.get(eventType);
      if (set) {
        set.delete(callback);
        if (set.size === 0) {
          this.eventListeners.delete(eventType);
        }
      }
    };
  }

  private setStatus(newStatus: WebSocketStatus): void {
    this.status = newStatus;
    this.statusListeners.forEach((cb) => cb(newStatus));
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer || this.isExplicitClose) return;
    this.setStatus('RECONNECTING');
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 4000);
  }

  private notifyEventListeners(envelope: WebSocketEventEnvelope): void {
    const specificListeners = this.eventListeners.get(envelope.event_type);
    if (specificListeners) {
      specificListeners.forEach((cb) => cb(envelope));
    }
    const wildcardListeners = this.eventListeners.get('*');
    if (wildcardListeners) {
      wildcardListeners.forEach((cb) => cb(envelope));
    }
  }
}
