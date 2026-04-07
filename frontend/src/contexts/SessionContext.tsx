import {
  createContext,
  useContext,
  useRef,
  useState,
  useCallback,
  type ReactNode,
} from 'react';
import type { ChatMessage, Session } from '@/lib/types';
import { readSessions, upsertSession, deleteSessionById } from '@/lib/sessions';

interface SessionContextType {
  sessions: Session[];
  currentId: string | null;
  sessionKey: number;
  persistMessages: (messages: ChatMessage[]) => void;
  startNew: () => void;
  openSession: (id: string) => void;
  removeSession: (id: string) => void;
}

const Ctx = createContext<SessionContextType | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<Session[]>(readSessions);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [sessionKey, setSessionKey] = useState(0);
  const activeRef = useRef<string | null>(null);

  const persistMessages = useCallback((messages: ChatMessage[]) => {
    if (messages.length === 0) return;
    const firstUser = messages.find((m) => m.role === 'user');
    if (!firstUser) return;

    if (!activeRef.current) {
      activeRef.current = `s_${Date.now()}`;
    }

    const raw = firstUser.content;
    const title = raw.length > 50 ? raw.slice(0, 47) + '…' : raw;

    const session: Session = {
      id: activeRef.current,
      title,
      createdAt: Date.now(),
      messages,
    };
    const updated = upsertSession(session);
    setSessions(updated);
  }, []);

  const startNew = useCallback(() => {
    activeRef.current = null;
    setCurrentId(null);
    setSessionKey((k) => k + 1);
  }, []);

  const openSession = useCallback((id: string) => {
    activeRef.current = id;
    setCurrentId(id);
  }, []);

  const removeSession = useCallback((id: string) => {
    const updated = deleteSessionById(id);
    setSessions(updated);
    if (activeRef.current === id) {
      activeRef.current = null;
      setCurrentId(null);
      setSessionKey((k) => k + 1);
    }
  }, []);

  return (
    <Ctx.Provider
      value={{ sessions, currentId, sessionKey, persistMessages, startNew, openSession, removeSession }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useSession(): SessionContextType {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useSession must be used within SessionProvider');
  return ctx;
}
