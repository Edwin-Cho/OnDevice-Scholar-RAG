import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import type { UserRole } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { useSession } from '@/contexts/SessionContext';
import { useEffect, useState } from 'react';
import { MessageSquare, FolderOpen, Settings, LogOut, Wifi, WifiOff, Sun, Moon, Plus, Trash2, Clock } from 'lucide-react';
import LiELogo from '@/components/LiELogo';

const allNavItems = [
  { to: '/', icon: MessageSquare, label: 'Query', minRole: 'researcher' as UserRole },
  { to: '/documents', icon: FolderOpen, label: 'Documents', minRole: 'lab_pi' as UserRole },
  { to: '/admin', icon: Settings, label: 'Admin', minRole: 'admin' as UserRole },
];

const ROLE_LEVEL: Record<UserRole, number> = { researcher: 1, lab_pi: 2, admin: 3 };

function hasAccess(userRole: UserRole | null, minRole: UserRole): boolean {
  if (!userRole) return false;
  return ROLE_LEVEL[userRole] >= ROLE_LEVEL[minRole];
}

function HealthBadge() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/api/health', { signal: AbortSignal.timeout(3000) });
        setOnline(res.ok);
      } catch {
        setOnline(false);
      }
    };
    void check();
    const id = setInterval(() => void check(), 30000);
    return () => clearInterval(id);
  }, []);

  if (online === null) return <span className="w-2 h-2 bg-slate-600 rounded-full animate-pulse" />;
  return online ? (
    <span className="flex items-center gap-1.5 text-xs text-emerald-400">
      <Wifi className="w-3 h-3" /> Online
    </span>
  ) : (
    <span className="flex items-center gap-1.5 text-xs text-red-400">
      <WifiOff className="w-3 h-3" /> Offline
    </span>
  );
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export default function Layout() {
  const { logout, role } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const { sessions, currentId, startNew, openSession, removeSession } = useSession();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleNewChat = () => {
    startNew();
    navigate('/');
  };

  const handleOpenSession = (id: string) => {
    openSession(id);
    navigate('/');
  };

  return (
    <div className="flex h-screen text-white overflow-hidden" style={{ background: 'var(--layout-bg)' }}>
      {/* Sidebar */}
      <aside className="w-60 flex flex-col shrink-0 border-r"
        style={{ background: 'var(--sidebar-bg)', borderColor: 'var(--sidebar-border)' }}>

        <div className="flex items-center justify-between px-5 py-4 border-b" style={{ borderColor: 'var(--sidebar-border)' }}>
          <LiELogo />
          <div className="mt-0.5"><HealthBadge /></div>
        </div>

        <nav className="px-3 py-4 space-y-0.5 shrink-0">
          {allNavItems.filter(({ minRole }) => hasAccess(role, minRole)).map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-150 ${
                  isActive
                    ? 'bg-slate-800 text-slate-100 font-medium'
                    : 'text-slate-500 hover:text-slate-200 hover:bg-slate-800/70'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : ''}`} />
                  {label}
                  {isActive && <span className="ml-auto w-1.5 h-1.5 bg-emerald-400 rounded-full" />}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Session History */}
        <div className="flex-1 flex flex-col min-h-0 px-3 border-t" style={{ borderColor: 'var(--sidebar-border)' }}>
          <button
            onClick={handleNewChat}
            className="flex items-center gap-2 w-full px-3 py-2.5 mt-3 mb-2 rounded-xl text-sm text-slate-400 hover:text-slate-100 hover:bg-slate-800/70 transition-all duration-150 border border-dashed border-slate-700/60 hover:border-slate-600"
          >
            <Plus className="w-3.5 h-3.5" />
            New Chat
          </button>

          <div className="flex items-center gap-1.5 px-1 mb-1.5">
            <Clock className="w-3 h-3 text-slate-700" />
            <span className="text-[10px] text-slate-700 font-medium uppercase tracking-wider">Recent</span>
          </div>

          <div className="overflow-y-auto flex-1 space-y-0.5 pb-2">
            {sessions.length === 0 ? (
              <p className="text-xs text-slate-700 px-2 py-1.5">No sessions yet</p>
            ) : (
              sessions.map((s) => (
                <div
                  key={s.id}
                  onClick={() => handleOpenSession(s.id)}
                  className={`group flex items-start gap-2 px-2.5 py-2 rounded-lg cursor-pointer transition-all duration-150 ${
                    currentId === s.id
                      ? 'bg-slate-800 text-slate-200'
                      : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-xs truncate leading-tight">{s.title}</p>
                    <p className="text-[10px] text-slate-700 mt-0.5">{formatRelativeTime(s.createdAt)}</p>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); removeSession(s.id); }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 mt-0.5 rounded text-slate-600 hover:text-red-400 transition shrink-0"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="px-3 py-4 border-t space-y-0.5" style={{ borderColor: 'var(--sidebar-border)' }}>
          <button
            onClick={toggleTheme}
            className="flex items-center gap-3 px-3 py-2.5 w-full rounded-xl text-sm text-slate-500 hover:text-slate-300 hover:bg-slate-800/70 transition-all duration-150"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </button>
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-3 py-2.5 w-full rounded-xl text-sm text-slate-600 hover:text-red-400 hover:bg-red-500/8 transition-all duration-150"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
