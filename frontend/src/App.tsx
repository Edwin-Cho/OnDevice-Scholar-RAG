import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from '@/contexts/AuthContext';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { SessionProvider, useSession } from '@/contexts/SessionContext';
import Layout from '@/components/Layout';
import LoginPage from '@/pages/LoginPage';
import QueryPage from '@/pages/QueryPage';
import DocumentsPage from '@/pages/DocumentsPage';
import AdminPage from '@/pages/AdminPage';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
}

function QueryPageWithKey() {
  const { currentId, sessionKey } = useSession();
  return <QueryPage key={currentId ?? `new-${sessionKey}`} />;
}

function AppRoutes() {
  const { isAuthenticated } = useAuth();
  return (
    <Routes>
      <Route path="/login" element={isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />} />
      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<QueryPageWithKey />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/admin" element={<AdminPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AuthProvider>
          <SessionProvider>
            <AppRoutes />
          </SessionProvider>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}
