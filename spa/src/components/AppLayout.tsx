import { Link, Outlet } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import TenantSelector from './TenantSelector';
import styles from './AppLayout.module.css';

export default function AppLayout() {
  const { logout } = useAuth();

  return (
    <div className={styles.layout}>
      <header className={styles.header}>
        <Link to="/" className={styles.brand}>
          SmartTerminal
        </Link>
        <div className={styles.headerRight}>
          <TenantSelector />
          <button
            type="button"
            className={styles.logoutButton}
            onClick={logout}
          >
            Logout
          </button>
        </div>
      </header>
      <main className={styles.content}>
        <Outlet />
      </main>
    </div>
  );
}
