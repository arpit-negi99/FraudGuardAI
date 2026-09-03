import {
  Activity,
  AlertTriangle,
  ClipboardList,
  CreditCard,
  Info,
  LayoutDashboard,
  Settings,
} from "lucide-react";

const nav = [
  ["Dashboard", LayoutDashboard],
  ["Transactions", ClipboardList],
  ["Review Queue", AlertTriangle],
  ["Payment Incidents", CreditCard],
  ["Risk Monitor", Activity],
  ["Policy", Settings],
  ["About", Info],
];

export function AppShell({ page, setPage, health, children }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <div className="brand-mark">FG</div>
          <h1>FraudGuard AI</h1>
          <p>Merchant Risk Console</p>
        </div>
        <nav>
          {nav.map(([item, Icon]) => (
            <button
              key={item}
              className={page === item ? "active" : ""}
              onClick={() => setPage(item)}
            >
              <Icon size={18} />
              <span>{item}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-status">
          <span>System Status</span>
          <strong><i />{health?.status === "ok" ? "Ready" : "Checking"}</strong>
          <small>Policy</small>
          <b>Balanced</b>
          <small>Review threshold</small>
          <b>0.60</b>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <h2>{page}</h2>
            <p>Defense-only decision support</p>
          </div>
          <div className="topbar-pill">Human-in-the-loop</div>
        </header>
        {children}
      </main>
    </div>
  );
}
