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
          <span>Model Online</span>
          <strong>{health?.status === "ok" ? "Ready" : "Checking"}</strong>
          <small>Policy: Balanced</small>
          <small>Threshold: 0.60</small>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <p>Defense-only fraud decision support</p>
            <h2>{page}</h2>
          </div>
          <div className="topbar-pill">ALLOW / REVIEW</div>
        </header>
        {children}
      </main>
    </div>
  );
}
