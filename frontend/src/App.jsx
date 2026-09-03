import { useEffect, useState } from "react";
import { AppShell } from "./components/AppShell";
import { Dashboard } from "./pages/Dashboard";
import { Transactions } from "./pages/Transactions";
import { ReviewQueue } from "./pages/ReviewQueue";
import { PaymentIncidents } from "./pages/PaymentIncidents";
import { TransactionDetails } from "./pages/TransactionDetails";
import { RiskMonitor } from "./pages/RiskMonitor";
import { Policy } from "./pages/Policy";
import { About } from "./pages/About";
import { useApi } from "./hooks/useApi";
import { api } from "./services/api";

export default function App() {
  const [page, setPage] = useState("Dashboard");
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [selectedPaymentId, setSelectedPaymentId] = useState(null);
  const [events, setEvents] = useState(["Policy changed to Balanced"]);
  const [actions, setActions] = useState({});

  const health = useApi(api.health, []);
  const summary = useApi(api.riskSummary, []);
  const transactions = useApi(api.demoTransactions, []);
  const queue = useApi(api.reviewQueue, []);
  const spike = useApi(api.riskSpike, []);
  const presets = useApi(api.policyPresets, []);
  const evaluation = useApi(api.finalEvaluation, []);
  const incidentSummary = useApi(api.getIncidentSummary, []);
  const lifecycleSummary = useApi(api.getLifecycleSummary, []);
  const monitoringSummary = useApi(api.monitoringSummary, []);
  const lifecyclePreview = useApi(
    () => api.getLifecycles({ status: "ACTIVE_INCIDENT", limit: 3 }),
    [],
  );
  const detail = useApi(
    () => api.demoTransaction(selectedTransaction || defaultTransactionId(transactions.data)),
    [selectedTransaction, transactions.data],
  );

  useEffect(() => {
    if (!selectedTransaction && transactions.data?.transactions?.length) {
      const top = [...transactions.data.transactions].sort((a, b) => b.risk_score - a.risk_score)[0];
      setSelectedTransaction(top.transaction_id);
    }
  }, [transactions.data, selectedTransaction]);

  const openTransaction = (transactionId) => {
    setSelectedTransaction(transactionId);
    setPage("Transactions");
    setTimeout(() => setPage("Transaction Details"), 0);
    setEvents((items) => [`Transaction ${transactionId} opened for review`, ...items].slice(0, 5));
  };

  const setAction = (action) => {
    if (!selectedTransaction) return;
    setActions((current) => ({ ...current, [selectedTransaction]: action }));
    setEvents((items) => [`Analyst selected ${action} for transaction ${selectedTransaction}`, ...items].slice(0, 5));
  };

  const openIncident = (paymentId) => {
    setSelectedPaymentId(paymentId);
    setPage("Payment Incidents");
    setEvents((items) => [`Payment incident ${paymentId} opened`, ...items].slice(0, 5));
  };

  const rows = transactions.data?.transactions || [];

  return (
    <AppShell page={page} setPage={setPage} health={health.data}>
      {page === "Dashboard" ? (
        <Dashboard
          summary={summary.data || {}}
          transactions={rows}
          incidentSummary={incidentSummary.data || {}}
          lifecycleSummary={lifecycleSummary.data || {}}
          monitoringSummary={monitoringSummary.data || {}}
          incidentRows={lifecyclePreview.data?.lifecycles || []}
          loading={summary.loading || transactions.loading || incidentSummary.loading || lifecycleSummary.loading || monitoringSummary.loading || lifecyclePreview.loading}
          error={summary.error || transactions.error || incidentSummary.error || lifecycleSummary.error || monitoringSummary.error || lifecyclePreview.error}
          onOpen={openTransaction}
          onOpenIncident={openIncident}
          events={events}
        />
      ) : null}
      {page === "Transactions" ? (
        <Transactions transactions={rows} loading={transactions.loading} error={transactions.error} onOpen={openTransaction} />
      ) : null}
      {page === "Review Queue" ? (
        <ReviewQueue rows={queue.data?.transactions || []} loading={queue.loading} error={queue.error} onOpen={openTransaction} />
      ) : null}
      {page === "Payment Incidents" ? (
        <PaymentIncidents
          summary={incidentSummary.data || {}}
          lifecycleSummary={lifecycleSummary.data || {}}
          summaryLoading={incidentSummary.loading}
          summaryError={incidentSummary.error || lifecycleSummary.error}
          selectedPaymentId={selectedPaymentId}
          setSelectedPaymentId={setSelectedPaymentId}
        />
      ) : null}
      {page === "Transaction Details" ? (
        <TransactionDetails detail={detail.data} loading={detail.loading || transactions.loading} error={detail.error || transactions.error} action={actions[selectedTransaction]} setAction={setAction} />
      ) : null}
      {page === "Risk Monitor" ? (
        <RiskMonitor
          summary={summary.data || {}}
          transactions={rows}
          incidentSummary={incidentSummary.data || {}}
          lifecycleSummary={lifecycleSummary.data || {}}
          monitoringSummary={monitoringSummary.data || {}}
          spike={spike.data}
          loading={summary.loading || transactions.loading || spike.loading || incidentSummary.loading || lifecycleSummary.loading || monitoringSummary.loading}
          error={summary.error || transactions.error || spike.error || incidentSummary.error || lifecycleSummary.error || monitoringSummary.error}
        />
      ) : null}
      {page === "Policy" ? (
        <Policy presets={presets.data?.presets || []} loading={presets.loading} error={presets.error} />
      ) : null}
      {page === "About" ? (
        <About
          evaluation={evaluation.data}
          incidentSummary={incidentSummary.data || {}}
          monitoringSummary={monitoringSummary.data || {}}
          loading={evaluation.loading || incidentSummary.loading || monitoringSummary.loading}
          error={evaluation.error || incidentSummary.error || monitoringSummary.error}
        />
      ) : null}
    </AppShell>
  );
}

function defaultTransactionId(data) {
  if (!data?.transactions?.length) return 3481071;
  return data.transactions[0].transaction_id;
}
