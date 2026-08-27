import { decisionClass, priorityClass, severityClass } from "../utils/format";

export function Badge({ children, type = "priority" }) {
  const className =
    type === "decision"
      ? decisionClass(children)
      : type === "severity"
        ? severityClass(children)
        : priorityClass(children);
  return <span className={`badge ${className}`}>{children}</span>;
}
