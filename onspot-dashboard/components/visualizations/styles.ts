export const chartColors = {
  primary: "hsl(var(--primary))",
  primaryLight: "hsl(var(--primary) / 0.2)",
  success: "hsl(var(--success))",
  successLight: "hsl(var(--success) / 0.2)",
  warning: "hsl(var(--warning))",
  warningLight: "hsl(var(--warning) / 0.2)",
  destructive: "hsl(var(--destructive))",
  destructiveLight: "hsl(var(--destructive) / 0.2)",
  muted: "hsl(var(--muted))",
  mutedForeground: "hsl(var(--muted-foreground))",
  background: "hsl(var(--background))",
  foreground: "hsl(var(--foreground))",
}

export const chartDefaults = {
  font: {
    family: "var(--font-sans)",
  },
  borderWidth: 2,
  animation: {
    duration: 750,
    easing: "easeOutQuart",
  },
  transitions: {
    default: "all 0.3s ease",
    slow: "all 0.5s ease",
  },
  shadows: {
    tooltip: "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
    card: "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
  },
}

export const gradients = {
  primary: [
    "rgba(var(--primary) / 0.2)",
    "rgba(var(--primary) / 0.1)",
    "rgba(var(--primary) / 0)",
  ],
  success: [
    "rgba(var(--success) / 0.2)",
    "rgba(var(--success) / 0.1)",
    "rgba(var(--success) / 0)",
  ],
  warning: [
    "rgba(var(--warning) / 0.2)",
    "rgba(var(--warning) / 0.1)",
    "rgba(var(--warning) / 0)",
  ],
  destructive: [
    "rgba(var(--destructive) / 0.2)",
    "rgba(var(--destructive) / 0.1)",
    "rgba(var(--destructive) / 0)",
  ],
}

export const tooltipStyles = {
  backgroundColor: "hsl(var(--background))",
  borderColor: "hsl(var(--border))",
  borderWidth: 1,
  borderRadius: 8,
  padding: {
    x: 12,
    y: 8,
  },
  bodyFont: {
    size: 12,
    family: "var(--font-sans)",
    weight: "500",
  },
  titleFont: {
    size: 13,
    family: "var(--font-sans)",
    weight: "600",
  },
  boxShadow: chartDefaults.shadows.tooltip,
} 