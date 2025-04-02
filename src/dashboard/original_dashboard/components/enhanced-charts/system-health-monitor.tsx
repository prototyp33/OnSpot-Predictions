"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { chartColors, chartDefaults, gradients, tooltipStyles } from "../visualizations/styles"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartData,
  ChartOptions,
} from "chart.js"
import { Line } from "react-chartjs-2"
import { motion } from "framer-motion"
import { Activity } from "lucide-react"

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

interface SystemHealthMonitorProps {
  metrics: {
    name: string
    value: number
    previous_value?: number
    change_percentage?: number
    is_improvement?: boolean
    timestamp: string
  }[]
  loading?: boolean
  className?: string
}

export function SystemHealthMonitor({
  metrics,
  loading = false,
  className,
}: SystemHealthMonitorProps) {
  // Calculate overall system health
  const healthScores = metrics.map((metric) => {
    const normalizedValue = Math.min(Math.max(metric.value, 0), 100)
    switch (metric.name) {
      case "cpu_usage":
        return 100 - normalizedValue // Lower is better
      case "memory_usage":
        return 100 - normalizedValue // Lower is better
      case "inference_time":
        return 100 - (normalizedValue / 2) // Scale inference time to 0-100
      default:
        return normalizedValue
    }
  })

  const overallHealth = healthScores.length
    ? Math.round(healthScores.reduce((a, b) => a + b, 0) / healthScores.length)
    : 0

  const getHealthStatus = (score: number) => {
    if (score >= 80) return { label: "Healthy", variant: "success" as const, color: chartColors.success }
    if (score >= 60) return { label: "Warning", variant: "warning" as const, color: chartColors.warning }
    return { label: "Critical", variant: "destructive" as const, color: chartColors.destructive }
  }

  const status = getHealthStatus(overallHealth)

  // Prepare chart data
  const chartData: ChartData<"line"> = {
    labels: metrics.map(() => ""), // Use empty labels for cleaner look
    datasets: metrics.map((metric) => ({
      label: metric.name,
      data: [metric.previous_value || 0, metric.value],
      borderColor: metric.is_improvement ? chartColors.success : chartColors.destructive,
      backgroundColor: (context: any) => {
        const ctx = context.chart.ctx
        const gradient = ctx.createLinearGradient(0, 0, 0, 200)
        const colors = metric.is_improvement ? gradients.success : gradients.destructive
        colors.forEach((stop, index) => {
          gradient.addColorStop(index / 2, stop)
        })
        return gradient
      },
      tension: 0.4,
      fill: true,
    })),
  }

  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      ...chartDefaults.animation,
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        ...tooltipStyles,
        mode: "index",
        intersect: false,
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        beginAtZero: true,
        grid: {
          color: "rgba(0, 0, 0, 0.1)",
        },
        ticks: {
          font: chartDefaults.font,
        },
      },
    },
  }

  return (
    <Card className={cn(
      "overflow-hidden border-0 transition-all duration-300",
      "hover:shadow-lg hover:-translate-y-1",
      "bg-gradient-to-br from-background to-muted/10",
      className
    )}>
      <CardHeader className="px-6 pt-6 pb-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-xl font-medium tracking-tight">System Health</CardTitle>
          </div>
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Badge variant={status.variant} className="text-sm font-medium">
              {status.label}
            </Badge>
          </motion.div>
        </div>
      </CardHeader>
      <CardContent className="p-6">
        {loading ? (
          <div className="space-y-4">
            <Skeleton className="h-[100px] w-full rounded-lg" />
            <div className="grid grid-cols-3 gap-4">
              <Skeleton className="h-20 rounded-lg" />
              <Skeleton className="h-20 rounded-lg" />
              <Skeleton className="h-20 rounded-lg" />
            </div>
          </div>
        ) : (
          <motion.div 
            className="space-y-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center justify-center">
              <motion.div 
                className="relative w-32 h-32"
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <svg className="w-full h-full" viewBox="0 0 100 100">
                  <circle
                    className="text-muted stroke-current"
                    strokeWidth="10"
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    opacity="0.2"
                  />
                  <motion.circle
                    className={cn("stroke-current", {
                      "text-success": status.variant === "success",
                      "text-warning": status.variant === "warning",
                      "text-destructive": status.variant === "destructive",
                    })}
                    strokeWidth="10"
                    strokeLinecap="round"
                    cx="50"
                    cy="50"
                    r="40"
                    fill="none"
                    initial={{ strokeDasharray: "0, 251.2" }}
                    animate={{ strokeDasharray: `${overallHealth * 2.51}, 251.2` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    style={{
                      transform: "rotate(-90deg)",
                      transformOrigin: "50% 50%",
                    }}
                  />
                </svg>
                <motion.div 
                  className="absolute inset-0 flex items-center justify-center"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.5 }}
                >
                  <div className="text-center">
                    <div className="text-3xl font-bold tracking-tight">{overallHealth}%</div>
                    <div className="text-sm text-muted-foreground">Health Score</div>
                  </div>
                </motion.div>
              </motion.div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {metrics.map((metric, index) => (
                <motion.div
                  key={metric.name}
                  className={cn(
                    "p-4 rounded-lg transition-all duration-300",
                    "hover:shadow-md hover:-translate-y-1",
                    "bg-gradient-to-br from-muted/50 to-muted/30"
                  )}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <div className="text-sm font-medium mb-2">{metric.name}</div>
                  <div className="text-2xl font-bold tracking-tight">{metric.value.toFixed(1)}</div>
                  {metric.change_percentage && (
                    <div
                      className={cn(
                        "text-sm font-medium",
                        metric.is_improvement
                          ? "text-success"
                          : "text-destructive"
                      )}
                    >
                      {metric.change_percentage > 0 ? "+" : ""}
                      {metric.change_percentage.toFixed(1)}%
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            <motion.div 
              className="h-[100px]"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <Line data={chartData} options={options} />
            </motion.div>
          </motion.div>
        )}
      </CardContent>
    </Card>
  )
} 