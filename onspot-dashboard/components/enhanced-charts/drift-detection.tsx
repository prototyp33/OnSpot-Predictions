"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { chartColors, chartDefaults, tooltipStyles } from "../visualizations/styles"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartData,
  ChartOptions,
} from "chart.js"
import { Bar } from "react-chartjs-2"
import { motion } from "framer-motion"

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
)

interface DriftDetectionProps {
  data: {
    feature: string
    p_value: number
    statistic: number
    drift_detected: boolean
    timestamp: string
  }[]
  loading?: boolean
  timeRange?: string
  className?: string
}

export function DriftDetection({
  data,
  loading = false,
  timeRange = "7d",
  className,
}: DriftDetectionProps) {
  // Group data by feature
  const features = [...new Set(data.map((d) => d.feature))]
  const groupedData = features.map((feature) => {
    const featureData = data.filter((d) => d.feature === feature)
    return {
      feature,
      driftCount: featureData.filter((d) => d.drift_detected).length,
      avgPValue: featureData.reduce((acc, d) => acc + d.p_value, 0) / featureData.length,
      maxStatistic: Math.max(...featureData.map((d) => d.statistic)),
    }
  })

  // Sort by drift count
  const sortedData = [...groupedData].sort((a, b) => b.driftCount - a.driftCount)

  const chartData: ChartData<"bar"> = {
    labels: sortedData.map((d) => d.feature),
    datasets: [
      {
        label: "Drift Count",
        data: sortedData.map((d) => d.driftCount),
        backgroundColor: chartColors.primary,
        borderRadius: 6,
        borderSkipped: false,
      },
      {
        label: "Average P-Value",
        data: sortedData.map((d) => d.avgPValue),
        backgroundColor: chartColors.warning,
        borderRadius: 6,
        borderSkipped: false,
      },
    ],
  }

  const options: ChartOptions<"bar"> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      ...chartDefaults.animation,
    },
    interaction: {
      mode: "index" as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: "top" as const,
        labels: {
          boxWidth: 12,
          usePointStyle: true,
          pointStyle: "circle",
          font: chartDefaults.font,
        },
      },
      tooltip: {
        ...tooltipStyles,
        mode: "index",
        intersect: false,
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
        ticks: {
          font: chartDefaults.font,
        },
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

  const hasDrift = data.some((d) => d.drift_detected)
  const badgeVariant = hasDrift ? "destructive" : "success"
  const badgeText = hasDrift ? "Drift Detected" : "No Drift"

  return (
    <Card className={cn(
      "overflow-hidden border-0 transition-all duration-300",
      "hover:shadow-lg hover:-translate-y-1",
      "bg-gradient-to-br from-background to-muted/10",
      className
    )}>
      <CardHeader className="px-6 pt-6 pb-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl font-medium tracking-tight">Drift Detection Analysis</CardTitle>
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Badge variant={badgeVariant} className="text-sm font-medium">
              {badgeText}
            </Badge>
          </motion.div>
        </div>
      </CardHeader>
      <CardContent className="p-6">
        {loading ? (
          <div className="space-y-4">
            <Skeleton className="h-[400px] w-full rounded-lg" />
          </div>
        ) : (
          <motion.div 
            className="space-y-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="h-[400px]">
              <Bar data={chartData} options={options} />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {sortedData.slice(0, 4).map((feature, index) => (
                <motion.div
                  key={feature.feature}
                  className={cn(
                    "p-4 rounded-lg transition-all duration-300",
                    "hover:shadow-md hover:-translate-y-1",
                    "bg-gradient-to-br from-muted/50 to-muted/30"
                  )}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <div className="text-sm font-medium mb-2 truncate">{feature.feature}</div>
                  <div className="text-2xl font-bold tracking-tight">{feature.driftCount}</div>
                  <div className="text-sm text-muted-foreground mt-1">
                    p-value: {feature.avgPValue.toFixed(3)}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </CardContent>
    </Card>
  )
} 