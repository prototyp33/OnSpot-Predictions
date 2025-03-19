"use client"

import { useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
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
  TimeScale,
  ChartData,
  ChartOptions,
} from "chart.js"
import { Line } from "react-chartjs-2"
import "chartjs-adapter-date-fns"
import { motion } from "framer-motion"
import { ZoomIn, ZoomOut, ChevronLeft, ChevronRight } from "lucide-react"

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
)

interface InteractiveTimeSeriesProps {
  data: {
    label: string
    data: number[]
    borderColor: string
  }[]
  labels: string[]
  title: string
  loading?: boolean
  className?: string
}

export function InteractiveTimeSeries({
  data,
  labels,
  title,
  loading = false,
  className,
}: InteractiveTimeSeriesProps) {
  const chartRef = useRef<ChartJS>(null)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [panOffset, setPanOffset] = useState(0)

  const chartData: ChartData<"line"> = {
    labels,
    datasets: data.map((dataset) => ({
      ...dataset,
      tension: 0.4,
      borderWidth: chartDefaults.borderWidth,
      pointRadius: 0,
      pointHoverRadius: 6,
      pointHoverBorderWidth: 2,
      pointHoverBackgroundColor: chartColors.background,
      pointHoverBorderColor: dataset.borderColor,
      backgroundColor: (context: any) => {
        const ctx = context.chart.ctx
        const gradient = ctx.createLinearGradient(0, 0, 0, 400)
        const color = dataset.borderColor.replace("hsl(var(--", "").replace("))", "")
        gradients[color as keyof typeof gradients].forEach((stop, index) => {
          gradient.addColorStop(index / 2, stop)
        })
        return gradient
      },
      fill: true,
    })),
  }

  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      ...chartDefaults.animation,
    },
    interaction: {
      mode: "nearest",
      axis: "x",
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
        position: "nearest",
      },
    },
    scales: {
      x: {
        type: "time",
        time: {
          unit: "day",
        },
        grid: {
          display: false,
        },
        ticks: {
          maxRotation: 0,
          font: chartDefaults.font,
        },
        min: labels[Math.floor(panOffset / zoomLevel)],
        max: labels[Math.floor((panOffset + labels.length) / zoomLevel) - 1],
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

  const handleZoomIn = () => {
    if (zoomLevel < 4) {
      setZoomLevel(zoomLevel + 1)
    }
  }

  const handleZoomOut = () => {
    if (zoomLevel > 1) {
      setZoomLevel(zoomLevel - 1)
    }
  }

  const handlePanLeft = () => {
    if (panOffset > 0) {
      setPanOffset(Math.max(0, panOffset - labels.length / 4))
    }
  }

  const handlePanRight = () => {
    if (panOffset < labels.length * (1 - 1/zoomLevel)) {
      setPanOffset(Math.min(labels.length * (1 - 1/zoomLevel), panOffset + labels.length / 4))
    }
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
          <CardTitle className="text-xl font-medium tracking-tight">{title}</CardTitle>
          <motion.div 
            className="flex items-center space-x-2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Button
              variant="outline"
              size="icon"
              onClick={handlePanLeft}
              disabled={panOffset <= 0}
              className="transition-all duration-200 hover:scale-105"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={handleZoomOut}
              disabled={zoomLevel <= 1}
              className="transition-all duration-200 hover:scale-105"
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={handleZoomIn}
              disabled={zoomLevel >= 4}
              className="transition-all duration-200 hover:scale-105"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={handlePanRight}
              disabled={panOffset >= labels.length * (1 - 1/zoomLevel)}
              className="transition-all duration-200 hover:scale-105"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
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
            className="h-[400px]"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Line ref={chartRef} data={chartData} options={options} />
          </motion.div>
        )}
      </CardContent>
    </Card>
  )
} 