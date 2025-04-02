"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import { chartColors, chartDefaults } from "./styles"
import { motion } from "framer-motion"

interface GaugeChartProps {
  value: number
  min?: number
  max?: number
  title: string
  loading?: boolean
  className?: string
}

export function GaugeChart({
  value,
  min = 0,
  max = 100,
  title,
  loading = false,
  className,
}: GaugeChartProps) {
  // Calculate percentage for the gauge
  const percentage = ((value - min) / (max - min)) * 100
  
  // Calculate color based on percentage
  const getColor = (percent: number) => {
    if (percent <= 33) return chartColors.success
    if (percent <= 66) return chartColors.warning
    return chartColors.destructive
  }

  // Calculate rotation for the gauge needle
  const rotation = (percentage / 100) * 180 - 90

  return (
    <Card className={cn(
      "overflow-hidden border-0 transition-all duration-300",
      "hover:shadow-lg hover:-translate-y-1",
      "bg-gradient-to-br from-background to-muted/10",
      className
    )}>
      <CardHeader className="px-6 pt-6 pb-0">
        <CardTitle className="text-xl font-medium tracking-tight">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        {loading ? (
          <div className="space-y-4">
            <Skeleton className="h-[200px] w-full rounded-lg" />
          </div>
        ) : (
          <motion.div 
            className="relative w-full aspect-square"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-full h-full max-w-[200px] max-h-[200px]">
                <svg viewBox="0 0 100 50" className="w-full drop-shadow-lg">
                  {/* Background arc with gradient */}
                  <defs>
                    <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor={chartColors.mutedForeground} stopOpacity="0.1" />
                      <stop offset="100%" stopColor={chartColors.mutedForeground} stopOpacity="0.3" />
                    </linearGradient>
                  </defs>
                  <path
                    d="M 10 45 A 40 40 0 1 1 90 45"
                    fill="none"
                    stroke="url(#gaugeGradient)"
                    strokeWidth="8"
                    strokeLinecap="round"
                  />
                  {/* Value arc with animation */}
                  <motion.path
                    d="M 10 45 A 40 40 0 1 1 90 45"
                    fill="none"
                    stroke={getColor(percentage)}
                    strokeWidth="8"
                    strokeLinecap="round"
                    initial={{ strokeDasharray: "0, 280" }}
                    animate={{ strokeDasharray: `${percentage * 2.8}, 280` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                  />
                  {/* Animated needle */}
                  <motion.g
                    initial={{ rotate: -90 }}
                    animate={{ rotate: rotation }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    style={{ transformOrigin: "50px 45px" }}
                  >
                    <line
                      x1="50"
                      y1="45"
                      x2="50"
                      y2="10"
                      stroke={chartColors.foreground}
                      strokeWidth="2"
                      strokeLinecap="round"
                      className="drop-shadow"
                    />
                    <circle
                      cx="50"
                      cy="45"
                      r="4"
                      fill={chartColors.background}
                      stroke={chartColors.foreground}
                      strokeWidth="2"
                      className="drop-shadow"
                    />
                  </motion.g>
                </svg>
              </div>
            </div>
            {/* Animated value display */}
            <motion.div 
              className="absolute inset-0 flex items-center justify-center"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <div className="text-center">
                <span className="text-4xl font-bold tracking-tight">{value.toFixed(1)}</span>
                <span className="text-sm text-muted-foreground ml-2">/ {max}</span>
              </div>
            </motion.div>
          </motion.div>
        )}
      </CardContent>
    </Card>
  )
} 