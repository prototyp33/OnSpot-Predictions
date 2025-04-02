'use client';

import React, { useEffect } from "react";
import { SidebarProvider } from "@/components/ui/sidebar";
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/toaster";
import { DashboardSidebar } from "@/components/dashboard-sidebar";
import GlobalErrorHandler from "@/components/common/GlobalErrorHandler";
import SentryUserProvider from "@/components/common/SentryUserProvider";
import { apiCache } from '@/lib/cache';

// Cache cleanup interval (5 minutes)
const CACHE_CLEANUP_INTERVAL = 5 * 60 * 1000;

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Set up cache cleanup interval
  useEffect(() => {
    // Clear expired cache entries on mount
    apiCache.clearExpired();
    
    // Set up interval to clear expired cache entries
    const intervalId = setInterval(() => {
      apiCache.clearExpired();
    }, CACHE_CLEANUP_INTERVAL);
    
    // Clean up interval on unmount
    return () => {
      clearInterval(intervalId);
    };
  }, []);

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <SidebarProvider>
        <SentryUserProvider />
        <div className="relative flex min-h-screen">
          <DashboardSidebar />
          <main className="flex-1 overflow-x-hidden">{children}</main>
        </div>
        <Toaster />
        <GlobalErrorHandler />
      </SidebarProvider>
    </ThemeProvider>
  );
} 