import { useEffect, useState } from 'react'
import { backendFetch } from '../lib/backend'

interface GpuStats {
  gpu_name: string
  vram_used_mb: number
  vram_total_mb: number
  gpu_utilization: number
  temperature: number
}

export function GpuStatsWidget() {
  const [stats, setStats] = useState<GpuStats | null>(null)

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const res = await backendFetch('/api/gpu/stats')
        if (res.ok && !cancelled) {
          setStats(await res.json())
        }
      } catch {
        // ignore
      }
    }

    void poll()
    const id = setInterval(poll, 2000) // Poll every 2 seconds
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  if (!stats) return null

  const usedGb = (stats.vram_used_mb / 1024).toFixed(1)
  const totalGb = (stats.vram_total_mb / 1024).toFixed(0)
  const pct = Math.round((stats.vram_used_mb / stats.vram_total_mb) * 100)

  // Color based on usage
  const barColor =
    pct > 90 ? 'bg-red-500' : pct > 70 ? 'bg-yellow-500' : 'bg-green-500'
  const textColor =
    pct > 90 ? 'text-red-400' : pct > 70 ? 'text-yellow-400' : 'text-green-400'

  return (
    <div className="flex items-center gap-2 px-3 py-1 bg-zinc-800/60 rounded-lg border border-zinc-700/50 text-xs select-none">
      {/* GPU name */}
      <span className="text-zinc-400 hidden lg:inline" title={stats.gpu_name}>
        {stats.gpu_name.replace('NVIDIA ', '').replace('GeForce ', '')}
      </span>

      {/* VRAM bar */}
      <div className="flex items-center gap-1.5">
        <div className="w-16 h-1.5 bg-zinc-700 rounded-full overflow-hidden" title={`VRAM: ${usedGb}GB / ${totalGb}GB (${pct}%)`}>
          <div
            className={`h-full rounded-full transition-all duration-500 ${barColor}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className={`font-mono tabular-nums ${textColor}`}>
          {usedGb}/{totalGb}GB
        </span>
      </div>

      {/* GPU utilization */}
      {stats.gpu_utilization > 0 && (
        <span className="text-zinc-500" title={`GPU: ${stats.gpu_utilization}%`}>
          {stats.gpu_utilization}%
        </span>
      )}

      {/* Temperature */}
      {stats.temperature > 0 && (
        <span className={`${stats.temperature > 80 ? 'text-red-400' : 'text-zinc-500'}`} title={`${stats.temperature}°C`}>
          {stats.temperature}°C
        </span>
      )}
    </div>
  )
}
