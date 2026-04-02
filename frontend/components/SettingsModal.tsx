import { AlertCircle, Check, Cpu, Download, Folder, Info, KeyRound, Settings, Sliders, Sparkles, Upload, X, Zap } from 'lucide-react'
import React, { useEffect, useRef, useState } from 'react'
import { Button } from './ui/button'
import { useAppSettings, type AppSettings } from '../contexts/AppSettingsContext'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'
import { ApiKeyHelperRow, LtxApiKeyInput, LtxApiKeyHelperRow } from './LtxApiKeyInput'

interface TextEncoderStatus {
  downloaded: boolean
  size_gb: number
  expected_size_gb: number
}

interface TextEncoderVariant {
  filename: string
  path: string
  size_mb: number
  format: string
  quant_level?: string | null
}

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  initialTab?: TabId
}

type TabId = 'general' | 'apiKeys' | 'inference' | 'promptEnhancer' | 'advanced' | 'about'

export function SettingsModal({ isOpen, onClose, initialTab }: SettingsModalProps) {
  const { settings, updateSettings, saveLtxApiKey, saveFalApiKey, saveGeminiApiKey } = useAppSettings()
  const isWebMode = !(window as any).__ELECTRON__
  const onSettingsChange = (next: AppSettings) => updateSettings(next)
  const [activeTab, setActiveTab] = useState<TabId>('general')
  const [ltxApiKeyInput, setLtxApiKeyInput] = useState('')
  const ltxApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [falApiKeyInput, setFalApiKeyInput] = useState('')
  const falApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [geminiApiKeyInput, setGeminiApiKeyInput] = useState('')
  const geminiApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [textEncoderStatus, setTextEncoderStatus] = useState<TextEncoderStatus | null>(null)
  const [textEncoderVariants, setTextEncoderVariants] = useState<TextEncoderVariant[]>([])
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [vramProfile, setVramProfile] = useState<any>(null)
  const [loraList, setLoraList] = useState<any[]>([])
  const [videoModelList, setVideoModelList] = useState<any[]>([])
  const [loraUploading, setLoraUploading] = useState(false)
  const [appVersion, setAppVersion] = useState('')
  const [externalModels, setExternalModels] = useState<any[]>([])
  const [downloadingExternal, setDownloadingExternal] = useState<string | null>(null)
  const [downloadExternalError, setDownloadExternalError] = useState<string | null>(null)
  const [downloadExternalSuccess, setDownloadExternalSuccess] = useState<string | null>(null)
  const [downloadExternalProgress, setDownloadExternalProgress] = useState(0)

  // Fetch VRAM profile and LoRA list when advanced tab opens
  useEffect(() => {
    if (!isOpen || activeTab !== 'advanced') return
    backendFetch('/api/gpu/vram-profile').then(r => r.json()).then(setVramProfile).catch(() => {})
    backendFetch('/api/gpu/loras').then(r => r.json()).then(d => setLoraList(d.loras || [])).catch(() => {})
    backendFetch('/api/gpu/video-models').then(r => r.json()).then(d => setVideoModelList(d.models || [])).catch(() => {})
    backendFetch('/api/gpu/text-encoders').then(r => r.json()).then(d => setTextEncoderVariants(d.variants || [])).catch(() => {})
    backendFetch('/api/gpu/external-models').then(r => r.json()).then(d => setExternalModels(d.models || [])).catch(() => {})
  }, [isOpen, activeTab])

  const handleUploadLora = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setLoraUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await backendFetch('/api/gpu/loras/upload', { method: 'POST', body: formData })
      if (res.ok) {
        const d = await (await backendFetch('/api/gpu/loras')).json()
        setLoraList(d.loras || [])
      }
    } catch (err) {
      logger.error(`LoRA upload failed: ${err}`)
    } finally {
      setLoraUploading(false)
      e.target.value = ''
    }
  }

  const handleDownloadExternalModel = async (model: any) => {
    setDownloadingExternal(model.id)
    setDownloadExternalError(null)
    setDownloadExternalSuccess(null)
    try {
      const targetSubdir = model.model_type === 'gguf' ? 'gguf' : model.model_type === 'lora' ? 'loras' : model.model_type === 'text_encoder' ? 'text_encoders' : ''
      const res = await backendFetch('/api/gpu/download-external-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_id: model.repo_id,
          filename: model.filename,
          target_subdir: targetSubdir,
        }),
      })
      const data = await res.json()
      if (data.status === 'already_downloaded') {
        setDownloadExternalSuccess(`${model.filename} already downloaded`)
      } else if (data.status === 'started') {
        // Poll for completion with progress
        const sessionId = data.session_id
        setDownloadExternalProgress(0)
        const poll = setInterval(async () => {
          try {
            const pr = await backendFetch(`/api/models/download/progress?sessionId=${sessionId}`)
            const pd = await pr.json()
            if (pd.total_progress) setDownloadExternalProgress(Math.min(99, Math.round(pd.total_progress)))
            if (pd.status === 'complete') {
              clearInterval(poll)
              setDownloadingExternal(null)
              setDownloadExternalProgress(100)
              setDownloadExternalSuccess(`${model.filename} downloaded successfully`)
              // Refresh external models list
    backendFetch('/api/gpu/video-models').then(r => r.json()).then(d => setVideoModelList(d.models || [])).catch(() => {})
    backendFetch('/api/gpu/text-encoders').then(r => r.json()).then(d => setTextEncoderVariants(d.variants || [])).catch(() => {})
    backendFetch('/api/gpu/external-models').then(r => r.json()).then(d => {
      setExternalModels(d.models || [])
    }).catch(() => {})
              backendFetch('/api/gpu/video-models').then(r => r.json()).then(d => setVideoModelList(d.models || [])).catch(() => {})
              backendFetch('/api/gpu/loras').then(r => r.json()).then(d => setLoraList(d.loras || [])).catch(() => {})
            } else if (pd.status === 'error') {
              clearInterval(poll)
              setDownloadingExternal(null)
              setDownloadExternalProgress(0)
              setDownloadExternalError(pd.error || 'Download failed')
            }
          } catch {
            // ignore poll errors
          }
        }, 2000)
        // Timeout after 60 minutes
        setTimeout(() => { clearInterval(poll); setDownloadingExternal(null) }, 60 * 60 * 1000)
      } else {
        setDownloadExternalError(data.message || 'Failed to start download')
        setDownloadingExternal(null)
      }
    } catch (err) {
      setDownloadExternalError(err instanceof Error ? err.message : 'Download failed')
      setDownloadingExternal(null)
    }
  }
  const [noticesText, setNoticesText] = useState<string | null>(null)
  const [noticesLoading, setNoticesLoading] = useState(false)
  const [showNotices, setShowNotices] = useState(false)
  const [modelLicenseText, setModelLicenseText] = useState<string | null>(null)
  const [modelLicenseLoading, setModelLicenseLoading] = useState(false)
  const [showModelLicense, setShowModelLicense] = useState(false)
  const [projectAssetsPath, setProjectAssetsPath] = useState('')

  // Sync active tab with initialTab prop when modal opens
  useEffect(() => {
    if (isOpen && initialTab) {
      setActiveTab(initialTab === 'apiKeys' || initialTab === 'promptEnhancer' ? 'general' : initialTab)
    }
  }, [isOpen, initialTab])

  // Fetch app version when About tab is shown
  useEffect(() => {
    if (activeTab !== 'about' || appVersion) return
    window.electronAPI.getAppInfo().then(info => setAppVersion(info.version)).catch(() => {})
  }, [activeTab, appVersion])

  useEffect(() => {
    if (!isOpen) return
    window.electronAPI.getProjectAssetsPath()
      .then((p: string) => setProjectAssetsPath(p))
      .catch(() => {})
  }, [isOpen])

  // Fetch text encoder status when modal opens
  useEffect(() => {
    if (!isOpen) return

    const fetchStatus = async () => {
      try {
        const response = await backendFetch('/api/models/status')
        if (response.ok) {
          const data = await response.json()
          setTextEncoderStatus(data.text_encoder_status)
        }
      } catch (e) {
        logger.error(`Failed to fetch text encoder status: ${e}`)
      }
    }

    fetchStatus()
    // Poll while downloading
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [isOpen, isDownloading])

  // Handle text encoder download
  const handleDownloadTextEncoder = async () => {
    setIsDownloading(true)
    setDownloadError(null)
    try {
      const response = await backendFetch('/api/text-encoder/download', { method: 'POST' })
      const data = await response.json()

      if (data.status === 'already_downloaded') {
        setTextEncoderStatus(prev => prev ? { ...prev, downloaded: true } : null)
      }
      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await backendFetch('/api/models/status')
          if (statusRes.ok) {
            const statusData = await statusRes.json()
            setTextEncoderStatus(statusData.text_encoder_status)
            if (statusData.text_encoder_status?.downloaded) {
              setIsDownloading(false)
              clearInterval(pollInterval)
            }
          }
        } catch {
          // ignore
        }
      }, 2000)

      // Timeout after 30 minutes
      setTimeout(() => {
        clearInterval(pollInterval)
        if (isDownloading) setIsDownloading(false)
      }, 30 * 60 * 1000)
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : 'Download failed')
      setIsDownloading(false)
    }
  }

  if (!isOpen) return null

  const handleToggleTorchCompile = () => {
    onSettingsChange({
      ...settings,
      useTorchCompile: !settings.useTorchCompile,
    })
  }

  const handleToggleLoadOnStartup = () => {
    onSettingsChange({
      ...settings,
      loadOnStartup: !settings.loadOnStartup,
    })
  }

  const handleFastUpscalerToggle = () => {
    onSettingsChange({
      ...settings,
      fastModel: { ...settings.fastModel, useUpscaler: !settings.fastModel?.useUpscaler },
    })
  }

  const handleProStepsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const steps = Math.max(1, Math.min(100, parseInt(e.target.value) || 20))
    onSettingsChange({
      ...settings,
      proModel: { ...settings.proModel, steps },
    })
  }

  const handleProUpscalerToggle = () => {
    onSettingsChange({
      ...settings,
      proModel: { ...settings.proModel, useUpscaler: !settings.proModel.useUpscaler },
    })
  }

  const handleCustomStepsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const steps = Math.max(1, Math.min(100, parseInt(e.target.value) || 20))
    onSettingsChange({
      ...settings,
      customModel: { ...settings.customModel, steps },
    })
  }

  const handleCustomUpscalerToggle = () => {
    onSettingsChange({
      ...settings,
      customModel: { ...settings.customModel, useUpscaler: !settings.customModel.useUpscaler },
    })
  }

  // Prompt Enhancer handlers
  const handleTogglePromptEnhancer = (mode: 't2v' | 'i2v') => {
    if (mode === 't2v') {
      onSettingsChange({ ...settings, promptEnhancerEnabledT2V: !settings.promptEnhancerEnabledT2V })
    } else {
      onSettingsChange({ ...settings, promptEnhancerEnabledI2V: !settings.promptEnhancerEnabledI2V })
    }
  }
  // Seed handlers
  const handleToggleSeedLock = () => {
    onSettingsChange({
      ...settings,
      seedLocked: !settings.seedLocked,
    })
  }

  const handleLockedSeedChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 0
    onSettingsChange({
      ...settings,
      lockedSeed: Math.max(0, Math.min(2147483647, value)),
    })
  }

  const handleRandomizeSeed = () => {
    onSettingsChange({
      ...settings,
      lockedSeed: Math.floor(Math.random() * 2147483647),
    })
  }

  const handleLoadModelLicense = async () => {
    setModelLicenseLoading(true)
    try {
      const text = await window.electronAPI.fetchLicenseText()
      setModelLicenseText(text)
      setShowModelLicense(true)
    } catch (e) {
      logger.error(`Failed to load model license: ${e}`)
    } finally {
      setModelLicenseLoading(false)
    }
  }

  const handleLoadNotices = async () => {
    setNoticesLoading(true)
    try {
      const text = await window.electronAPI.getNoticesText()
      setNoticesText(text)
      setShowNotices(true)
    } catch (e) {
      logger.error(`Failed to load notices: ${e}`)
    } finally {
      setNoticesLoading(false)
    }
  }

  const tabs = [
    { id: 'general' as TabId, label: 'General', icon: Settings },
    { id: 'inference' as TabId, label: 'Inference', icon: Sliders },
    { id: 'advanced' as TabId, label: 'GPU & Models', icon: Cpu },
    { id: 'about' as TabId, label: 'About', icon: Info },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-xl mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-zinc-400" />
            <h2 className="text-lg font-semibold text-white">Settings</h2>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-8 w-8 text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-zinc-800">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-white border-b-2 border-blue-500 -mb-px'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* Content */}
        <div className="px-6 py-5 space-y-6 h-[60vh] overflow-y-auto">
          {activeTab === 'general' && (
            <>
              {/* Project Assets Path */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Project Assets Path</h3>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  Where generated video and image assets are saved. Each project gets a subfolder.
                </p>
                <div className="flex gap-2">
                  <input
                    value={projectAssetsPath}
                    onChange={(e) => setProjectAssetsPath(e.target.value)}
                    placeholder={isWebMode ? 'Enter an absolute folder path on the server…' : 'Not set'}
                    className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {isWebMode ? (
                    <Button
                      variant="outline"
                      className="border-zinc-700 flex-shrink-0"
                      onClick={async () => {
                        const response = await backendFetch('/web/project-assets/path', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ path: projectAssetsPath }),
                        })
                        if (response.ok) {
                          const data = await response.json()
                          setProjectAssetsPath(data.path || projectAssetsPath)
                        }
                      }}
                    >
                      Save
                    </Button>
                  ) : (
                    <Button
                      variant="outline"
                      className="border-zinc-700 flex-shrink-0"
                      onClick={async () => {
                        const result = await window.electronAPI.openProjectAssetsPathChangeDialog()
                        if (result.success && result.path) {
                          setProjectAssetsPath(result.path)
                        }
                      }}
                    >
                      <Folder className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>

              {/* Text Encoder status — compact */}
              {!textEncoderStatus?.downloaded && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-amber-400" />
                    <h3 className="text-sm font-semibold text-white">Text Encoder Not Downloaded</h3>
                  </div>
                  <p className="text-xs text-zinc-500">Required for prompt encoding. ~{textEncoderStatus?.expected_size_gb || 8} GB download.</p>
                  {isDownloading ? (
                    <div className="flex items-center gap-2 text-xs text-blue-400">
                      <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                      <span>Downloading text encoder...</span>
                    </div>
                  ) : (
                    <Button size="sm" onClick={handleDownloadTextEncoder} className="w-full bg-blue-600 hover:bg-blue-500 text-white text-xs">
                      <Download className="h-3 w-3 mr-2" /> Download Text Encoder
                    </Button>
                  )}
                  {downloadError && <p className="text-xs text-red-400">{downloadError}</p>}
                </div>
              )}

              {/* Load on Startup Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Preload models on startup
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Load AI models in the background after the app starts. The video model is loaded
                      and warmed up on GPU, and the image model is preloaded into CPU RAM for faster
                      first generation. When disabled, models load on first use (faster startup, slower
                      first generation). Requires app restart to take effect.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleLoadOnStartup}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.loadOnStartup ? 'bg-blue-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.loadOnStartup ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.loadOnStartup
                    ? 'bg-blue-500/10 text-blue-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.loadOnStartup ? 'bg-blue-400' : 'bg-zinc-600'
                  }`} />
                  {settings.loadOnStartup ? 'Models preload in background at startup' : 'Models load on first generation'}
                </div>
              </div>

              {/* Torch Compile Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-orange-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Torch Compile
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Compiles the model for optimized inference. <span className="text-orange-400">Experimental:</span> First
                      generation can take 5-10+ minutes for compilation. Subsequent generations may be
                      20-40% faster. Requires app restart to take effect.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleTorchCompile}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.useTorchCompile ? 'bg-orange-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.useTorchCompile ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.useTorchCompile
                    ? 'bg-orange-500/10 text-orange-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.useTorchCompile ? 'bg-orange-400' : 'bg-zinc-600'
                  }`} />
                  {settings.useTorchCompile ? 'Optimized inference (recommended)' : 'Standard inference'}
                </div>
              </div>

              {/* Seed Lock Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Lock Seed
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Use the same seed for reproducible generations. When unlocked, a random seed is used each time.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleSeedLock}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.seedLocked ? 'bg-emerald-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.seedLocked ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Seed input - only show when locked */}
                {settings.seedLocked && (
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      value={settings.lockedSeed ?? 42}
                      onChange={handleLockedSeedChange}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      placeholder="Enter seed..."
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleRandomizeSeed}
                      className="h-9 px-3 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                      title="Generate random seed"
                    >
                      <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
                      </svg>
                    </Button>
                  </div>
                )}

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.seedLocked
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.seedLocked ? 'bg-emerald-400' : 'bg-zinc-600'
                  }`} />
                  {settings.seedLocked ? `Seed locked: ${settings.lockedSeed ?? 42}` : 'Random seed each generation'}
                </div>
              </div>

            </>
          )}

          {activeTab === 'apiKeys' && (
            <>
              {/* LTX API Key Section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">LTX API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your LTX API key is used for cloud text encoding, prompt enhancement, and Pro generation.
                  Add your key below to unlock these features.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={ltxApiKeyInputRef}
                      value={ltxApiKeyInput}
                      onChange={(e) => setLtxApiKeyInput(e.target.value)}
                      placeholder={settings.hasLtxApiKey ? 'Enter new key to replace...' : 'Enter your LTX API key...'}
                      stopPropagation
                      className="flex-1"
                    />
                    <button
                      onClick={() => {
                        const trimmed = ltxApiKeyInput.trim()
                        if (!trimmed) return
                        void saveLtxApiKey(trimmed)
                        setLtxApiKeyInput('')
                      }}
                      disabled={!ltxApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <LtxApiKeyHelperRow stopPropagation />
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasLtxApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-amber-500/10 text-amber-400'
                    }`}>
                      {settings.hasLtxApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          API key required
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* FAL API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <KeyRound className="h-4 w-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">FAL AI</h3>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">Optional</span>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your FAL AI key is used for generating images with Z Image Turbo when API generations are enabled.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={falApiKeyInputRef}
                      value={falApiKeyInput}
                      onChange={(e) => setFalApiKeyInput(e.target.value)}
                      placeholder={settings.hasFalApiKey ? 'Enter new key to replace...' : 'Enter your FAL AI API key...'}
                      stopPropagation
                      className="flex-1"
                    />
                    <button
                      onClick={() => {
                        const trimmed = falApiKeyInput.trim()
                        if (!trimmed) return
                        void saveFalApiKey(trimmed)
                        setFalApiKeyInput('')
                      }}
                      disabled={!falApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <ApiKeyHelperRow
                    stopPropagation
                    label="Get FAL API key"
                    onOpenKey={() => window.electronAPI.openFalApiKeyPage()}
                  />
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasFalApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-zinc-800 text-zinc-500'
                    }`}>
                      {settings.hasFalApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          Optional
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Gemini API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-white">Gemini API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your Gemini API key is used for AI-powered prompt suggestions when filling timeline gaps.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <input
                      ref={geminiApiKeyInputRef}
                      type="password"
                      value={geminiApiKeyInput}
                      onChange={(e) => setGeminiApiKeyInput(e.target.value)}
                      placeholder={settings.hasGeminiApiKey ? 'Enter new key to replace...' : 'Enter your Gemini API key...'}
                      onKeyDown={(e) => e.stopPropagation()}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      onClick={() => {
                        const trimmed = geminiApiKeyInput.trim()
                        if (!trimmed) return
                        void saveGeminiApiKey(trimmed)
                        setGeminiApiKeyInput('')
                      }}
                      disabled={!geminiApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasGeminiApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-amber-500/10 text-amber-400'
                    }`}>
                      {settings.hasGeminiApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          API key required
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <a
                      href="https://aistudio.google.com/app/apikey"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 transition-colors underline underline-offset-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Get Gemini API key →
                    </a>
                  </div>
                </div>
              </div>
            </>
          )}

          {activeTab === 'inference' && (
            <>
              {/* Fast Model Settings */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-green-400" />
                  <h3 className="text-sm font-semibold text-white">LTX 2.3 Fast / Balanced</h3>
                </div>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-4">
                  {/* Steps Info */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Inference Steps</label>
                      <p className="text-xs text-zinc-500">Fixed at 8 steps (built into distilled model)</p>
                    </div>
                    <span className="px-3 py-1.5 bg-zinc-700 rounded-lg text-sm text-zinc-400">8</span>
                  </div>

                  {/* Upscaler Toggle */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">2x Upscaler</label>
                      <p className="text-xs text-zinc-500">When off, generates at native resolution</p>
                    </div>
                    <button
                      onClick={handleFastUpscalerToggle}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        settings.fastModel?.useUpscaler !== false ? 'bg-green-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                          settings.fastModel?.useUpscaler !== false ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                </div>

                {/* Summary */}
                <div className="text-xs text-zinc-500">
                  Current: 8 steps. Used by LTX 2.3 Fast (distilled) and LTX 2.3 Balanced (dev + distilled LoRA). {settings.fastModel?.useUpscaler !== false ? 'Upscaler enabled.' : 'Native resolution.'}
                </div>
              </div>

              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <svg className="h-4 w-4 text-amber-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 7h18" />
                    <path d="M6 12h12" />
                    <path d="M10 17h4" />
                  </svg>
                  <h3 className="text-sm font-semibold text-white">LTX 2.3 Custom</h3>
                </div>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Inference Steps</label>
                      <p className="text-xs text-zinc-500">Used only by Custom mode with the selected models from Settings</p>
                    </div>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={settings.customModel?.steps ?? 20}
                      onChange={handleCustomStepsChange}
                      className="w-20 px-3 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-amber-500"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">2x Upscaler</label>
                      <p className="text-xs text-zinc-500">Saved with Custom mode settings for future pipeline support</p>
                    </div>
                    <button
                      onClick={handleCustomUpscalerToggle}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        settings.customModel?.useUpscaler !== false ? 'bg-amber-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                          settings.customModel?.useUpscaler !== false ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                </div>

                <div className="text-xs text-zinc-500">
                  Current: {settings.customModel?.steps ?? 20} steps. Custom mode uses the selected checkpoint/GGUF, LoRAs, and text encoder from Settings.
                </div>
              </div>

              {/* Info Box */}
              <div className="bg-zinc-800/30 rounded-lg p-3 mt-4">
                <p className="text-xs text-zinc-400">
                  <span className="text-blue-400 font-medium">Tip:</span> Lower steps = faster but lower quality.
                  Higher steps = better quality but slower.
                </p>
              </div>
            </>
          )}

          {activeTab === 'promptEnhancer' && (
            <>
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Prompt Enhancer</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Automatically enhances your prompts via the LTX API with rich visual details, sound descriptions,
                  and motion cues to help generate higher quality videos. Control independently for each generation type.
                </p>

                {!settings.hasLtxApiKey ? (
                  <div className="space-y-4 mt-2">
                    <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-4 space-y-3">
                      <div className="flex items-start gap-2.5">
                        <AlertCircle className="h-4 w-4 text-amber-400 mt-0.5 flex-shrink-0" />
                        <div className="space-y-2">
                          <p className="text-sm text-amber-300 font-medium">LTX API key required</p>
                          <p className="text-xs text-zinc-400 leading-relaxed">
                            Prompt enhancement runs server-side on the LTX API. To use this feature, you need to configure
                            an API key in the API Keys tab.
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => setActiveTab('apiKeys')}
                        className="w-full mt-1 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-lg transition-colors"
                      >
                        Set API Key
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* T2V Toggle */}
                    <div
                      className="flex items-center justify-between bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 cursor-pointer"
                      onClick={() => handleTogglePromptEnhancer('t2v')}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-blue-400 bg-blue-400/10 px-1.5 py-0.5 rounded">T2V</span>
                        <div>
                          <span className="text-sm text-zinc-200">Text-to-Video</span>
                          <p className="text-[10px] text-zinc-500 mt-0.5">
                            {settings.promptEnhancerEnabledT2V ? 'Prompts will be enhanced before T2V generation' : 'T2V prompts used as-is'}
                          </p>
                        </div>
                      </div>
                      <div className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                        settings.promptEnhancerEnabledT2V ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}>
                        <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                          settings.promptEnhancerEnabledT2V ? 'translate-x-5' : 'translate-x-0'
                        }`} />
                      </div>
                    </div>

                    {/* I2V Toggle */}
                    <div
                      className="flex items-center justify-between bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 cursor-pointer"
                      onClick={() => handleTogglePromptEnhancer('i2v')}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded">I2V</span>
                        <div>
                          <span className="text-sm text-zinc-200">Image-to-Video</span>
                          <p className="text-[10px] text-zinc-500 mt-0.5">
                            {settings.promptEnhancerEnabledI2V ? 'Prompts will be enhanced before I2V generation' : 'I2V prompts used as-is'}
                          </p>
                        </div>
                      </div>
                      <div className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                        settings.promptEnhancerEnabledI2V ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}>
                        <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                          settings.promptEnhancerEnabledI2V ? 'translate-x-5' : 'translate-x-0'
                        }`} />
                      </div>
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {activeTab === 'advanced' && (
            <>
              {/* GPU Info */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Your GPU</h3>
                </div>

                {vramProfile ? (
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-zinc-300">VRAM</span>
                      <span className="text-sm font-medium text-white">{vramProfile.vram_total_gb} GB</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-zinc-300">Tier</span>
                      <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                        vramProfile.tier === 'high' ? 'bg-green-500/20 text-green-400' :
                        vramProfile.tier === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>{vramProfile.tier.toUpperCase()}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-zinc-300">Max Resolution</span>
                      <span className="text-sm text-white">{vramProfile.max_resolution}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-zinc-300">SageAttention</span>
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        vramProfile.sage_attention_available ? 'bg-green-500/20 text-green-400' : 'bg-zinc-700 text-zinc-400'
                      }`}>{vramProfile.sage_attention_available ? 'Available ✓' : 'Not installed'}</span>
                    </div>
                    <div className="text-xs text-zinc-500 pt-2 border-t border-zinc-700">
                      Max frames: 1080p={vramProfile.max_frames_1080p} · 720p={vramProfile.max_frames_720p} · 540p={vramProfile.max_frames_540p}
                    </div>
                  </div>
                ) : (
                  <div className="bg-zinc-800/50 rounded-lg p-4 text-sm text-zinc-400">
                    Loading GPU info...
                  </div>
                )}
              </div>

              {/* Run Mode Selector */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Sliders className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Run Configuration</h3>
                </div>
                <p className="text-xs text-zinc-500">
                  Choose how aggressively the app offloads models. Auto selects the fastest config for your GPU. Lower VRAM modes work on weaker GPUs but are slower.
                </p>
                <select
                  value={settings.runMode || 'auto'}
                  onChange={(e) => onSettingsChange({ ...settings, runMode: e.target.value })}
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {vramProfile?.run_modes?.map((mode: any) => (
                    <option key={mode.value} value={mode.value}>
                      {mode.label} — {mode.description}
                    </option>
                  )) || (
                    <>
                      <option value="auto">Auto (detect from GPU)</option>
                      <option value="high_vram">High VRAM (≥24 GB)</option>
                      <option value="medium_vram">Medium VRAM (16-23 GB)</option>
                      <option value="low_vram">Low VRAM (12-15 GB)</option>
                      <option value="very_low_vram">Very Low VRAM (8-11 GB)</option>
                    </>
                  )}
                </select>
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.runMode === 'auto' ? 'bg-blue-500/10 text-blue-400' : 'bg-yellow-500/10 text-yellow-400'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.runMode === 'auto' ? 'bg-blue-400' : 'bg-yellow-400'
                  }`} />
                  {settings.runMode === 'auto'
                    ? `Auto-detected tier: ${vramProfile?.auto_tier?.toUpperCase() || '...'}`
                    : `Manual override: ${settings.runMode.replace('_', ' ')}`
                  }
                </div>
              </div>

              {/* Block Swap Controls */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-white">Transformer Block Swap</h3>
                </div>
                <p className="text-xs text-zinc-500">
                  Controls how many of the 48 transformer blocks stay on GPU during generation. More blocks on GPU = faster but uses more VRAM. Set to -1 for automatic.
                </p>
                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-zinc-300">Blocks on GPU</span>
                    <span className="text-sm font-medium text-white">
                      {settings.numBlocksToSwap < 0
                        ? `Auto (${vramProfile?.auto_blocks_on_gpu ?? '...'})`
                        : settings.numBlocksToSwap
                      }
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-1"
                    max="48"
                    step="1"
                    value={settings.numBlocksToSwap}
                    onChange={(e) => onSettingsChange({ ...settings, numBlocksToSwap: parseInt(e.target.value) })}
                    className="w-full accent-purple-500"
                  />
                  <div className="flex justify-between text-[10px] text-zinc-500">
                    <span>Auto (-1)</span>
                    <span>0 (min VRAM)</span>
                    <span>24 (balanced)</span>
                    <span>48 (max speed)</span>
                  </div>
                  {settings.numBlocksToSwap >= 0 && (
                    <button
                      onClick={() => onSettingsChange({ ...settings, numBlocksToSwap: -1 })}
                      className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2"
                    >
                      Reset to Auto
                    </button>
                  )}
                </div>
              </div>

              {/* Download Models from HuggingFace */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-green-400" />
                  <h3 className="text-sm font-semibold text-white">Download Models</h3>
                </div>
                <p className="text-xs text-zinc-500">
                  Download GGUF quantized models, distilled LoRA, and checkpoints directly from HuggingFace.
                </p>
                {downloadExternalSuccess && (
                  <div className="flex items-center gap-2 text-xs text-green-400 bg-green-500/10 px-3 py-2 rounded-lg">
                    <Check className="h-3 w-3" />
                    <span>{downloadExternalSuccess}</span>
                  </div>
                )}
                {downloadExternalError && (
                  <div className="flex items-center gap-2 text-xs text-red-400 bg-red-500/10 px-3 py-2 rounded-lg">
                    <AlertCircle className="h-3 w-3" />
                    <span>{downloadExternalError}</span>
                  </div>
                )}
                <div className="space-y-4">
                  {[
                    { title: 'LTX Video GGUF Models', items: externalModels.filter((m: any) => m.model_type === 'gguf' && !m.id?.startsWith('zit-gguf')) },
                    { title: 'LTX Checkpoints & LoRAs', items: externalModels.filter((m: any) => m.model_type === 'checkpoint' || m.model_type === 'lora') },
                    { title: 'Z-Image GGUF Models', items: externalModels.filter((m: any) => m.id?.startsWith('zit-gguf')) },
                    { title: 'Text Encoders', items: externalModels.filter((m: any) => m.model_type === 'text_encoder') },
                  ].map((group) => group.items.length > 0 && (
                    <div key={group.title} className="space-y-2">
                      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-400">{group.title}</h4>
                      {group.items.map((model: any) => {
                        const isDownloaded = model.description?.includes('[DOWNLOADED]')
                        const isDownloading = downloadingExternal === model.id
                        const cleanDesc = model.description?.replace(' [DOWNLOADED]', '') || ''
                        const displayName = typeof model.filename === 'string' ? model.filename.split('/').pop() : model.filename
                        return (
                          <div key={model.id} className={`bg-zinc-800/50 rounded-lg px-4 py-3 border ${
                            isDownloaded ? 'border-green-500/30' : 'border-zinc-700/50'
                          }`}>
                            <div className="flex items-center justify-between gap-2">
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-2">
                                  <span className="text-sm text-white truncate">{displayName}</span>
                                  {model.quant_level && (
                                    <span className="text-[10px] px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded flex-shrink-0">{model.quant_level}</span>
                                  )}
                                  <span className={`text-[10px] px-1.5 py-0.5 rounded flex-shrink-0 ${
                                    model.model_type === 'gguf' ? 'bg-purple-500/20 text-purple-400' :
                                    model.model_type === 'lora' ? 'bg-green-500/20 text-green-400' :
                                    model.model_type === 'text_encoder' ? 'bg-cyan-500/20 text-cyan-400' :
                                    'bg-zinc-700 text-zinc-400'
                                  }`}>{model.model_type}</span>
                                </div>
                                <p className="text-xs text-zinc-500 mt-0.5">{cleanDesc} · {model.size_gb} GB</p>
                              </div>
                              {isDownloaded ? (
                                <span className="text-xs text-green-400 flex items-center gap-1 flex-shrink-0">
                                  <Check className="h-3 w-3" /> Downloaded
                                </span>
                              ) : isDownloading ? (
                                <div className="flex flex-col gap-1 flex-shrink-0 min-w-[120px]">
                                  <span className="text-xs text-blue-400 flex items-center gap-1">
                                    <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                                    {downloadExternalProgress > 0 ? `${downloadExternalProgress}%` : 'Starting...'}
                                  </span>
                                  <div className="w-full h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                                    <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${downloadExternalProgress}%` }} />
                                  </div>
                                </div>
                              ) : (
                                <Button
                                  size="sm"
                                  onClick={() => handleDownloadExternalModel(model)}
                                  disabled={!!downloadingExternal}
                                  className="bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 flex-shrink-0"
                                >
                                  <Download className="h-3 w-3 mr-1" />
                                  Download
                                </Button>
                              )}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ))}
                </div>
                <div className="text-xs text-zinc-500 space-y-1">
                  <p>Sources: <a href="https://huggingface.co/unsloth/LTX-2.3-GGUF" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 underline">unsloth/LTX-2.3-GGUF</a> · <a href="https://huggingface.co/unsloth/Z-Image-Turbo-GGUF" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 underline">unsloth/Z-Image-Turbo-GGUF</a> · <a href="https://huggingface.co/Lightricks/LTX-2.3" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 underline">Lightricks/LTX-2.3</a> · <a href="https://huggingface.co/Comfy-Org/ltx-2/tree/main/split_files/text_encoders" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 underline">Comfy-Org text encoders</a></p>
                </div>
              </div>

              {/* Model Recommendations */}
              {vramProfile?.model_recommendations?.length > 0 && (
                <div className="space-y-3 pt-4 border-t border-zinc-800">
                  <h3 className="text-sm font-semibold text-white">Recommended Models for Your GPU</h3>
                  <div className="space-y-2">
                    {vramProfile.model_recommendations.map((rec: any, i: number) => (
                      <div key={i} className={`bg-zinc-800/50 rounded-lg px-4 py-3 border ${
                        rec.recommended === 'true' ? 'border-blue-500/50' : 'border-transparent'
                      }`}>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-white">{rec.model}</span>
                          {rec.recommended === 'true' && (
                            <span className="text-[10px] px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">Best for you</span>
                          )}
                        </div>
                        <p className="text-xs text-zinc-400 mt-0.5">{rec.desc}</p>
                      </div>
                    ))}
                  </div>
                  {vramProfile.gguf_recommended && (
                    <p className="text-xs text-zinc-500">
                      💡 Recommended quantization: <span className="text-blue-400 font-medium">{vramProfile.recommended_gguf_quant}</span> — place GGUF files in your models/gguf folder.
                    </p>
                  )}
                </div>
              )}

              {/* Default local model selection */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <h3 className="text-sm font-semibold text-white">Default Local Video Model</h3>
                <p className="text-xs text-zinc-500">
                  Pick the base model used for local generation. Automatic falls back to the app default.
                </p>
                <select
                  value={settings.preferredModelPath || ''}
                  onChange={(e) => onSettingsChange({ ...settings, preferredModelPath: e.target.value })}
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Automatic</option>
                  {videoModelList.map((model: any) => (
                    <option key={model.path} value={model.path}>
                      {model.filename}{model.quant_level ? ` (${model.quant_level})` : ''}
                    </option>
                  ))}
                </select>
                {videoModelList.length === 0 && (
                  <p className="text-xs text-amber-400">No selectable video models were found under your models directory.</p>
                )}
              </div>

              {/* Z-Image model selection */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <h3 className="text-sm font-semibold text-white">Z-Image Model</h3>
                <p className="text-xs text-zinc-500">
                  Choose which Z-Image-Turbo model to use for image generation. GGUF versions use less VRAM.
                </p>
                <select
                  value={settings.preferredZitModelPath || ''}
                  onChange={(e) => onSettingsChange({ ...settings, preferredZitModelPath: e.target.value })}
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Default (full model)</option>
                  {externalModels.filter((m: any) => m.id?.startsWith('zit-gguf')).map((model: any) => {
                    const isDownloaded = model.description?.includes('[DOWNLOADED]')
                    const displayName = model.filename.split('/').pop()
                    return (
                      <option key={model.id} value={`gguf/${displayName}`} disabled={!isDownloaded}>
                        {displayName}{model.quant_level ? ` (${model.quant_level})` : ''} — {model.size_gb} GB {!isDownloaded ? '(not downloaded)' : '✓'}
                      </option>
                    )
                  })}
                </select>
              </div>

              {/* Text Encoder selection */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <h3 className="text-sm font-semibold text-white">Text Encoder</h3>
                <p className="text-xs text-zinc-500">
                  Choose a pre-quantized text encoder for faster loading. FP8 is ~8 GB vs ~23 GB for BF16. Download from the Models section below.
                </p>
                <select
                  value={settings.preferredTextEncoderPath || ''}
                  onChange={(e) => onSettingsChange({ ...settings, preferredTextEncoderPath: e.target.value })}
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Default (bundled Gemma)</option>
                  {textEncoderVariants.map((variant) => {
                    const relativePath = `text_encoders/${variant.filename}`
                    return (
                      <option key={variant.path} value={relativePath}>
                        {variant.filename}
                        {variant.quant_level ? ` (${variant.quant_level})` : ''}
                        {` — ${(variant.size_mb / 1024).toFixed(1)} GB`}
                        {variant.format === 'gguf' ? ' [GGUF]' : variant.format === 'folder' ? ' [Folder]' : ''}
                      </option>
                    )
                  })}
                </select>
              </div>
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <h3 className="text-sm font-semibold text-white">Generation LoRAs</h3>
                <p className="text-xs text-zinc-500">
                  Enable one or more LoRAs and set their strengths. All enabled LoRAs are applied during local generation.
                </p>
                {loraList.filter((l: any) => !l.is_ic_lora).length > 0 ? (
                  <div className="space-y-2">
                    {(() => {
                      const selectedModelPath = (settings.preferredModelPath || '').toLowerCase()
                      const baseModelIsDistilled = selectedModelPath.includes('distilled')
                      return loraList.filter((l: any) => !l.is_ic_lora).map((lora: any) => {
                        const isDistilledLoraOnDistilledModel = lora.is_distilled && baseModelIsDistilled
                        const selected = settings.selectedLoras.find((entry: any) => entry.path === lora.path)
                        const enabled = !!selected && !isDistilledLoraOnDistilledModel
                        const strength = selected?.strength ?? lora.suggested_strength ?? 0.8
                        return (
                          <div key={lora.path} className={`bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 ${isDistilledLoraOnDistilledModel ? 'opacity-50' : ''}`}>
                            <label className="flex items-center gap-3 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={enabled}
                                disabled={isDistilledLoraOnDistilledModel}
                                onChange={(e) => {
                                  const next = settings.selectedLoras.filter((entry: any) => entry.path !== lora.path)
                                  if (e.target.checked) next.push({ path: lora.path, strength })
                                  onSettingsChange({ ...settings, selectedLoras: next })
                                }}
                                className="accent-blue-500"
                              />
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-2 min-w-0">
                                  <span className="text-sm text-white truncate">{lora.filename}</span>
                                  {lora.is_distilled && <span className="text-[10px] px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded flex-shrink-0">Distilled</span>}
                                </div>
                                {isDistilledLoraOnDistilledModel ? (
                                  <p className="text-xs text-amber-400 mt-1">Skipped — your selected model is already distilled</p>
                                ) : (
                                  <p className="text-xs text-zinc-500 mt-1">Suggested strength: {lora.suggested_strength}</p>
                                )}
                              </div>
                            </label>
                            {enabled && !isDistilledLoraOnDistilledModel && (
                              <div className="mt-3 space-y-1">
                                <div className="flex items-center justify-between text-xs text-zinc-400">
                                  <span>Strength</span>
                                  <span>{strength.toFixed(2)}</span>
                                </div>
                                <input
                                  type="range"
                                  min="0"
                                  max="2"
                                  step="0.05"
                                  value={strength}
                                  onChange={(e) => {
                                    const next = settings.selectedLoras.map((entry: any) => entry.path === lora.path ? { ...entry, strength: Number(e.target.value) } : entry)
                                    onSettingsChange({ ...settings, selectedLoras: next })
                                  }}
                                  className="w-full accent-blue-500"
                                />
                              </div>
                            )}
                          </div>
                        )
                      })
                    })()}
                  </div>
                ) : (
                  <p className="text-xs text-zinc-500">No LoRAs found. Put LoRA safetensors in <code>models/loras</code> or upload one below.</p>
                )}
              </div>

              {/* LoRA Management */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white">LoRA Models</h3>
                  <label className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg cursor-pointer transition-colors ${
                    loraUploading ? 'bg-zinc-700 text-zinc-400' : 'bg-blue-600 hover:bg-blue-500 text-white'
                  }`}>
                    <Upload className="h-3 w-3" />
                    {loraUploading ? 'Uploading...' : 'Upload LoRA'}
                    <input type="file" accept=".safetensors" className="hidden" onChange={handleUploadLora} disabled={loraUploading} />
                  </label>
                </div>

                <p className="text-xs text-zinc-500">
                  Add custom LoRA files (.safetensors) to enhance generation style. Distilled LoRA uses 8 fast steps even with the dev base model.
                </p>

                {loraList.length > 0 ? (
                  <div className="space-y-2">
                    {loraList.map((lora: any, i: number) => (
                      <div key={i} className="bg-zinc-800/50 rounded-lg px-4 py-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="text-sm text-white truncate">{lora.filename}</span>
                            {lora.is_distilled && <span className="text-[10px] px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded flex-shrink-0">Distilled</span>}
                            {lora.is_ic_lora && <span className="text-[10px] px-1.5 py-0.5 bg-purple-500/20 text-purple-400 rounded flex-shrink-0">IC-LoRA</span>}
                          </div>
                          <span className="text-xs text-zinc-500 flex-shrink-0 ml-2">{lora.size_mb} MB</span>
                        </div>
                        <p className="text-xs text-zinc-500 mt-1">
                          Suggested strength: {lora.suggested_strength} · {lora.is_distilled ? '8-step distilled schedule' : lora.is_ic_lora ? 'Image conditioning' : 'Custom style'}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="bg-zinc-800/30 rounded-lg p-4 text-center">
                    <p className="text-sm text-zinc-400">No LoRA files found</p>
                    <p className="text-xs text-zinc-500 mt-1">Upload a .safetensors LoRA or place files in your models/loras folder</p>
                  </div>
                )}
              </div>

              {/* Tips */}
              <div className="bg-zinc-800/30 rounded-lg p-3 mt-4">
                <p className="text-xs text-zinc-400">
                  <span className="text-blue-400 font-medium">Quick guide:</span> Your GPU uses block swap + FP8 to fit
                  the 22B model in VRAM. SageAttention speeds up the attention layers. Distilled LoRA lets you use the
                  high-quality dev base model with only 8 inference steps.
                </p>
              </div>
            </>
          )}

          {activeTab === 'about' && (
            <>
              {showModelLicense ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">LTX-2 Model License</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowModelLicense(false)}
                      className="h-7 px-2 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                    >
                      Back
                    </Button>
                  </div>
                  <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-800/50 rounded-lg p-4 max-h-[50vh] overflow-y-auto border border-zinc-700/50">
                    {modelLicenseText}
                  </pre>
                </div>
              ) : showNotices ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">Third-Party Notices</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowNotices(false)}
                      className="h-7 px-2 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                    >
                      Back
                    </Button>
                  </div>
                  <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-800/50 rounded-lg p-4 max-h-[50vh] overflow-y-auto border border-zinc-700/50">
                    {noticesText}
                  </pre>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* App Identity */}
                  <div className="text-center space-y-2">
                    <h3 className="text-lg font-bold text-white">LTX Desktop</h3>
                    <p className="text-sm text-zinc-400">Version {appVersion || '...'}</p>
                    <p className="text-xs text-zinc-500">AI-Powered Video Editor</p>
                  </div>

                  {/* License */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <Info className="h-4 w-4 text-blue-400" />
                      <span className="text-sm font-medium text-white">License</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      Licensed under the Apache License, Version 2.0
                    </p>
                  </div>

                  {/* LTX-2 Model License */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                      </svg>
                      <span className="text-sm font-medium text-white">LTX-2 Model License</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      The LTX-2 model is subject to the LTX-2 Community License Agreement, accepted during first-run setup.
                    </p>
                    <Button
                      size="sm"
                      onClick={handleLoadModelLicense}
                      disabled={modelLicenseLoading}
                      className="w-full bg-zinc-700 hover:bg-zinc-600 text-white text-xs"
                    >
                      {modelLicenseLoading ? 'Loading...' : 'View Model License'}
                    </Button>
                  </div>

                  {/* Third-Party Notices */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                      </svg>
                      <span className="text-sm font-medium text-white">Third-Party Notices</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      This application uses open-source software and AI models subject to their own license terms.
                    </p>
                    <Button
                      size="sm"
                      onClick={handleLoadNotices}
                      disabled={noticesLoading}
                      className="w-full bg-zinc-700 hover:bg-zinc-600 text-white text-xs"
                    >
                      {noticesLoading ? 'Loading...' : 'View Third-Party Notices'}
                    </Button>
                  </div>

                  {/* Copyright */}
                  <p className="text-center text-xs text-zinc-600">
                    Copyright © 2026 Lightricks
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-zinc-800 flex justify-end">
          <Button
            onClick={onClose}
            className="bg-zinc-700 hover:bg-zinc-600 text-white"
          >
            Done
          </Button>
        </div>
      </div>
    </div>
  )
}

export type { AppSettings, TabId as SettingsTabId }
