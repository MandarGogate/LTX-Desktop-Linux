// Install the Electron API shim for web-mode operation.
// Must be imported before anything that uses window.electronAPI.
import './lib/electron-shim'
import { initServableUrls } from './lib/serve-url'

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

function renderApp() {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
}

// Pre-initialize the URL rewriter for web mode before the app reads persisted projects.
void initServableUrls().finally(renderApp)
