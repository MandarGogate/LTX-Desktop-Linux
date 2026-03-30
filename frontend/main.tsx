// Install the Electron API shim for web-mode operation.
// Must be imported before anything that uses window.electronAPI.
import './lib/electron-shim'

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
