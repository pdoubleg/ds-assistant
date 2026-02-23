import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { CopilotKitProvider } from "@copilotkit/react-core/v2"
import { HttpAgent } from "@ag-ui/client"
import { ThemeProvider } from "@/lib/theme-context"
import './index.css'
import App from './App.tsx'

// Backend URL for AG-UI / CopilotKit connection
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001'

console.log('AG-UI Backend URL:', BACKEND_URL)

// Create HttpAgent pointing to our AG-UI endpoint
const auditAgent = new HttpAgent({
  url: BACKEND_URL,
  agentId: "audit_agent",
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <CopilotKitProvider
        agents__unsafe_dev_only={{
          audit_agent: auditAgent,
        }}
      >
        <App />
      </CopilotKitProvider>
    </ThemeProvider>
  </StrictMode>,
)
