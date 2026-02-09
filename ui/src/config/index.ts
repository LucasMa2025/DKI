// Application configuration
export const config = {
  // API settings
  api: {
    baseUrl: import.meta.env.VITE_API_BASE_URL || '/api',
    timeout: 30000,
  },
  
  // Stats page authentication
  // In production, this should be handled by proper backend authentication
  statsAuth: {
    enabled: true,
    // Simple password protection for stats page (non-production standard)
    password: import.meta.env.VITE_STATS_PASSWORD || 'dki_admin_2024',
  },
  
  // Default settings
  defaults: {
    model: 'dki-default',
    temperature: 0.7,
    maxTokens: 2048,
    dkiAlpha: 0.3,
  },
  
  // Feature flags
  features: {
    streaming: true,
    hybridInjection: true,
    debugMode: import.meta.env.DEV,
  },
}

export default config
