import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import type { AppSettings, ModelSettings, DKISettings, APISettings } from '@/types'
import config from '@/config'

export const useSettingsStore = defineStore('settings', () => {
  // App settings
  const language = ref<'zh-CN' | 'en-US'>('zh-CN')
  const theme = ref<'light' | 'dark' | 'system'>('light')
  const fontSize = ref(14)
  const sendOnEnter = ref(true)
  const showTimestamps = ref(true)
  const compactMode = ref(false)
  
  // Model settings
  const defaultModel = ref(config.defaults.model)
  const temperature = ref(config.defaults.temperature)
  const maxTokens = ref(config.defaults.maxTokens)
  const topP = ref(1.0)
  
  // DKI settings
  const dkiEnabled = ref(true)
  const dkiDefaultAlpha = ref(config.defaults.dkiAlpha)
  const dkiUseHybrid = ref(true)
  const dkiDebugMode = ref(config.features.debugMode)
  
  // API settings
  const apiBaseUrl = ref(config.api.baseUrl)
  const apiKey = ref('')
  const apiTimeout = ref(config.api.timeout)
  
  // Apply system theme
  watch(theme, (newTheme) => {
    if (newTheme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      document.documentElement.classList.toggle('dark', prefersDark)
    } else {
      document.documentElement.classList.toggle('dark', newTheme === 'dark')
    }
  }, { immediate: true })
  
  // Getters
  function getAppSettings(): AppSettings {
    return {
      language: language.value,
      theme: theme.value,
      fontSize: fontSize.value,
      sendOnEnter: sendOnEnter.value,
      showTimestamps: showTimestamps.value,
      compactMode: compactMode.value,
    }
  }
  
  function getModelSettings(): ModelSettings {
    return {
      defaultModel: defaultModel.value,
      temperature: temperature.value,
      maxTokens: maxTokens.value,
      topP: topP.value,
    }
  }
  
  function getDKISettings(): DKISettings {
    return {
      enabled: dkiEnabled.value,
      defaultAlpha: dkiDefaultAlpha.value,
      useHybrid: dkiUseHybrid.value,
      debugMode: dkiDebugMode.value,
    }
  }
  
  function getAPISettings(): APISettings {
    return {
      baseUrl: apiBaseUrl.value,
      apiKey: apiKey.value,
      timeout: apiTimeout.value,
    }
  }
  
  // Setters
  function updateAppSettings(settings: Partial<AppSettings>) {
    if (settings.language !== undefined) language.value = settings.language
    if (settings.theme !== undefined) theme.value = settings.theme
    if (settings.fontSize !== undefined) fontSize.value = settings.fontSize
    if (settings.sendOnEnter !== undefined) sendOnEnter.value = settings.sendOnEnter
    if (settings.showTimestamps !== undefined) showTimestamps.value = settings.showTimestamps
    if (settings.compactMode !== undefined) compactMode.value = settings.compactMode
  }
  
  function updateModelSettings(settings: Partial<ModelSettings>) {
    if (settings.defaultModel !== undefined) defaultModel.value = settings.defaultModel
    if (settings.temperature !== undefined) temperature.value = settings.temperature
    if (settings.maxTokens !== undefined) maxTokens.value = settings.maxTokens
    if (settings.topP !== undefined) topP.value = settings.topP
  }
  
  function updateDKISettings(settings: Partial<DKISettings>) {
    if (settings.enabled !== undefined) dkiEnabled.value = settings.enabled
    if (settings.defaultAlpha !== undefined) dkiDefaultAlpha.value = settings.defaultAlpha
    if (settings.useHybrid !== undefined) dkiUseHybrid.value = settings.useHybrid
    if (settings.debugMode !== undefined) dkiDebugMode.value = settings.debugMode
  }
  
  function updateAPISettings(settings: Partial<APISettings>) {
    if (settings.baseUrl !== undefined) apiBaseUrl.value = settings.baseUrl
    if (settings.apiKey !== undefined) apiKey.value = settings.apiKey
    if (settings.timeout !== undefined) apiTimeout.value = settings.timeout
  }
  
  function resetToDefaults() {
    language.value = 'zh-CN'
    theme.value = 'light'
    fontSize.value = 14
    sendOnEnter.value = true
    showTimestamps.value = true
    compactMode.value = false
    
    defaultModel.value = config.defaults.model
    temperature.value = config.defaults.temperature
    maxTokens.value = config.defaults.maxTokens
    topP.value = 1.0
    
    dkiEnabled.value = true
    dkiDefaultAlpha.value = config.defaults.dkiAlpha
    dkiUseHybrid.value = true
    dkiDebugMode.value = config.features.debugMode
    
    apiBaseUrl.value = config.api.baseUrl
    apiKey.value = ''
    apiTimeout.value = config.api.timeout
  }
  
  return {
    // App settings
    language,
    theme,
    fontSize,
    sendOnEnter,
    showTimestamps,
    compactMode,
    
    // Model settings
    defaultModel,
    temperature,
    maxTokens,
    topP,
    
    // DKI settings
    dkiEnabled,
    dkiDefaultAlpha,
    dkiUseHybrid,
    dkiDebugMode,
    
    // API settings
    apiBaseUrl,
    apiKey,
    apiTimeout,
    
    // Getters
    getAppSettings,
    getModelSettings,
    getDKISettings,
    getAPISettings,
    
    // Setters
    updateAppSettings,
    updateModelSettings,
    updateDKISettings,
    updateAPISettings,
    resetToDefaults,
  }
}, {
  persist: true,
})
