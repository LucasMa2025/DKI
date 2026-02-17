import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import config from '@/config'

export const useStatsAuthStore = defineStore('statsAuth', () => {
  const authenticated = ref(false)
  const lastAuthTime = ref<number | null>(null)
  
  // Session expires after 1 hour
  const SESSION_DURATION = 60 * 60 * 1000
  
  const isAuthenticated = computed(() => {
    if (!config.statsAuth.enabled) return true
    if (!authenticated.value) return false
    if (!lastAuthTime.value) return false
    
    // Check if session has expired
    const now = Date.now()
    if (now - lastAuthTime.value > SESSION_DURATION) {
      authenticated.value = false
      lastAuthTime.value = null
      return false
    }
    
    return true
  })
  
  function authenticate(password: string): boolean {
    if (!config.statsAuth.enabled) {
      authenticated.value = true
      lastAuthTime.value = Date.now()
      return true
    }
    
    if (password === config.statsAuth.password) {
      authenticated.value = true
      lastAuthTime.value = Date.now()
      return true
    }
    
    return false
  }
  
  function logout() {
    authenticated.value = false
    lastAuthTime.value = null
  }
  
  return {
    isAuthenticated,
    authenticate,
    logout,
  }
}, {
  persist: {
    paths: ['authenticated', 'lastAuthTime'],
  },
})
