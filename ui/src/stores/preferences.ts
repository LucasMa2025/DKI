import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { UserPreference } from '@/types'
import { api } from '@/services/api'
import { useAuthStore } from './auth'

export const usePreferencesStore = defineStore('preferences', () => {
  const preferences = ref<UserPreference[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  
  const activePreferences = computed(() => {
    return preferences.value.filter(p => p.isActive)
  })
  
  const preferencesByCategory = computed(() => {
    const grouped: Record<string, UserPreference[]> = {}
    for (const pref of preferences.value) {
      const category = pref.category || 'Uncategorized'
      if (!grouped[category]) {
        grouped[category] = []
      }
      grouped[category].push(pref)
    }
    return grouped
  })
  
  // Load preferences
  async function loadPreferences() {
    const authStore = useAuthStore()
    if (!authStore.user?.id) return
    
    try {
      loading.value = true
      error.value = null
      preferences.value = await api.preferences.list(authStore.user.id)
    } catch (e) {
      error.value = 'Failed to load preferences'
    } finally {
      loading.value = false
    }
  }
  
  // Create preference
  async function createPreference(preference: Omit<UserPreference, 'id' | 'userId' | 'createdAt' | 'updatedAt'>) {
    const authStore = useAuthStore()
    if (!authStore.user?.id) return null
    
    try {
      loading.value = true
      error.value = null
      
      const newPref = await api.preferences.create({
        ...preference,
        userId: authStore.user.id,
      })
      
      preferences.value.push(newPref)
      return newPref
    } catch (e) {
      error.value = 'Failed to create preference'
      return null
    } finally {
      loading.value = false
    }
  }
  
  // Update preference
  async function updatePreference(id: string, updates: Partial<UserPreference>) {
    try {
      loading.value = true
      error.value = null
      
      const updated = await api.preferences.update(id, updates)
      const index = preferences.value.findIndex(p => p.id === id)
      if (index !== -1) {
        preferences.value[index] = updated
      }
      return updated
    } catch (e) {
      error.value = 'Failed to update preference'
      return null
    } finally {
      loading.value = false
    }
  }
  
  // Delete preference
  async function deletePreference(id: string) {
    try {
      loading.value = true
      error.value = null
      
      await api.preferences.delete(id)
      preferences.value = preferences.value.filter(p => p.id !== id)
      return true
    } catch (e) {
      error.value = 'Failed to delete preference'
      return false
    } finally {
      loading.value = false
    }
  }
  
  // Toggle preference active state
  async function togglePreference(id: string) {
    const pref = preferences.value.find(p => p.id === id)
    if (!pref) return
    
    return updatePreference(id, { isActive: !pref.isActive })
  }
  
  // Clear error
  function clearError() {
    error.value = null
  }
  
  return {
    preferences,
    loading,
    error,
    activePreferences,
    preferencesByCategory,
    loadPreferences,
    createPreference,
    updatePreference,
    deletePreference,
    togglePreference,
    clearError,
  }
})
