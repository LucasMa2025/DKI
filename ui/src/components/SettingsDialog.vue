<template>
  <el-dialog
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    title="Settings"
    width="700px"
    :close-on-click-modal="false"
    class="settings-dialog"
  >
    <el-tabs v-model="activeTab" tab-position="left">
      <!-- General Settings -->
      <el-tab-pane label="General" name="general">
        <div class="settings-section">
          <h3>Interface Settings</h3>
          
          <el-form label-position="left" label-width="140px">
            <el-form-item label="Language">
              <el-select v-model="settings.language" style="width: 200px">
                <el-option label="简体中文" value="zh-CN" />
                <el-option label="English" value="en-US" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="Theme">
              <el-radio-group v-model="settings.theme">
                <el-radio-button value="light">
                  <el-icon><Sunny /></el-icon> Light
                </el-radio-button>
                <el-radio-button value="dark">
                  <el-icon><Moon /></el-icon> Dark
                </el-radio-button>
                <el-radio-button value="system">
                  <el-icon><Monitor /></el-icon> System
                </el-radio-button>
              </el-radio-group>
            </el-form-item>
            
            <el-form-item label="Font Size">
              <el-slider
                v-model="settings.fontSize"
                :min="12"
                :max="20"
                :step="1"
                :marks="{ 12: 'Small', 14: 'Default', 18: 'Large' }"
                style="width: 200px"
              />
            </el-form-item>
          </el-form>
        </div>
        
        <el-divider />
        
        <div class="settings-section">
          <h3>Chat Settings</h3>
          
          <el-form label-position="left" label-width="140px">
            <el-form-item label="Send on Enter">
              <el-switch v-model="settings.sendOnEnter" />
              <span class="setting-hint">Press Enter to send, Shift+Enter for new line</span>
            </el-form-item>
            
            <el-form-item label="Show Timestamps">
              <el-switch v-model="settings.showTimestamps" />
            </el-form-item>
            
            <el-form-item label="Compact Mode">
              <el-switch v-model="settings.compactMode" />
              <span class="setting-hint">Reduce message spacing</span>
            </el-form-item>
          </el-form>
        </div>
      </el-tab-pane>
      
      <!-- Model Settings -->
      <el-tab-pane label="Model" name="model">
        <div class="settings-section">
          <h3>Model Configuration</h3>
          
          <el-form label-position="left" label-width="140px">
            <el-form-item label="Default Model">
              <el-select v-model="settings.defaultModel" style="width: 200px">
                <el-option label="DKI Default" value="dki-default" />
                <el-option label="GPT-4" value="gpt-4" />
                <el-option label="GPT-3.5 Turbo" value="gpt-3.5-turbo" />
                <el-option label="Claude 3" value="claude-3" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="Temperature">
              <el-slider
                v-model="settings.temperature"
                :min="0"
                :max="2"
                :step="0.1"
                :marks="{ 0: 'Precise', 0.7: 'Balanced', 2: 'Creative' }"
                style="width: 300px"
              />
              <span class="setting-value">{{ settings.temperature }}</span>
            </el-form-item>
            
            <el-form-item label="Max Tokens">
              <el-input-number
                v-model="settings.maxTokens"
                :min="256"
                :max="8192"
                :step="256"
              />
            </el-form-item>
            
            <el-form-item label="Top P">
              <el-slider
                v-model="settings.topP"
                :min="0"
                :max="1"
                :step="0.05"
                style="width: 300px"
              />
              <span class="setting-value">{{ settings.topP }}</span>
            </el-form-item>
          </el-form>
        </div>
      </el-tab-pane>
      
      <!-- DKI Settings -->
      <el-tab-pane label="DKI" name="dki">
        <div class="settings-section">
          <h3>DKI Injection Settings</h3>
          
          <el-alert
            type="info"
            :closable="false"
            show-icon
            style="margin-bottom: 20px"
          >
            DKI (Dynamic KV Injection) enhances model response personalization by injecting user preferences and conversation history.
          </el-alert>
          
          <el-form label-position="left" label-width="160px">
            <el-form-item label="Enable DKI">
              <el-switch v-model="settings.dkiEnabled" />
              <span class="setting-hint">Enable dynamic K/V injection</span>
            </el-form-item>
            
            <el-form-item label="Default Alpha">
              <el-slider
                v-model="settings.dkiDefaultAlpha"
                :min="0"
                :max="1"
                :step="0.05"
                :marks="{ 0: 'None', 0.3: 'Recommended', 1: 'Max' }"
                style="width: 300px"
                :disabled="!settings.dkiEnabled"
              />
              <span class="setting-value">{{ settings.dkiDefaultAlpha }}</span>
            </el-form-item>
            
            <el-form-item label="Hybrid Injection">
              <el-switch
                v-model="settings.dkiUseHybrid"
                :disabled="!settings.dkiEnabled"
              />
              <span class="setting-hint">Use preference K/V injection + history suffix prompting</span>
            </el-form-item>
            
            <el-form-item label="Debug Mode">
              <el-switch
                v-model="settings.dkiDebugMode"
                :disabled="!settings.dkiEnabled"
              />
              <span class="setting-hint">Show DKI injection details and metadata</span>
            </el-form-item>
          </el-form>
        </div>
        
        <el-divider />
        
        <div class="settings-section">
          <h3>Injection Strategy Details</h3>
          
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="Preference Injection">
              K/V injection (negative positions), 50-200 tokens, cacheable
            </el-descriptions-item>
            <el-descriptions-item label="History Injection">
              Suffix prompting (positive positions), 100-4000 tokens, dynamic
            </el-descriptions-item>
            <el-descriptions-item label="Alpha Value">
              Controls injection strength, 0 = no effect, 1 = maximum effect
            </el-descriptions-item>
          </el-descriptions>
        </div>
      </el-tab-pane>
      
      <!-- API Settings -->
      <el-tab-pane label="API" name="api">
        <div class="settings-section">
          <h3>API Configuration</h3>
          
          <el-form label-position="left" label-width="140px">
            <el-form-item label="API URL">
              <el-input
                v-model="settings.apiBaseUrl"
                placeholder="http://localhost:8080/api"
                style="width: 300px"
              />
            </el-form-item>
            
            <el-form-item label="API Key">
              <el-input
                v-model="settings.apiKey"
                type="password"
                placeholder="Optional"
                show-password
                style="width: 300px"
              />
            </el-form-item>
            
            <el-form-item label="Timeout">
              <el-input-number
                v-model="settings.apiTimeout"
                :min="5000"
                :max="120000"
                :step="1000"
              />
              <span class="setting-hint">milliseconds</span>
            </el-form-item>
          </el-form>
        </div>
        
        <el-divider />
        
        <div class="settings-section">
          <h3>Connection Test</h3>
          
          <el-button
            type="primary"
            :loading="testing"
            @click="testConnection"
          >
            Test Connection
          </el-button>
          
          <el-tag
            v-if="connectionStatus"
            :type="connectionStatus === 'success' ? 'success' : 'danger'"
            style="margin-left: 12px"
          >
            {{ connectionStatus === 'success' ? 'Connected' : 'Connection Failed' }}
          </el-tag>
        </div>
      </el-tab-pane>
      
      <!-- About -->
      <el-tab-pane label="About" name="about">
        <div class="settings-section about-section">
          <div class="about-logo">
            <img src="/logo.svg" alt="DKI" />
          </div>
          
          <h2>DKI Chat</h2>
          <p class="version">Version 1.0.0</p>
          
          <p class="description">
            Dynamic KV Injection (DKI) is an attention-level memory augmentation technique
            that provides personalized AI conversation experiences by dynamically injecting user preferences and conversation history.
          </p>
          
          <el-divider />
          
          <div class="about-links">
            <el-link type="primary" :underline="false">
              <el-icon><Document /></el-icon> Documentation
            </el-link>
            <el-link type="primary" :underline="false">
              <el-icon><Link /></el-icon> GitHub
            </el-link>
            <el-link type="primary" :underline="false">
              <el-icon><ChatDotRound /></el-icon> Feedback
            </el-link>
          </div>
          
          <el-divider />
          
          <p class="copyright">
            © 2024 DKI Project. All rights reserved.
          </p>
        </div>
      </el-tab-pane>
    </el-tabs>
    
    <template #footer>
      <el-button @click="handleReset">Reset Defaults</el-button>
      <el-button @click="$emit('update:modelValue', false)">Cancel</el-button>
      <el-button type="primary" @click="handleSave">Save</el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, reactive, watch } from 'vue'
import {
  Sunny,
  Moon,
  Monitor,
  Document,
  Link,
  ChatDotRound,
} from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useSettingsStore } from '@/stores/settings'
import { api } from '@/services/api'

defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
}>()

const settingsStore = useSettingsStore()

const activeTab = ref('general')
const testing = ref(false)
const connectionStatus = ref<'success' | 'error' | null>(null)

const settings = reactive({
  // General
  language: settingsStore.language,
  theme: settingsStore.theme,
  fontSize: settingsStore.fontSize,
  sendOnEnter: settingsStore.sendOnEnter,
  showTimestamps: settingsStore.showTimestamps,
  compactMode: settingsStore.compactMode,
  
  // Model
  defaultModel: settingsStore.defaultModel,
  temperature: settingsStore.temperature,
  maxTokens: settingsStore.maxTokens,
  topP: settingsStore.topP,
  
  // DKI
  dkiEnabled: settingsStore.dkiEnabled,
  dkiDefaultAlpha: settingsStore.dkiDefaultAlpha,
  dkiUseHybrid: settingsStore.dkiUseHybrid,
  dkiDebugMode: settingsStore.dkiDebugMode,
  
  // API
  apiBaseUrl: settingsStore.apiBaseUrl,
  apiKey: settingsStore.apiKey,
  apiTimeout: settingsStore.apiTimeout,
})

// Sync settings when dialog opens
watch(() => settingsStore.$state, () => {
  settings.language = settingsStore.language
  settings.theme = settingsStore.theme
  settings.fontSize = settingsStore.fontSize
  settings.sendOnEnter = settingsStore.sendOnEnter
  settings.showTimestamps = settingsStore.showTimestamps
  settings.compactMode = settingsStore.compactMode
  settings.defaultModel = settingsStore.defaultModel
  settings.temperature = settingsStore.temperature
  settings.maxTokens = settingsStore.maxTokens
  settings.topP = settingsStore.topP
  settings.dkiEnabled = settingsStore.dkiEnabled
  settings.dkiDefaultAlpha = settingsStore.dkiDefaultAlpha
  settings.dkiUseHybrid = settingsStore.dkiUseHybrid
  settings.dkiDebugMode = settingsStore.dkiDebugMode
  settings.apiBaseUrl = settingsStore.apiBaseUrl
  settings.apiKey = settingsStore.apiKey
  settings.apiTimeout = settingsStore.apiTimeout
}, { deep: true })

async function testConnection() {
  testing.value = true
  connectionStatus.value = null
  
  try {
    await api.health.check()
    connectionStatus.value = 'success'
    ElMessage.success('Connection successful')
  } catch {
    connectionStatus.value = 'error'
    ElMessage.error('Connection failed')
  } finally {
    testing.value = false
  }
}

async function handleReset() {
  await ElMessageBox.confirm('Are you sure you want to reset all settings to defaults?', 'Reset Defaults', {
    type: 'warning',
    confirmButtonText: 'OK',
    cancelButtonText: 'Cancel',
  })
  
  settingsStore.resetToDefaults()
  ElMessage.success('Settings restored to defaults')
}

function handleSave() {
  settingsStore.updateAppSettings({
    language: settings.language,
    theme: settings.theme,
    fontSize: settings.fontSize,
    sendOnEnter: settings.sendOnEnter,
    showTimestamps: settings.showTimestamps,
    compactMode: settings.compactMode,
  })
  
  settingsStore.updateModelSettings({
    defaultModel: settings.defaultModel,
    temperature: settings.temperature,
    maxTokens: settings.maxTokens,
    topP: settings.topP,
  })
  
  settingsStore.updateDKISettings({
    enabled: settings.dkiEnabled,
    defaultAlpha: settings.dkiDefaultAlpha,
    useHybrid: settings.dkiUseHybrid,
    debugMode: settings.dkiDebugMode,
  })
  
  settingsStore.updateAPISettings({
    baseUrl: settings.apiBaseUrl,
    apiKey: settings.apiKey,
    timeout: settings.apiTimeout,
  })
  
  ElMessage.success('Settings saved')
  emit('update:modelValue', false)
}
</script>

<style lang="scss" scoped>
.settings-dialog {
  :deep(.el-dialog__body) {
    padding: 0;
  }
  
  :deep(.el-tabs) {
    height: 500px;
    
    .el-tabs__header {
      width: 120px;
      margin-right: 0;
    }
    
    .el-tabs__content {
      padding: 24px;
      height: 100%;
      overflow-y: auto;
    }
  }
}

.settings-section {
  h3 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 20px;
  }
}

.setting-hint {
  font-size: 12px;
  color: var(--text-muted);
  margin-left: 12px;
}

.setting-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--primary-color);
  margin-left: 12px;
  min-width: 40px;
  display: inline-block;
}

.about-section {
  text-align: center;
  
  .about-logo {
    img {
      width: 80px;
      height: 80px;
    }
  }
  
  h2 {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 16px 0 8px;
  }
  
  .version {
    font-size: 14px;
    color: var(--text-muted);
    margin: 0 0 16px;
  }
  
  .description {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 400px;
    margin: 0 auto;
  }
  
  .about-links {
    display: flex;
    justify-content: center;
    gap: 24px;
    
    .el-link {
      display: flex;
      align-items: center;
      gap: 4px;
    }
  }
  
  .copyright {
    font-size: 12px;
    color: var(--text-muted);
    margin: 0;
  }
}

:deep(.el-form-item) {
  margin-bottom: 24px;
  
  .el-form-item__content {
    align-items: center;
  }
}

:deep(.el-slider) {
  .el-slider__marks-text {
    font-size: 11px;
  }
}
</style>
