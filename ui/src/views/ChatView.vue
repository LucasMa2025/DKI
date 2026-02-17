<template>
  <div class="chat-view">
    <!-- Chat Header -->
    <header class="chat-header">
      <div class="header-left">
        <h2 class="session-title">{{ currentSession?.title || 'New Chat' }}</h2>
        <el-tag v-if="settingsStore.dkiEnabled" type="success" size="small">
          DKI Enabled
        </el-tag>
      </div>
      <div class="header-right">
        <el-tooltip content="DKI Debug Info" v-if="settingsStore.dkiDebugMode">
          <el-button :icon="InfoFilled" text @click="showDebugPanel = !showDebugPanel" />
        </el-tooltip>
        <el-tooltip content="Clear Chat">
          <el-button :icon="Delete" text @click="handleClearChat" />
        </el-tooltip>
      </div>
    </header>
    
    <!-- Messages Area -->
    <div class="messages-container" ref="messagesContainer">
      <!-- Empty state -->
      <div v-if="messages.length === 0" class="empty-state">
        <img src="/logo.svg" alt="DKI" class="empty-logo" />
        <h3>Start a New Chat</h3>
        <p>Ask your questions, DKI will provide personalized answers based on your preferences and history</p>
        <div class="quick-prompts">
          <el-button
            v-for="prompt in quickPrompts"
            :key="prompt"
            class="quick-prompt-btn"
            @click="handleQuickPrompt(prompt)"
          >
            {{ prompt }}
          </el-button>
        </div>
      </div>
      
      <!-- Messages -->
      <div v-else class="messages-list">
        <div
          v-for="message in messages"
          :key="message.id"
          class="message-wrapper"
          :class="[`message-${message.role}`]"
        >
          <div class="message-avatar">
            <el-avatar v-if="message.role === 'user'" :size="36">
              {{ authStore.user?.username?.charAt(0).toUpperCase() }}
            </el-avatar>
            <div v-else class="assistant-avatar">
              <img src="/logo.svg" alt="DKI" />
            </div>
          </div>
          
          <div class="message-content">
            <div class="message-header">
              <span class="message-author">
                {{ message.role === 'user' ? authStore.user?.username : 'DKI Assistant' }}
              </span>
              <span class="message-time" v-if="settingsStore.showTimestamps">
                {{ formatTime(message.timestamp) }}
              </span>
            </div>
            
            <div
              class="message-body"
              :class="{ loading: message.content === '' && chatStore.loading }"
            >
              <template v-if="message.content === '' && chatStore.loading">
                <div class="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </template>
              <template v-else>
                <div
                  v-if="message.role === 'assistant'"
                  class="markdown-content"
                  v-html="renderMarkdown(message.content)"
                />
                <div v-else class="plain-content">{{ message.content }}</div>
              </template>
            </div>
            
            <!-- DKI Metadata Badge -->
            <div
              v-if="message.dkiMetadata && settingsStore.dkiDebugMode"
              class="dki-metadata"
            >
              <el-tag size="small" :type="message.dkiMetadata.cacheHit ? 'success' : 'info'">
                {{ message.dkiMetadata.cacheTier || 'COMPUTE' }}
              </el-tag>
              <el-tag size="small">α={{ message.dkiMetadata.alpha?.toFixed(2) }}</el-tag>
              <el-tag size="small">{{ message.dkiMetadata.latencyMs }}ms</el-tag>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Input Area -->
    <div class="input-area">
      <div class="input-container">
        <el-input
          v-model="inputMessage"
          type="textarea"
          :rows="1"
          :autosize="{ minRows: 1, maxRows: 6 }"
          placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
          resize="none"
          @keydown="handleKeydown"
        />
        <el-button
          type="primary"
          :icon="Promotion"
          :loading="chatStore.loading"
          :disabled="!inputMessage.trim()"
          @click="handleSend"
        />
      </div>
      <div class="input-footer">
        <span class="input-hint">
          DKI {{ settingsStore.dkiEnabled ? 'Enabled' : 'Disabled' }} · 
          {{ settingsStore.dkiUseHybrid ? 'Hybrid Injection' : 'Standard Injection' }}
        </span>
      </div>
    </div>
    
    <!-- Debug Panel -->
    <el-drawer
      v-model="showDebugPanel"
      title="DKI Debug Info"
      direction="rtl"
      size="400px"
    >
      <div class="debug-panel">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="DKI Status">
            <el-tag :type="settingsStore.dkiEnabled ? 'success' : 'danger'">
              {{ settingsStore.dkiEnabled ? 'Enabled' : 'Disabled' }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="Injection Mode">
            {{ settingsStore.dkiUseHybrid ? 'Hybrid Injection' : 'Standard Injection' }}
          </el-descriptions-item>
          <el-descriptions-item label="Default Alpha">
            {{ settingsStore.dkiDefaultAlpha }}
          </el-descriptions-item>
          <el-descriptions-item label="Current Session">
            {{ currentSession?.id || 'None' }}
          </el-descriptions-item>
          <el-descriptions-item label="Message Count">
            {{ messages.length }}
          </el-descriptions-item>
        </el-descriptions>
        
        <h4>Latest Injection Details</h4>
        <div v-if="lastDkiMetadata" class="last-injection">
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="Injection Enabled">
              {{ lastDkiMetadata.injectionEnabled ? 'Yes' : 'No' }}
            </el-descriptions-item>
            <el-descriptions-item label="Alpha Value">
              {{ lastDkiMetadata.alpha?.toFixed(4) }}
            </el-descriptions-item>
            <el-descriptions-item label="Preference Tokens">
              {{ lastDkiMetadata.preferenceTokens }}
            </el-descriptions-item>
            <el-descriptions-item label="History Tokens">
              {{ lastDkiMetadata.historyTokens }}
            </el-descriptions-item>
            <el-descriptions-item label="Cache Tier">
              {{ lastDkiMetadata.cacheTier || 'N/A' }}
            </el-descriptions-item>
            <el-descriptions-item label="Latency">
              {{ lastDkiMetadata.latencyMs }}ms
            </el-descriptions-item>
          </el-descriptions>
          
          <div v-if="lastDkiMetadata.gatingDecision" class="gating-info">
            <h5>Gating Decision</h5>
            <el-descriptions :column="1" border size="small">
              <el-descriptions-item label="Should Inject">
                {{ lastDkiMetadata.gatingDecision.shouldInject ? 'Yes' : 'No' }}
              </el-descriptions-item>
              <el-descriptions-item label="Relevance Score">
                {{ lastDkiMetadata.gatingDecision.relevanceScore?.toFixed(4) }}
              </el-descriptions-item>
              <el-descriptions-item label="Entropy">
                {{ lastDkiMetadata.gatingDecision.entropy?.toFixed(4) }}
              </el-descriptions-item>
              <el-descriptions-item label="Reasoning">
                {{ lastDkiMetadata.gatingDecision.reasoning }}
              </el-descriptions-item>
            </el-descriptions>
          </div>
        </div>
        <el-empty v-else description="No injection data yet" />
      </div>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, watch } from 'vue'
import { Delete, InfoFilled, Promotion } from '@element-plus/icons-vue'
import { ElMessageBox } from 'element-plus'
import { useChatStore } from '@/stores/chat'
import { useAuthStore } from '@/stores/auth'
import { useSettingsStore } from '@/stores/settings'
import { renderMarkdown } from '@/utils/markdown'
import dayjs from 'dayjs'

const chatStore = useChatStore()
const authStore = useAuthStore()
const settingsStore = useSettingsStore()

const messagesContainer = ref<HTMLElement>()
const inputMessage = ref('')
const showDebugPanel = ref(false)

const messages = computed(() => chatStore.messages)
const currentSession = computed(() => chatStore.currentSession)

const lastDkiMetadata = computed(() => {
  for (let i = messages.value.length - 1; i >= 0; i--) {
    if (messages.value[i].dkiMetadata) {
      return messages.value[i].dkiMetadata
    }
  }
  return null
})

const quickPrompts = [
  'Explain how the DKI system works',
  'How can I optimize my user preferences?',
  'Explain the advantages of hybrid injection strategy',
  'Help me analyze a piece of code',
]

function formatTime(timestamp: string) {
  return dayjs(timestamp).format('HH:mm')
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

async function handleSend() {
  if (!inputMessage.value.trim() || chatStore.loading) return
  
  const message = inputMessage.value
  inputMessage.value = ''
  
  await chatStore.sendMessage(message)
  scrollToBottom()
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey && settingsStore.sendOnEnter) {
    e.preventDefault()
    handleSend()
  }
}

function handleQuickPrompt(prompt: string) {
  inputMessage.value = prompt
  handleSend()
}

async function handleClearChat() {
  if (messages.value.length === 0) return
  
  await ElMessageBox.confirm('Are you sure you want to clear the current chat?', 'Clear Chat', {
    type: 'warning',
    confirmButtonText: 'Clear',
    cancelButtonText: 'Cancel',
  })
  
  chatStore.clearMessages()
}

// Auto scroll on new messages
watch(
  () => messages.value.length,
  () => scrollToBottom()
)
</script>

<style lang="scss" scoped>
.chat-view {
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: var(--bg-color);
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border-color);
  background-color: var(--bg-surface);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
  
  .session-title {
    font-size: 16px;
    font-weight: 600;
    margin: 0;
    color: var(--text-primary);
  }
}

.header-right {
  display: flex;
  align-items: center;
  gap: 4px;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

.empty-state {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: var(--text-secondary);
  
  .empty-logo {
    width: 80px;
    height: 80px;
    opacity: 0.5;
    margin-bottom: 24px;
  }
  
  h3 {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 8px;
  }
  
  p {
    font-size: 14px;
    max-width: 400px;
    margin: 0 0 32px;
  }
}

.quick-prompts {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  max-width: 600px;
}

.quick-prompt-btn {
  border-radius: 20px;
  font-size: 13px;
}

.messages-list {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.message-wrapper {
  display: flex;
  gap: 16px;
  animation: slideUp 0.3s ease;
  
  &.message-user {
    .message-body {
      background-color: var(--primary-color);
      color: white;
      border-radius: 16px 16px 4px 16px;
    }
  }
  
  &.message-assistant {
    .message-body {
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 16px 16px 16px 4px;
    }
  }
}

.message-avatar {
  flex-shrink: 0;
  
  .assistant-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #10b981, #059669);
    display: flex;
    align-items: center;
    justify-content: center;
    
    img {
      width: 24px;
      height: 24px;
      filter: brightness(0) invert(1);
    }
  }
}

.message-content {
  flex: 1;
  min-width: 0;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  
  .message-author {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .message-time {
    font-size: 12px;
    color: var(--text-muted);
  }
}

.message-body {
  padding: 12px 16px;
  max-width: 80%;
  
  &.loading {
    padding: 16px 24px;
  }
}

.typing-indicator {
  display: flex;
  gap: 4px;
  
  span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--text-muted);
    animation: typing 1.4s infinite ease-in-out;
    
    &:nth-child(1) { animation-delay: 0s; }
    &:nth-child(2) { animation-delay: 0.2s; }
    &:nth-child(3) { animation-delay: 0.4s; }
  }
}

@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-8px); }
}

.markdown-content {
  font-size: 14px;
  line-height: 1.7;
  
  :deep(p) {
    margin: 0 0 12px;
    
    &:last-child {
      margin-bottom: 0;
    }
  }
  
  :deep(pre) {
    margin: 12px 0;
    border-radius: 8px;
    overflow-x: auto;
    
    code {
      font-family: 'Fira Code', 'Consolas', monospace;
      font-size: 13px;
    }
  }
  
  :deep(code:not(pre code)) {
    background-color: var(--bg-hover);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
  }
  
  :deep(ul), :deep(ol) {
    margin: 12px 0;
    padding-left: 24px;
  }
  
  :deep(blockquote) {
    margin: 12px 0;
    padding-left: 16px;
    border-left: 4px solid var(--primary-color);
    color: var(--text-secondary);
  }
  
  :deep(.table-wrapper) {
    overflow-x: auto;
    margin: 12px 0;
  }
  
  :deep(table) {
    border-collapse: collapse;
    width: 100%;
    
    th, td {
      border: 1px solid var(--border-color);
      padding: 8px 12px;
      text-align: left;
    }
    
    th {
      background-color: var(--bg-hover);
    }
  }
}

.plain-content {
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.dki-metadata {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  flex-wrap: wrap;
}

.input-area {
  padding: 16px 24px;
  border-top: 1px solid var(--border-color);
  background-color: var(--bg-surface);
}

.input-container {
  display: flex;
  gap: 12px;
  align-items: flex-end;
  
  .el-input {
    flex: 1;
    
    :deep(.el-textarea__inner) {
      border-radius: 12px;
      padding: 12px 16px;
      font-size: 14px;
      resize: none;
    }
  }
  
  .el-button {
    height: 44px;
    width: 44px;
    border-radius: 12px;
  }
}

.input-footer {
  margin-top: 8px;
  text-align: center;
  
  .input-hint {
    font-size: 12px;
    color: var(--text-muted);
  }
}

.debug-panel {
  h4, h5 {
    margin: 24px 0 12px;
    color: var(--text-primary);
  }
  
  h4:first-child {
    margin-top: 0;
  }
}

.last-injection {
  .gating-info {
    margin-top: 16px;
  }
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
