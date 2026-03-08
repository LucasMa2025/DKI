<template>
  <div class="chat-view">
    <!-- Chat Header -->
    <header class="chat-header">
      <div class="header-left">
        <template v-if="editingTitle">
          <el-input
            v-model="editTitleValue"
            size="small"
            class="title-input"
            maxlength="50"
            show-word-limit
            @keydown.enter="saveTitle"
            @keydown.escape="cancelEditTitle"
            @blur="saveTitle"
            ref="titleInputRef"
          />
          <el-button size="small" type="primary" text @click="saveTitle">保存</el-button>
          <el-button size="small" text @click="cancelEditTitle">取消</el-button>
        </template>
        <template v-else>
          <h2
            class="session-title"
            :title="'点击编辑会话名称'"
            @click="startEditTitle"
          >
            {{ currentSession?.title || 'New Chat' }}
          </h2>
          <el-icon
            v-if="currentSession"
            class="edit-title-icon"
            @click="startEditTitle"
          >
            <Edit />
          </el-icon>
        </template>
        <el-tag v-if="settingsStore.dkiEnabled" type="success" size="small">
          DKI Enabled
        </el-tag>
        <el-tag v-if="settingsStore.streamingEnabled" type="warning" size="small">
          Stream
        </el-tag>
      </div>
      <div class="header-right">
        <!-- 对话锚点导航按钮 -->
        <el-tooltip content="对话导航" v-if="userAnchors.length > 0">
          <el-button :icon="List" text @click="showAnchorPanel = !showAnchorPanel" />
        </el-tooltip>
        <el-tooltip content="DKI Debug Info" v-if="settingsStore.dkiDebugMode">
          <el-button :icon="InfoFilled" text @click="showDebugPanel = !showDebugPanel" />
        </el-tooltip>
        <el-tooltip content="Clear Chat">
          <el-button :icon="Delete" text @click="handleClearChat" />
        </el-tooltip>
      </div>
    </header>
    
    <!-- Messages Area -->
    <div class="messages-container" ref="messagesContainer" @scroll="handleScroll">
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
          v-for="(message, idx) in messages"
          :key="message.id"
          :id="'msg-' + message.id"
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
              :class="{
                loading: message.content === '' && (chatStore.loading || chatStore.streaming),
                streaming: chatStore.streaming && idx === messages.length - 1 && message.role === 'assistant',
              }"
            >
              <template v-if="message.content === '' && (chatStore.loading || chatStore.streaming)">
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
              <el-tag v-if="message.dkiMetadata.retrievalMode && message.dkiMetadata.retrievalMode !== 'unknown'" size="small" type="warning">
                {{ message.dkiMetadata.retrievalMode }}
              </el-tag>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Scroll to Top / Bottom Buttons -->
    <transition name="fade">
      <div v-if="showScrollTop" class="scroll-btn scroll-top-btn" @click="scrollToTop">
        <el-icon><ArrowUp /></el-icon>
      </div>
    </transition>
    <transition name="fade">
      <div v-if="showScrollBottom" class="scroll-btn scroll-bottom-btn" @click="scrollToBottom">
        <el-icon><ArrowDown /></el-icon>
      </div>
    </transition>
    
    <!-- Anchor Navigation Panel (右侧浮层) -->
    <transition name="slide-right">
      <div v-if="showAnchorPanel && userAnchors.length > 0" class="anchor-panel">
        <div class="anchor-panel-header">
          <span>对话导航</span>
          <el-icon class="anchor-close" @click="showAnchorPanel = false"><Close /></el-icon>
        </div>
        <div class="anchor-list">
          <div
            v-for="(anchor, idx) in userAnchors"
            :key="anchor.id"
            class="anchor-item"
            :class="{ active: activeAnchorId === anchor.id }"
            @click="scrollToAnchor(anchor.id)"
          >
            <span class="anchor-index">{{ idx + 1 }}</span>
            <span class="anchor-text">{{ anchor.preview }}</span>
          </div>
        </div>
      </div>
    </transition>
    
    <!-- Input Area -->
    <div class="input-area">
      <div class="input-container">
        <el-input
          v-model="inputMessage"
          type="textarea"
          :rows="1"
          :autosize="{ minRows: 1, maxRows: 10 }"
          placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
          resize="none"
          class="chat-textarea"
          @keydown="handleKeydown"
        />
        <el-button
          type="primary"
          :icon="Promotion"
          :loading="chatStore.loading || chatStore.streaming"
          :disabled="!inputMessage.trim()"
          @click="handleSend"
        />
      </div>
      <div class="input-footer">
        <span class="input-hint">
          DKI {{ settingsStore.dkiEnabled ? 'Enabled' : 'Disabled' }} · 
          {{ settingsStore.dkiUseHybrid ? 'Hybrid Injection' : 'Standard Injection' }}
          {{ settingsStore.streamingEnabled ? ' · Streaming' : '' }}
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
          <el-descriptions-item label="Streaming">
            {{ settingsStore.streamingEnabled ? 'Enabled' : 'Disabled' }}
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
import { ref, computed, nextTick, watch, onMounted, onUnmounted } from 'vue'
import { Delete, Edit, InfoFilled, Promotion, ArrowUp, ArrowDown, List, Close } from '@element-plus/icons-vue'
import { ElMessageBox, ElMessage } from 'element-plus'
import { useChatStore } from '@/stores/chat'
import { useAuthStore } from '@/stores/auth'
import { useSettingsStore } from '@/stores/settings'
import { renderMarkdown } from '@/utils/markdown'
import { api } from '@/services/api'
import type { ChatRequest } from '@/types'
import dayjs from 'dayjs'

const chatStore = useChatStore()
const authStore = useAuthStore()
const settingsStore = useSettingsStore()

const messagesContainer = ref<HTMLElement>()
const inputMessage = ref('')
const showDebugPanel = ref(false)

// 标题编辑状态
const editingTitle = ref(false)
const editTitleValue = ref('')
const titleInputRef = ref<any>(null)

// 滚动按钮状态
const showScrollTop = ref(false)
const showScrollBottom = ref(false)

// 锚点导航状态
const showAnchorPanel = ref(false)
const activeAnchorId = ref<string | null>(null)

// 当前活跃的 EventSource (流式)
let activeEventSource: EventSource | null = null

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

// 对话锚点: 提取所有用户消息的前 5-20 个字作为预览
const userAnchors = computed(() => {
  return messages.value
    .filter(m => m.role === 'user' && m.content)
    .map(m => {
      const text = m.content.trim().replace(/\s+/g, ' ')
      const maxLen = 20
      const preview = text.length > maxLen ? text.slice(0, maxLen) + '...' : text
      return { id: m.id, preview }
    })
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

// ============ 滚动控制 ============
function handleScroll() {
  if (!messagesContainer.value) return
  const el = messagesContainer.value
  const scrollTop = el.scrollTop
  const scrollHeight = el.scrollHeight
  const clientHeight = el.clientHeight
  
  // 距离顶部超过 200px 时显示回到顶部按钮
  showScrollTop.value = scrollTop > 200
  // 距离底部超过 200px 时显示回到底部按钮
  showScrollBottom.value = (scrollHeight - scrollTop - clientHeight) > 200
}

function scrollToTop() {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTo({ top: 0, behavior: 'smooth' })
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTo({
        top: messagesContainer.value.scrollHeight,
        behavior: 'smooth',
      })
    }
  })
}

// ============ 锚点导航 ============
function scrollToAnchor(messageId: string) {
  const el = document.getElementById('msg-' + messageId)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    activeAnchorId.value = messageId
    // 高亮闪烁效果
    el.classList.add('anchor-highlight')
    setTimeout(() => el.classList.remove('anchor-highlight'), 1500)
  }
}

// ============ 发送消息 (支持普通/流式切换) ============
async function handleSend() {
  if (!inputMessage.value.trim() || chatStore.loading || chatStore.streaming) return
  
  const message = inputMessage.value
  inputMessage.value = ''
  
  if (settingsStore.streamingEnabled) {
    await handleSendStream(message)
  } else {
    await chatStore.sendMessage(message)
  }
  scrollToBottom()
}

// 流式发送
async function handleSendStream(content: string) {
  if (!content.trim()) return
  
  const isNewSession = !chatStore.currentSessionId
  
  // 确保有会话
  if (!chatStore.currentSessionId) {
    const session = await chatStore.createSession()
    if (!session) return
    chatStore.currentSessionId = session.id
  }
  
  // 添加用户消息到本地
  const userMessage = {
    id: `temp-${Date.now()}`,
    sessionId: chatStore.currentSessionId!,
    role: 'user' as const,
    content: content.trim(),
    timestamp: new Date().toISOString(),
  }
  chatStore.messages.push(userMessage)
  
  // 添加助手占位消息
  const assistantMessage = {
    id: `temp-assistant-${Date.now()}`,
    sessionId: chatStore.currentSessionId!,
    role: 'assistant' as const,
    content: '',
    timestamp: new Date().toISOString(),
  }
  chatStore.messages.push(assistantMessage)
  
  chatStore.streaming = true
  chatStore.loading = true
  
  try {
    const request: ChatRequest = {
      query: content.trim(),
      dkiUserId: authStore.user?.id,
      dkiSessionId: chatStore.currentSessionId!,
      model: settingsStore.defaultModel,
      temperature: settingsStore.temperature,
      maxTokens: settingsStore.maxTokens,
      stream: true,
      forceAlpha: settingsStore.dkiEnabled ? settingsStore.dkiDefaultAlpha : 0,
      useHybrid: settingsStore.dkiUseHybrid,
    }
    
    // 使用 fetch + ReadableStream 处理 SSE (比 EventSource 更灵活, 支持 POST)
    const authData = localStorage.getItem('auth')
    let token = ''
    if (authData) {
      try { token = JSON.parse(authData).token || '' } catch { /* ignore */ }
    }
    
    const baseUrl = settingsStore.apiBaseUrl || '/api'
    const response = await fetch(`${baseUrl}/v1/dki/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        query: request.query,
        user_id: request.dkiUserId,
        session_id: request.dkiSessionId,
        model: request.model,
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        force_alpha: request.forceAlpha,
        use_hybrid: request.useHybrid,
      }),
    })
    
    if (!response.ok) {
      throw new Error(`Stream request failed: ${response.status} ${response.statusText}`)
    }
    
    const reader = response.body?.getReader()
    if (!reader) throw new Error('No readable stream')
    
    const decoder = new TextDecoder()
    let buffer = ''
    chatStore.loading = false  // 停止 loading 动画, 开始流式显示
    
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      buffer += decoder.decode(value, { stream: true })
      
      // 解析 SSE 事件
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''  // 保留不完整的行
      
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          // 事件类型行, 下一行是 data
          continue
        }
        if (line.startsWith('data: ')) {
          const dataStr = line.slice(6)
          try {
            const data = JSON.parse(dataStr)
            const lastMsg = chatStore.messages[chatStore.messages.length - 1]
            
            if (data.content !== undefined && lastMsg?.role === 'assistant') {
              // token 事件
              lastMsg.content += data.content
              scrollToBottom()
            } else if (data.text !== undefined) {
              // done 事件
              if (lastMsg?.role === 'assistant') {
                lastMsg.content = data.text
              }
            } else if (data.error) {
              // error 事件
              if (lastMsg?.role === 'assistant') {
                lastMsg.content = `❌ 流式生成错误: ${data.error}`
              }
            }
          } catch {
            // 忽略解析错误
          }
        }
      }
    }
    
    // 更新会话信息
    const session = chatStore.sessions.find(s => s.id === chatStore.currentSessionId)
    if (session) {
      session.preview = content.slice(0, 50)
      session.messageCount = chatStore.messages.length
      session.updatedAt = new Date().toISOString()
    }
    
    // 自动命名新会话
    if (isNewSession && chatStore.currentSessionId) {
      const autoTitle = chatStore.generateSessionTitle
        ? (chatStore as any).generateSessionTitle(content)
        : content.slice(0, 30)
      chatStore.renameSession(chatStore.currentSessionId, autoTitle).catch(() => {})
    }
    
  } catch (e) {
    const errMsg = e instanceof Error ? e.message : 'Stream failed'
    chatStore.error = errMsg
    const lastMsg = chatStore.messages[chatStore.messages.length - 1]
    if (lastMsg?.role === 'assistant' && !lastMsg.content) {
      lastMsg.content = `❌ 流式请求失败: ${errMsg}`
    }
  } finally {
    chatStore.streaming = false
    chatStore.loading = false
  }
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

// ============ 标题编辑 ============
function startEditTitle() {
  if (!currentSession.value) return
  editTitleValue.value = currentSession.value.title || ''
  editingTitle.value = true
  nextTick(() => {
    titleInputRef.value?.focus?.()
  })
}

async function saveTitle() {
  if (!editingTitle.value) return
  const newTitle = editTitleValue.value.trim()
  if (!newTitle || !currentSession.value) {
    cancelEditTitle()
    return
  }
  if (newTitle !== currentSession.value.title) {
    await chatStore.renameSession(currentSession.value.id, newTitle)
    ElMessage.success('会话名称已更新')
  }
  editingTitle.value = false
}

function cancelEditTitle() {
  editingTitle.value = false
  editTitleValue.value = ''
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

// 清理流式连接
onUnmounted(() => {
  if (activeEventSource) {
    activeEventSource.close()
    activeEventSource = null
  }
})

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
    cursor: pointer;
    transition: color 0.2s;
    
    &:hover {
      color: var(--primary-color);
    }
  }
  
  .edit-title-icon {
    font-size: 14px;
    color: var(--text-muted);
    cursor: pointer;
    transition: color 0.2s;
    
    &:hover {
      color: var(--primary-color);
    }
  }
  
  .title-input {
    width: 300px;
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
  
  .chat-textarea {
    flex: 1;
    
    :deep(.el-textarea__inner) {
      border-radius: 12px;
      padding: 12px 16px;
      font-size: 14px;
      line-height: 1.6;
      resize: none;
      max-height: 240px;      /* 最大高度限制 ≈ 10 行 */
      overflow-y: auto;
      transition: height 0.15s ease;
      scrollbar-width: thin;   /* Firefox 细滚动条 */
      
      &::-webkit-scrollbar {
        width: 4px;
      }
      &::-webkit-scrollbar-thumb {
        background-color: var(--border-color);
        border-radius: 4px;
      }
    }
  }
  
  .el-button {
    height: 44px;
    width: 44px;
    border-radius: 12px;
    flex-shrink: 0;
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

// ============ 滚动按钮 ============
.scroll-btn {
  position: fixed;
  right: 40px;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: var(--bg-surface);
  border: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  z-index: 100;
  transition: all 0.2s ease;
  
  &:hover {
    background-color: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
    transform: scale(1.1);
  }
  
  .el-icon {
    font-size: 18px;
  }
}

.scroll-top-btn {
  bottom: 160px;
}

.scroll-bottom-btn {
  bottom: 110px;
}

// ============ 锚点导航面板 ============
.anchor-panel {
  position: fixed;
  right: 16px;
  top: 50%;
  transform: translateY(-50%);
  width: 220px;
  max-height: 60vh;
  background-color: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  z-index: 200;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.anchor-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color);
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  
  .anchor-close {
    cursor: pointer;
    color: var(--text-muted);
    transition: color 0.2s;
    
    &:hover {
      color: var(--text-primary);
    }
  }
}

.anchor-list {
  overflow-y: auto;
  padding: 8px 0;
  scrollbar-width: thin;
  
  &::-webkit-scrollbar {
    width: 4px;
  }
  &::-webkit-scrollbar-thumb {
    background-color: var(--border-color);
    border-radius: 4px;
  }
}

.anchor-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  cursor: pointer;
  transition: all 0.15s ease;
  
  &:hover {
    background-color: var(--bg-hover);
  }
  
  &.active {
    background-color: rgba(var(--primary-color-rgb, 64, 158, 255), 0.1);
    
    .anchor-index {
      background-color: var(--primary-color);
      color: white;
    }
  }
  
  .anchor-index {
    flex-shrink: 0;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: var(--bg-hover);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
  }
  
  .anchor-text {
    font-size: 13px;
    color: var(--text-secondary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
}

// ============ 锚点高亮动画 ============
.anchor-highlight {
  animation: anchorFlash 1.5s ease;
}

@keyframes anchorFlash {
  0%, 100% { background-color: transparent; }
  25% { background-color: rgba(var(--primary-color-rgb, 64, 158, 255), 0.15); }
  50% { background-color: transparent; }
  75% { background-color: rgba(var(--primary-color-rgb, 64, 158, 255), 0.08); }
}

// ============ 流式消息光标 ============
.message-body.streaming {
  .markdown-content::after,
  .plain-content::after {
    content: '▌';
    animation: blink 0.8s infinite;
    color: var(--primary-color);
    font-weight: bold;
  }
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

// ============ 过渡动画 ============
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.slide-right-enter-active,
.slide-right-leave-active {
  transition: all 0.3s ease;
}
.slide-right-enter-from,
.slide-right-leave-to {
  opacity: 0;
  transform: translateY(-50%) translateX(20px);
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
