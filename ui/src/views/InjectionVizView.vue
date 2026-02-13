<template>
  <div class="injection-viz-view">
    <!-- Header -->
    <header class="page-header">
      <div class="header-content">
        <h1>注入可视化</h1>
        <p>DKI 注入过程的详细可视化分析</p>
      </div>
      <div class="header-actions">
        <el-button :icon="Refresh" @click="refreshData" :loading="loading">
          刷新
        </el-button>
      </div>
    </header>

    <!-- Empty State Notice -->
    <div v-if="!hasRealData && !loading" class="empty-notice">
      <el-alert
        type="info"
        title="暂无注入数据"
        description="请先通过聊天界面发送消息，系统会自动记录每次注入的详细过程。下方流程图为 DKI 系统的标准注入流程说明。"
        show-icon
        :closable="false"
      />
    </div>

    <!-- Main Content -->
    <div class="viz-content">
      <!-- Flow Diagram Section -->
      <section class="flow-section">
        <h2>
          <el-icon><Connection /></el-icon>
          注入流程图
        </h2>
        <div class="flow-diagram">
          <div class="flow-nodes">
            <!-- Input Node -->
            <div class="flow-node input-node">
              <div class="node-icon">
                <el-icon><Edit /></el-icon>
              </div>
              <div class="node-label">用户输入</div>
              <div class="node-detail">原始查询</div>
            </div>

            <div class="flow-arrow">→</div>

            <!-- Adapter Node -->
            <div class="flow-node process-node">
              <div class="node-icon">
                <el-icon><DataLine /></el-icon>
              </div>
              <div class="node-label">外部数据适配器</div>
              <div class="node-detail">读取偏好 + 历史</div>
            </div>

            <div class="flow-arrow-split">
              <div class="arrow-up">↗</div>
              <div class="arrow-down">↘</div>
            </div>

            <!-- Injection Nodes -->
            <div class="injection-nodes">
              <div class="flow-node injection-node preference-node">
                <div class="node-icon">
                  <el-icon><User /></el-icon>
                </div>
                <div class="node-label">偏好 K/V 注入</div>
                <div class="node-detail">负位置 (α={{ latestData?.injection_layers?.[0]?.alpha?.toFixed(2) || '0.40' }})</div>
                <div class="node-tokens" v-if="latestData?.token_distribution?.preference">
                  {{ latestData.token_distribution.preference }} tokens
                </div>
              </div>

              <div class="flow-node injection-node history-node">
                <div class="node-icon">
                  <el-icon><ChatDotRound /></el-icon>
                </div>
                <div class="node-label">历史后缀注入</div>
                <div class="node-detail">正位置 (显式)</div>
                <div class="node-tokens" v-if="latestData?.token_distribution?.history">
                  {{ latestData.token_distribution.history }} tokens
                </div>
              </div>
            </div>

            <div class="flow-arrow-merge">
              <div class="arrow-down-merge">↘</div>
              <div class="arrow-up-merge">↗</div>
            </div>

            <!-- LLM Node -->
            <div class="flow-node process-node llm-node">
              <div class="node-icon">
                <el-icon><Cpu /></el-icon>
              </div>
              <div class="node-label">LLM 推理</div>
              <div class="node-detail">带注入的注意力计算</div>
            </div>

            <div class="flow-arrow">→</div>

            <!-- Output Node -->
            <div class="flow-node output-node">
              <div class="node-icon">
                <el-icon><ChatLineSquare /></el-icon>
              </div>
              <div class="node-label">输出响应</div>
              <div class="node-detail">个性化回复</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Token Distribution Section -->
      <section class="token-section">
        <h2>
          <el-icon><PieChart /></el-icon>
          Token 分布
        </h2>
        <div class="token-charts">
          <div class="chart-card">
            <v-chart :option="tokenPieOption" autoresize style="height: 280px" />
          </div>
          <div class="token-details">
            <div class="token-item">
              <div class="token-color query"></div>
              <div class="token-info">
                <span class="token-label">查询 (Query)</span>
                <span class="token-value">{{ latestData?.token_distribution?.query || 0 }} tokens</span>
              </div>
              <div class="token-desc">用户原始输入，正位置</div>
            </div>
            <div class="token-item">
              <div class="token-color preference"></div>
              <div class="token-info">
                <span class="token-label">偏好 (Preference)</span>
                <span class="token-value">{{ latestData?.token_distribution?.preference || 0 }} tokens</span>
              </div>
              <div class="token-desc">K/V 注入，负位置，不占用 Context</div>
            </div>
            <div class="token-item">
              <div class="token-color history"></div>
              <div class="token-info">
                <span class="token-label">历史 (History)</span>
                <span class="token-value">{{ latestData?.token_distribution?.history || 0 }} tokens</span>
              </div>
              <div class="token-desc">后缀注入，正位置，占用 Context</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Injection Layers Section -->
      <section class="layers-section">
        <h2>
          <el-icon><Grid /></el-icon>
          注入层详情
        </h2>
        <div class="layers-grid">
          <div class="layer-card l1">
            <div class="layer-header">
              <span class="layer-badge">L1</span>
              <span class="layer-name">偏好层</span>
            </div>
            <div class="layer-body">
              <div class="layer-method">
                <el-tag type="success" size="small">K/V 注入</el-tag>
                <el-tag type="info" size="small">负位置</el-tag>
              </div>
              <div class="layer-stats">
                <div class="stat-item">
                  <span class="stat-label">Token 数</span>
                  <span class="stat-value">{{ latestData?.injection_layers?.[0]?.token_count || 0 }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Alpha</span>
                  <span class="stat-value">{{ latestData?.injection_layers?.[0]?.alpha?.toFixed(2) || '0.40' }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">缓存</span>
                  <span class="stat-value">{{ latestData?.injection_layers?.[0]?.cache_status || 'N/A' }}</span>
                </div>
              </div>
              <div class="layer-desc">
                短期稳定的用户偏好，通过 K/V 缓存注入到注意力计算中。
                <strong>不占用 Context 窗口</strong>。
              </div>
            </div>
          </div>

          <div class="layer-card l2">
            <div class="layer-header">
              <span class="layer-badge">L2</span>
              <span class="layer-name">历史层</span>
            </div>
            <div class="layer-body">
              <div class="layer-method">
                <el-tag type="warning" size="small">后缀注入</el-tag>
                <el-tag type="info" size="small">正位置</el-tag>
              </div>
              <div class="layer-stats">
                <div class="stat-item">
                  <span class="stat-label">Token 数</span>
                  <span class="stat-value">{{ latestData?.injection_layers?.[1]?.token_count || 0 }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Alpha</span>
                  <span class="stat-value">1.00</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">消息数</span>
                  <span class="stat-value">{{ historyMessageCount }}</span>
                </div>
              </div>
              <div class="layer-desc">
                动态的会话历史，作为后缀拼接到输入中。
                <strong>占用 Context 窗口</strong>，但支持显式引用。
              </div>
            </div>
          </div>

          <div class="layer-card l3">
            <div class="layer-header">
              <span class="layer-badge">L3</span>
              <span class="layer-name">查询层</span>
            </div>
            <div class="layer-body">
              <div class="layer-method">
                <el-tag type="primary" size="small">原始输入</el-tag>
                <el-tag type="info" size="small">正位置</el-tag>
              </div>
              <div class="layer-stats">
                <div class="stat-item">
                  <span class="stat-label">Token 数</span>
                  <span class="stat-value">{{ latestData?.token_distribution?.query || 0 }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">位置</span>
                  <span class="stat-value">0 ~ N</span>
                </div>
              </div>
              <div class="layer-desc">
                用户当前的查询输入，作为主要的推理目标。
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Flow Steps Section -->
      <section class="steps-section">
        <h2>
          <el-icon><List /></el-icon>
          注入流程步骤
        </h2>
        <div class="steps-timeline">
          <el-timeline>
            <el-timeline-item
              v-for="step in flowSteps"
              :key="step.step_id"
              :type="getStepType(step.status)"
              :hollow="step.status === 'skipped'"
              :timestamp="`${step.duration_ms.toFixed(1)}ms`"
              placement="top"
            >
              <div class="step-card">
                <div class="step-header">
                  <span class="step-name">{{ step.step_name }}</span>
                  <el-tag :type="getStepTagType(step.status)" size="small">
                    {{ getStepStatusText(step.status) }}
                  </el-tag>
                </div>
                <div class="step-desc">{{ step.description }}</div>
                <div class="step-details" v-if="Object.keys(step.details).length > 0">
                  <el-descriptions :column="3" size="small" border>
                    <el-descriptions-item
                      v-for="(value, key) in step.details"
                      :key="key"
                      :label="formatDetailKey(key)"
                    >
                      {{ formatDetailValue(value) }}
                    </el-descriptions-item>
                  </el-descriptions>
                </div>
              </div>
            </el-timeline-item>
          </el-timeline>
        </div>
      </section>

      <!-- Injection Text Display Section -->
      <section class="injection-text-section">
        <h2>
          <el-icon><Document /></el-icon>
          注入内容显示
          <el-tag v-if="latestData?.mode" :type="latestData?.mode === 'dki' ? 'success' : latestData?.mode === 'rag' ? 'warning' : 'info'" size="small" style="margin-left: 12px;">
            {{ latestData?.mode === 'dki' ? 'DKI 模式' : latestData?.mode === 'rag' ? 'RAG 模式' : '基线模式' }}
          </el-tag>
          <el-button-group style="margin-left: auto;">
            <el-button size="small" @click="toggleInjectionExpand">
              {{ injectionExpanded ? '收起' : '展开' }}
            </el-button>
            <el-button size="small" @click="copyInjectionText">
              <el-icon><CopyDocument /></el-icon>
              复制
            </el-button>
          </el-button-group>
        </h2>
        
        <div class="injection-text-grid" :class="{ expanded: injectionExpanded }">
          <!-- ==================== DKI 模式 ==================== -->
          <template v-if="!latestData?.mode || latestData?.mode === 'dki'">
            <!-- DKI 偏好注入 -->
            <div class="injection-text-card preference-card">
              <div class="card-header">
                <span class="card-title">
                  <el-icon><User /></el-icon>
                  DKI 偏好注入 (K/V)
                </span>
                <el-tag type="success" size="small">
                  {{ latestData?.token_distribution?.preference || 0 }} tokens
                </el-tag>
              </div>
              <div class="card-body">
                <div class="text-label">偏好原文 (不显示实际 K/V):</div>
                <el-input
                  type="textarea"
                  :value="latestData?.preference_text || '(无偏好注入)'"
                  :rows="injectionExpanded ? 8 : 3"
                  readonly
                  resize="none"
                />
                <div class="text-meta">
                  <span>Alpha: {{ latestData?.injection_layers?.[0]?.alpha?.toFixed(2) || '0.00' }}</span>
                  <span>位置: 负位置 (不占用 Context)</span>
                </div>
              </div>
            </div>
            
            <!-- DKI 历史后缀 -->
            <div class="injection-text-card history-card">
              <div class="card-header">
                <span class="card-title">
                  <el-icon><ChatDotRound /></el-icon>
                  DKI 历史后缀 (Suffix)
                </span>
                <el-tag type="warning" size="small">
                  {{ latestData?.token_distribution?.history || 0 }} tokens
                </el-tag>
              </div>
              <div class="card-body">
                <div class="text-label">历史后缀原文:</div>
                <el-input
                  type="textarea"
                  :value="latestData?.history_suffix_text || '(无历史后缀)'"
                  :rows="injectionExpanded ? 8 : 3"
                  readonly
                  resize="none"
                />
                <div class="text-meta">
                  <span>消息数: {{ latestData?.history_messages?.length || 0 }}</span>
                  <span>位置: 正位置 (占用 Context)</span>
                </div>
              </div>
            </div>
          </template>
          
          <!-- ==================== RAG 模式 ==================== -->
          <template v-if="latestData?.mode === 'rag'">
            <!-- RAG 检索上下文 -->
            <div class="injection-text-card rag-context-card">
              <div class="card-header">
                <span class="card-title">
                  <el-icon><DataLine /></el-icon>
                  RAG 检索上下文
                </span>
                <el-tag type="warning" size="small">Context</el-tag>
              </div>
              <div class="card-body">
                <div class="text-label">检索到的上下文:</div>
                <el-input
                  type="textarea"
                  :value="latestData?.rag_context_text || '(无检索上下文)'"
                  :rows="injectionExpanded ? 8 : 3"
                  readonly
                  resize="none"
                />
              </div>
            </div>
            
            <!-- RAG 完整提示词 -->
            <div class="injection-text-card rag-prompt-card">
              <div class="card-header">
                <span class="card-title">
                  <el-icon><Document /></el-icon>
                  RAG 完整提示词
                </span>
                <el-tag type="warning" size="small">Prompt</el-tag>
              </div>
              <div class="card-body">
                <div class="text-label">发送到 LLM 的完整提示词:</div>
                <el-input
                  type="textarea"
                  :value="latestData?.rag_prompt_text || '(无完整提示词)'"
                  :rows="injectionExpanded ? 12 : 5"
                  readonly
                  resize="none"
                />
              </div>
            </div>
          </template>
          
          <!-- ==================== 通用: 历史消息列表 ==================== -->
          <div class="injection-text-card messages-card" v-if="latestData?.history_messages?.length > 0">
            <div class="card-header">
              <span class="card-title">
                <el-icon><List /></el-icon>
                历史消息详情
              </span>
              <el-tag type="info" size="small">
                {{ latestData?.history_messages?.length || 0 }} 条
              </el-tag>
            </div>
            <div class="card-body messages-list">
              <div
                v-for="(msg, idx) in latestData?.history_messages || []"
                :key="idx"
                class="message-item"
                :class="[`message-${msg.role}`]"
              >
                <span class="message-role">{{ msg.role === 'user' ? '用户' : '助手' }}:</span>
                <span class="message-content">{{ msg.content }}</span>
              </div>
              <el-empty v-if="!latestData?.history_messages?.length" description="无历史消息" />
            </div>
          </div>
          
          <!-- ==================== 通用: 最终输入预览 ==================== -->
          <div class="injection-text-card final-card">
            <div class="card-header">
              <span class="card-title">
                <el-icon><View /></el-icon>
                最终输入预览
              </span>
              <el-tag type="primary" size="small">
                {{ latestData?.token_distribution?.total || 0 }} tokens
              </el-tag>
            </div>
            <div class="card-body">
              <el-input
                type="textarea"
                :value="latestData?.final_input_preview || latestData?.original_query || '(无数据)'"
                :rows="injectionExpanded ? 10 : 4"
                readonly
                resize="none"
              />
            </div>
          </div>
        </div>
      </section>

      <!-- History Section -->
      <section class="history-section">
        <h2>
          <el-icon><Clock /></el-icon>
          注入历史
        </h2>
        <el-table :data="historyItems" stripe style="width: 100%">
          <el-table-column prop="timestamp" label="时间" width="180">
            <template #default="{ row }">
              {{ formatTimestamp(row.timestamp) }}
            </template>
          </el-table-column>
          <el-table-column prop="query_preview" label="查询" min-width="200" />
          <el-table-column prop="injection_enabled" label="注入" width="80">
            <template #default="{ row }">
              <el-tag :type="row.injection_enabled ? 'success' : 'info'" size="small">
                {{ row.injection_enabled ? '是' : '否' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="alpha" label="Alpha" width="80">
            <template #default="{ row }">
              {{ row.alpha.toFixed(2) }}
            </template>
          </el-table-column>
          <el-table-column prop="preference_tokens" label="偏好 Tokens" width="100" />
          <el-table-column prop="history_tokens" label="历史 Tokens" width="100" />
          <el-table-column prop="latency_ms" label="延迟" width="100">
            <template #default="{ row }">
              {{ row.latency_ms.toFixed(1) }}ms
            </template>
          </el-table-column>
          <el-table-column label="操作" width="80">
            <template #default="{ row }">
              <el-button type="primary" link size="small" @click="viewDetail(row.request_id)">
                详情
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </section>
    </div>

    <!-- Detail Dialog -->
    <el-dialog
      v-model="showDetailDialog"
      title="注入详情"
      width="800px"
    >
      <div v-if="detailData" class="detail-content">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="请求 ID">{{ detailData.request_id }}</el-descriptions-item>
          <el-descriptions-item label="时间">{{ detailData.timestamp }}</el-descriptions-item>
          <el-descriptions-item label="用户 ID">{{ detailData.user_id }}</el-descriptions-item>
          <el-descriptions-item label="会话 ID">{{ detailData.session_id }}</el-descriptions-item>
          <el-descriptions-item label="总延迟">{{ detailData.total_latency_ms?.toFixed(1) }}ms</el-descriptions-item>
          <el-descriptions-item label="注入开销">{{ detailData.injection_overhead_ms?.toFixed(1) }}ms</el-descriptions-item>
        </el-descriptions>

        <h4>原始查询</h4>
        <el-input
          type="textarea"
          :value="detailData.original_query"
          :rows="3"
          readonly
        />

        <h4>最终输入预览</h4>
        <el-input
          type="textarea"
          :value="detailData.final_input_preview"
          :rows="5"
          readonly
        />
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import {
  Refresh,
  Connection,
  Edit,
  DataLine,
  User,
  ChatDotRound,
  Cpu,
  ChatLineSquare,
  PieChart,
  Grid,
  List,
  Clock,
  Document,
  CopyDocument,
  View,
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart as EchartsPieChart } from 'echarts/charts'
import { TitleComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { api } from '@/services/api'
import dayjs from 'dayjs'

// Register ECharts components
use([CanvasRenderer, EchartsPieChart, TitleComponent, TooltipComponent, LegendComponent])

// State
const loading = ref(false)
const latestData = ref<any>(null)
const historyItems = ref<any[]>([])
const showDetailDialog = ref(false)
const detailData = ref<any>(null)
const injectionExpanded = ref(false)
const hasRealData = ref(false)

// Computed
const flowSteps = computed(() => latestData.value?.flow_steps || defaultFlowSteps)

const historyMessageCount = computed(() => {
  const step = flowSteps.value.find((s: any) => s.step_name === '历史后缀注入')
  return step?.details?.messages_count || 0
})

const tokenPieOption = computed(() => ({
  tooltip: {
    trigger: 'item',
    formatter: '{b}: {c} tokens ({d}%)',
  },
  legend: {
    orient: 'vertical',
    left: 'left',
    textStyle: { color: 'var(--text-secondary)' },
  },
  series: [
    {
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 10,
        borderColor: 'var(--bg-surface)',
        borderWidth: 2,
      },
      label: { show: false },
      emphasis: {
        label: { show: true, fontSize: 14, fontWeight: 'bold' },
      },
      data: [
        {
          value: latestData.value?.token_distribution?.query ?? 0,
          name: '查询',
          itemStyle: { color: '#3b82f6' },
        },
        {
          value: latestData.value?.token_distribution?.preference ?? 0,
          name: '偏好 (K/V)',
          itemStyle: { color: '#10b981' },
        },
        {
          value: latestData.value?.token_distribution?.history ?? 0,
          name: '历史 (后缀)',
          itemStyle: { color: '#f59e0b' },
        },
      ],
    },
  ],
}))

// Default flow steps for demo
const defaultFlowSteps = [
  { step_id: 1, step_name: '接收输入', description: '接收用户原始查询和标识', status: 'completed', duration_ms: 0.1, details: {} },
  { step_id: 2, step_name: 'Memory Trigger 检测', description: '检测是否触发记忆存储/更新', status: 'skipped', duration_ms: 0.5, details: {} },
  { step_id: 3, step_name: 'Reference Resolver 解析', description: '解析指代表达，确定召回范围', status: 'skipped', duration_ms: 0.5, details: {} },
  { step_id: 4, step_name: '读取外部数据', description: '通过适配器读取用户偏好和历史消息', status: 'completed', duration_ms: 5.2, details: { preferences_count: 3, history_count: 8 } },
  { step_id: 5, step_name: '偏好 K/V 注入', description: '将用户偏好编码为 K/V 并注入到负位置', status: 'completed', duration_ms: 2.1, details: { tokens: 45, cache_hit: true, alpha: 0.4 } },
  { step_id: 6, step_name: '历史后缀注入', description: '将相关历史格式化为后缀并拼接到输入', status: 'completed', duration_ms: 1.3, details: { tokens: 120, messages_count: 5 } },
  { step_id: 7, step_name: 'LLM 推理', description: '调用 LLM 进行推理生成', status: 'completed', duration_ms: 156.8, details: { input_tokens: 175 } },
]

// Methods
async function refreshData() {
  loading.value = true
  try {
    // Try to fetch real data from backend
    const [latestRes, historyRes] = await Promise.all([
      api.visualization.getLatest().catch(() => null),
      api.visualization.getHistory().catch(() => ({ items: [] })),
    ])
    
    if (latestRes) {
      latestData.value = latestRes
      hasRealData.value = true
    } else {
      // No visualization data yet - show empty state instead of fake demo data
      latestData.value = null
      hasRealData.value = false
    }
    
    // Only show real history items (no fake demo data)
    historyItems.value = historyRes.items || []
  } finally {
    loading.value = false
  }
}

async function viewDetail(requestId: string) {
  try {
    const res = await api.visualization.getDetail(requestId).catch(() => null)
    if (res) {
      detailData.value = res
    } else {
      // Use demo data
      detailData.value = latestData.value
    }
    showDetailDialog.value = true
  } catch (error) {
    console.error('Failed to load detail:', error)
  }
}

function getStepType(status: string) {
  switch (status) {
    case 'completed': return 'success'
    case 'running': return 'primary'
    case 'skipped': return 'info'
    default: return 'info'
  }
}

function getStepTagType(status: string) {
  switch (status) {
    case 'completed': return 'success'
    case 'running': return 'primary'
    case 'skipped': return 'info'
    default: return 'info'
  }
}

function getStepStatusText(status: string) {
  switch (status) {
    case 'completed': return '完成'
    case 'running': return '进行中'
    case 'skipped': return '跳过'
    case 'pending': return '等待'
    default: return status
  }
}

function formatDetailKey(key: string | number) {
  const k = String(key)
  const keyMap: Record<string, string> = {
    query_length: '查询长度',
    triggered: '已触发',
    trigger_type: '触发类型',
    resolved: '已解析',
    reference_type: '指代类型',
    preferences_count: '偏好数',
    history_count: '历史数',
    tokens: 'Token 数',
    cache_hit: '缓存命中',
    alpha: 'Alpha',
    messages_count: '消息数',
    input_tokens: '输入 Tokens',
  }
  return keyMap[k] || k
}

function formatDetailValue(value: any) {
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'number') return value.toFixed ? value.toFixed(2) : value
  return value || 'N/A'
}

function formatTimestamp(ts: string) {
  return dayjs(ts).format('YYYY-MM-DD HH:mm:ss')
}

function toggleInjectionExpand() {
  injectionExpanded.value = !injectionExpanded.value
}

function copyInjectionText() {
  if (!latestData.value) {
    ElMessage.warning('没有可复制的数据')
    return
  }
  
  const mode = latestData.value.mode || 'dki'
  const lines = []
  lines.push('═══════════════════════════════════════════════════════')
  lines.push(`  ${mode === 'dki' ? 'DKI 注入信息' : mode === 'rag' ? 'RAG 提示词信息' : '基线模式信息'}`)
  lines.push('═══════════════════════════════════════════════════════')
  lines.push('')
  lines.push(`【原始查询】`)
  lines.push(latestData.value.original_query || '(无)')
  lines.push('')
  
  if (mode === 'dki') {
    if (latestData.value.preference_text) {
      lines.push(`【偏好注入】(K/V 注入, α=${latestData.value.injection_layers?.[0]?.alpha?.toFixed(2) || '0.00'})`)
      lines.push('───────────────────────────────────────────────────────')
      lines.push(latestData.value.preference_text)
      lines.push('')
    }
    
    if (latestData.value.history_suffix_text) {
      lines.push(`【历史后缀】(${latestData.value.token_distribution?.history || 0} tokens)`)
      lines.push('───────────────────────────────────────────────────────')
      lines.push(latestData.value.history_suffix_text)
      lines.push('')
    }
  } else if (mode === 'rag') {
    if (latestData.value.rag_context_text) {
      lines.push(`【RAG 检索上下文】`)
      lines.push('───────────────────────────────────────────────────────')
      lines.push(latestData.value.rag_context_text)
      lines.push('')
    }
    
    if (latestData.value.rag_prompt_text) {
      lines.push(`【RAG 完整提示词】`)
      lines.push('───────────────────────────────────────────────────────')
      lines.push(latestData.value.rag_prompt_text)
      lines.push('')
    }
  }
  
  if (latestData.value.history_messages?.length > 0) {
    lines.push(`【历史消息】(${latestData.value.history_messages.length} 条)`)
    lines.push('───────────────────────────────────────────────────────')
    for (const msg of latestData.value.history_messages) {
      const role = msg.role === 'user' ? '用户' : '助手'
      lines.push(`  [${role}] ${msg.content}`)
    }
    lines.push('')
  }
  
  lines.push(`【最终输入预览】`)
  lines.push('───────────────────────────────────────────────────────')
  lines.push(latestData.value.final_input_preview || latestData.value.original_query || '(无)')
  lines.push('')
  lines.push('═══════════════════════════════════════════════════════')
  
  const text = lines.join('\n')
  navigator.clipboard.writeText(text).then(() => {
    ElMessage.success('已复制到剪贴板')
  }).catch(() => {
    ElMessage.error('复制失败')
  })
}

// Lifecycle
onMounted(() => {
  refreshData()
})
</script>

<style lang="scss" scoped>
.injection-viz-view {
  height: 100%;
  overflow-y: auto;
  padding: 24px;
}

.empty-notice {
  margin-bottom: 16px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;

  .header-content {
    h1 {
      font-size: 24px;
      font-weight: 700;
      color: var(--text-primary);
      margin: 0 0 8px;
    }

    p {
      font-size: 14px;
      color: var(--text-secondary);
      margin: 0;
    }
  }
}

.viz-content {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

section {
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 20px;

  h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
}

// Flow Diagram
.flow-diagram {
  overflow-x: auto;
  padding: 20px 0;
}

.flow-nodes {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  min-width: 900px;
}

.flow-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px;
  border-radius: 12px;
  min-width: 120px;
  text-align: center;

  .node-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin-bottom: 8px;
  }

  .node-label {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
  }

  .node-detail {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .node-tokens {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
    padding: 2px 8px;
    background: var(--bg-hover);
    border-radius: 10px;
  }

  &.input-node {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    .node-icon { background: rgba(255,255,255,0.2); color: white; }
    .node-label, .node-detail { color: white; }
  }

  &.process-node {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    .node-icon { background: var(--bg-surface); color: var(--text-secondary); }
  }

  &.injection-node {
    &.preference-node {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      .node-icon { background: rgba(255,255,255,0.2); color: white; }
      .node-label, .node-detail, .node-tokens { color: white; }
      .node-tokens { background: rgba(255,255,255,0.2); }
    }

    &.history-node {
      background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
      .node-icon { background: rgba(255,255,255,0.2); color: white; }
      .node-label, .node-detail, .node-tokens { color: white; }
      .node-tokens { background: rgba(255,255,255,0.2); }
    }
  }

  &.output-node {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
    .node-icon { background: rgba(255,255,255,0.2); color: white; }
    .node-label, .node-detail { color: white; }
  }
}

.flow-arrow {
  font-size: 24px;
  color: var(--text-muted);
}

.flow-arrow-split, .flow-arrow-merge {
  display: flex;
  flex-direction: column;
  gap: 40px;
  font-size: 20px;
  color: var(--text-muted);
}

.injection-nodes {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

// Token Section
.token-charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  align-items: center;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.chart-card {
  background: var(--bg-hover);
  border-radius: 8px;
  padding: 16px;
}

.token-details {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.token-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;

  .token-color {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    margin-top: 2px;

    &.query { background: #3b82f6; }
    &.preference { background: #10b981; }
    &.history { background: #f59e0b; }
  }

  .token-info {
    display: flex;
    flex-direction: column;

    .token-label {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .token-value {
      font-size: 12px;
      color: var(--text-secondary);
    }
  }

  .token-desc {
    flex: 1;
    font-size: 12px;
    color: var(--text-muted);
    text-align: right;
  }
}

// Layers Section
.layers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
}

.layer-card {
  border-radius: 12px;
  overflow: hidden;

  .layer-header {
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 12px;

    .layer-badge {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      font-weight: 700;
      color: white;
    }

    .layer-name {
      font-size: 16px;
      font-weight: 600;
      color: white;
    }
  }

  .layer-body {
    padding: 16px;
    background: var(--bg-hover);

    .layer-method {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
    }

    .layer-stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 12px;

      .stat-item {
        text-align: center;

        .stat-label {
          font-size: 12px;
          color: var(--text-muted);
          display: block;
        }

        .stat-value {
          font-size: 16px;
          font-weight: 600;
          color: var(--text-primary);
        }
      }
    }

    .layer-desc {
      font-size: 13px;
      color: var(--text-secondary);
      line-height: 1.5;

      strong {
        color: var(--text-primary);
      }
    }
  }

  &.l1 .layer-header {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    .layer-badge { background: rgba(255,255,255,0.2); }
  }

  &.l2 .layer-header {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    .layer-badge { background: rgba(255,255,255,0.2); }
  }

  &.l3 .layer-header {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    .layer-badge { background: rgba(255,255,255,0.2); }
  }
}

// Steps Section
.steps-timeline {
  padding: 0 16px;
}

.step-card {
  .step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;

    .step-name {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary);
    }
  }

  .step-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .step-details {
    margin-top: 8px;
  }
}

// Injection Text Section
.injection-text-section {
  h2 {
    display: flex;
    align-items: center;
  }
}

.injection-text-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  
  &.expanded {
    grid-template-columns: 1fr;
  }
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
}

.injection-text-card {
  background: var(--bg-hover);
  border-radius: 8px;
  overflow: hidden;
  
  .card-header {
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    
    .card-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      font-weight: 600;
      color: white;
    }
  }
  
  .card-body {
    padding: 16px;
    
    .text-label {
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    
    .text-meta {
      display: flex;
      gap: 16px;
      margin-top: 8px;
      font-size: 12px;
      color: var(--text-secondary);
    }
    
    :deep(.el-textarea__inner) {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      font-family: 'Fira Code', 'Consolas', monospace;
      font-size: 12px;
      line-height: 1.5;
    }
  }
  
  &.preference-card .card-header {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  }
  
  &.history-card .card-header {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  }
  
  &.messages-card .card-header {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  }
  
  &.rag-context-card .card-header {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  }
  
  &.rag-prompt-card .card-header {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
  }
  
  &.final-card .card-header {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  }
}

.messages-list {
  max-height: 200px;
  overflow-y: auto;
  
  .message-item {
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 13px;
    
    &.message-user {
      background: var(--bg-surface);
    }
    
    &.message-assistant {
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .message-role {
      font-weight: 600;
      color: var(--text-secondary);
      margin-right: 8px;
    }
    
    .message-content {
      color: var(--text-primary);
    }
  }
}

// History Section
.history-section {
  :deep(.el-table) {
    background: transparent;
  }
}

// Detail Dialog
.detail-content {
  h4 {
    margin: 16px 0 8px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);

    &:first-child {
      margin-top: 16px;
    }
  }
}
</style>
