<template>
  <div class="injection-viz-view">
    <!-- Header -->
    <header class="page-header">
      <div class="header-content">
        <h1>Injection Visualization</h1>
        <p>Detailed visualization analysis of DKI injection process</p>
      </div>
      <div class="header-actions">
        <el-button :icon="Refresh" @click="refreshData" :loading="loading">
          Refresh
        </el-button>
      </div>
    </header>

    <!-- Empty State Notice -->
    <div v-if="!hasRealData && !loading" class="empty-notice">
      <el-alert
        type="info"
        title="No Injection Data"
        description="Please send a message via the chat interface first. The system will automatically record each injection process. The flow diagram below shows the standard DKI injection flow."
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
          Injection Flow Diagram
        </h2>
        <div class="flow-diagram">
          <div class="flow-nodes">
            <!-- Input Node -->
            <div class="flow-node input-node">
              <div class="node-icon">
                <el-icon><Edit /></el-icon>
              </div>
              <div class="node-label">User Input</div>
              <div class="node-detail">Original Query</div>
            </div>

            <div class="flow-arrow">→</div>

            <!-- Adapter Node -->
            <div class="flow-node process-node">
              <div class="node-icon">
                <el-icon><DataLine /></el-icon>
              </div>
              <div class="node-label">External Data Adapter</div>
              <div class="node-detail">Load Preferences + History</div>
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
                <div class="node-label">Preference K/V Injection</div>
                <div class="node-detail">Negative Position (α={{ latestData?.injection_layers?.[0]?.alpha?.toFixed(2) || '0.40' }})</div>
                <div class="node-tokens" v-if="latestData?.token_distribution?.preference">
                  {{ latestData.token_distribution.preference }} tokens
                </div>
              </div>

              <div class="flow-node injection-node history-node">
                <div class="node-icon">
                  <el-icon><ChatDotRound /></el-icon>
                </div>
                <div class="node-label">History Suffix Injection</div>
                <div class="node-detail">Positive Position (Explicit)</div>
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
              <div class="node-label">LLM Inference</div>
              <div class="node-detail">Attention Computation with Injection</div>
            </div>

            <div class="flow-arrow">→</div>

            <!-- Output Node -->
            <div class="flow-node output-node">
              <div class="node-icon">
                <el-icon><ChatLineSquare /></el-icon>
              </div>
              <div class="node-label">Output Response</div>
              <div class="node-detail">Personalized Reply</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Token Distribution Section -->
      <section class="token-section">
        <h2>
          <el-icon><PieChart /></el-icon>
          Token Distribution
        </h2>
        <div class="token-charts">
          <div class="chart-card">
            <v-chart :option="tokenPieOption" autoresize style="height: 280px" />
          </div>
          <div class="token-details">
            <div class="token-item">
              <div class="token-color query"></div>
              <div class="token-info">
                <span class="token-label">Query</span>
                <span class="token-value">{{ latestData?.token_distribution?.query || 0 }} tokens</span>
              </div>
              <div class="token-desc">User original input, positive position</div>
            </div>
            <div class="token-item">
              <div class="token-color preference"></div>
              <div class="token-info">
                <span class="token-label">Preference</span>
                <span class="token-value">{{ latestData?.token_distribution?.preference || 0 }} tokens</span>
              </div>
              <div class="token-desc">K/V injection, negative position, no Context cost</div>
            </div>
            <div class="token-item">
              <div class="token-color history"></div>
              <div class="token-info">
                <span class="token-label">History</span>
                <span class="token-value">{{ latestData?.token_distribution?.history || 0 }} tokens</span>
              </div>
              <div class="token-desc">Suffix injection, positive position, uses Context</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Injection Layers Section -->
      <section class="layers-section">
        <h2>
          <el-icon><Grid /></el-icon>
          Injection Layer Details
        </h2>
        <div class="layers-grid">
          <div class="layer-card l1">
            <div class="layer-header">
              <span class="layer-badge">L1</span>
              <span class="layer-name">Preference Layer</span>
            </div>
            <div class="layer-body">
              <div class="layer-method">
                <el-tag type="success" size="small">K/V Injection</el-tag>
                <el-tag type="info" size="small">Negative Position</el-tag>
              </div>
              <div class="layer-stats">
                <div class="stat-item">
                  <span class="stat-label">Tokens</span>
                  <span class="stat-value">{{ latestData?.injection_layers?.[0]?.token_count || 0 }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Alpha</span>
                  <span class="stat-value">{{ latestData?.injection_layers?.[0]?.alpha?.toFixed(2) || '0.40' }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Cache</span>
                  <span class="stat-value">{{ latestData?.injection_layers?.[0]?.cache_status || 'N/A' }}</span>
                </div>
              </div>
              <div class="layer-desc">
                Stable user preferences injected into attention computation via K/V cache.
                <strong>Does not consume Context window</strong>.
              </div>
            </div>
          </div>

          <div class="layer-card l2">
            <div class="layer-header">
              <span class="layer-badge">L2</span>
              <span class="layer-name">History Layer</span>
            </div>
            <div class="layer-body">
              <div class="layer-method">
                <el-tag type="warning" size="small">Suffix Injection</el-tag>
                <el-tag type="info" size="small">Positive Position</el-tag>
              </div>
              <div class="layer-stats">
                <div class="stat-item">
                  <span class="stat-label">Tokens</span>
                  <span class="stat-value">{{ latestData?.injection_layers?.[1]?.token_count || 0 }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Alpha</span>
                  <span class="stat-value">1.00</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Messages</span>
                  <span class="stat-value">{{ historyMessageCount }}</span>
                </div>
              </div>
              <div class="layer-desc">
                Dynamic conversation history appended as suffix to input.
                <strong>Consumes Context window</strong>, but supports explicit references.
              </div>
            </div>
          </div>

          <div class="layer-card l3">
            <div class="layer-header">
              <span class="layer-badge">L3</span>
              <span class="layer-name">Query Layer</span>
            </div>
            <div class="layer-body">
              <div class="layer-method">
                <el-tag type="primary" size="small">Original Input</el-tag>
                <el-tag type="info" size="small">Positive Position</el-tag>
              </div>
              <div class="layer-stats">
                <div class="stat-item">
                  <span class="stat-label">Tokens</span>
                  <span class="stat-value">{{ latestData?.token_distribution?.query || 0 }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Position</span>
                  <span class="stat-value">0 ~ N</span>
                </div>
              </div>
              <div class="layer-desc">
                User's current query input, serving as the primary inference target.
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Flow Steps Section -->
      <section class="steps-section">
        <h2>
          <el-icon><List /></el-icon>
          Injection Flow Steps
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
          Injection Content Display
          <el-tag v-if="latestData?.mode" :type="latestData?.mode === 'dki' ? 'success' : latestData?.mode === 'rag' ? 'warning' : 'info'" size="small" style="margin-left: 12px;">
            {{ latestData?.mode === 'dki' ? 'DKI Mode' : latestData?.mode === 'rag' ? 'RAG Mode' : 'Baseline Mode' }}
          </el-tag>
          <el-button-group style="margin-left: auto;">
            <el-button size="small" @click="toggleInjectionExpand">
            {{ injectionExpanded ? 'Collapse' : 'Expand' }}
          </el-button>
          <el-button size="small" @click="copyInjectionText">
              <el-icon><CopyDocument /></el-icon>
              Copy
            </el-button>
          </el-button-group>
        </h2>
        
        <div class="injection-text-grid" :class="{ expanded: injectionExpanded }">
          <!-- ==================== DKI Mode ==================== -->
          <template v-if="!latestData?.mode || latestData?.mode === 'dki'">
            <!-- DKI Preference Injection -->
            <div class="injection-text-card preference-card">
              <div class="card-header">
                <span class="card-title">
                  <el-icon><User /></el-icon>
                  DKI Preference Injection (K/V)
                </span>
                <el-tag type="success" size="small">
                  {{ latestData?.token_distribution?.preference || 0 }} tokens
                </el-tag>
              </div>
              <div class="card-body">
                <div class="text-label">Preference plaintext (actual K/V not shown):</div>
                <el-input
                  type="textarea"
                  :value="latestData?.preference_text || '(No preference injection)'"
                  :rows="injectionExpanded ? 8 : 3"
                  readonly
                  resize="none"
                />
                <div class="text-meta">
                  <span>Alpha: {{ latestData?.injection_layers?.[0]?.alpha?.toFixed(2) || '0.00' }}</span>
                  <span>Position: Negative (no Context cost)</span>
                </div>
              </div>
            </div>
            
            <!-- DKI History Suffix -->
            <div class="injection-text-card history-card">
              <div class="card-header">
                <span class="card-title">
                  <el-icon><ChatDotRound /></el-icon>
                  DKI History Suffix
                </span>
                <el-tag type="warning" size="small">
                  {{ latestData?.token_distribution?.history || 0 }} tokens
                </el-tag>
              </div>
              <div class="card-body">
                <div class="text-label">History suffix plaintext:</div>
                <el-input
                  type="textarea"
                  :value="latestData?.history_suffix_text || '(No history suffix)'"
                  :rows="injectionExpanded ? 8 : 3"
                  readonly
                  resize="none"
                />
                <div class="text-meta">
                  <span>Messages: {{ latestData?.history_messages?.length || 0 }}</span>
                  <span>Position: Positive (uses Context)</span>
                </div>
              </div>
            </div>
          </template>
          
          <!-- ==================== RAG Mode ==================== -->
          <template v-if="latestData?.mode === 'rag'">
            <!-- RAG Retrieved Context -->
            <div class="injection-text-card rag-context-card">
              <div class="card-header">
                <span class="card-title">
                  <el-icon><DataLine /></el-icon>
                  RAG Retrieved Context
                </span>
                <el-tag type="warning" size="small">Context</el-tag>
              </div>
              <div class="card-body">
                <div class="text-label">Retrieved context:</div>
                <el-input
                  type="textarea"
                  :value="latestData?.rag_context_text || '(No retrieved context)'"
                  :rows="injectionExpanded ? 8 : 3"
                  readonly
                  resize="none"
                />
              </div>
            </div>
            
            <!-- RAG Full Prompt -->
            <div class="injection-text-card rag-prompt-card">
              <div class="card-header">
                <span class="card-title">
                  <el-icon><Document /></el-icon>
                  RAG Full Prompt
                </span>
                <el-tag type="warning" size="small">Prompt</el-tag>
              </div>
              <div class="card-body">
                <div class="text-label">Full prompt sent to LLM:</div>
                <el-input
                  type="textarea"
                  :value="latestData?.rag_prompt_text || '(No full prompt)'"
                  :rows="injectionExpanded ? 12 : 5"
                  readonly
                  resize="none"
                />
              </div>
            </div>
          </template>
          
          <!-- ==================== Common: History Message List ==================== -->
          <div class="injection-text-card messages-card" v-if="latestData?.history_messages?.length > 0">
            <div class="card-header">
              <span class="card-title">
                <el-icon><List /></el-icon>
                History Message Details
              </span>
              <el-tag type="info" size="small">
                {{ latestData?.history_messages?.length || 0 }} items
              </el-tag>
            </div>
            <div class="card-body messages-list">
              <div
                v-for="(msg, idx) in latestData?.history_messages || []"
                :key="idx"
                class="message-item"
                :class="[`message-${msg.role}`]"
              >
                <span class="message-role">{{ msg.role === 'user' ? 'User' : 'Assistant' }}:</span>
                <span class="message-content">{{ msg.content }}</span>
              </div>
              <el-empty v-if="!latestData?.history_messages?.length" description="No history messages" />
            </div>
          </div>
          
          <!-- ==================== Common: Final Input Preview ==================== -->
          <div class="injection-text-card final-card">
            <div class="card-header">
              <span class="card-title">
                <el-icon><View /></el-icon>
                Final Input Preview
              </span>
              <el-tag type="primary" size="small">
                {{ latestData?.token_distribution?.total || 0 }} tokens
              </el-tag>
            </div>
            <div class="card-body">
              <el-input
                type="textarea"
                :value="latestData?.final_input_preview || latestData?.original_query || '(No data)'"
                :rows="injectionExpanded ? 10 : 4"
                readonly
                resize="none"
              />
            </div>
          </div>
        </div>
      </section>

      <!-- Recall v4 Information Section -->
      <section class="recall-section" v-if="latestData?.recall_v4_enabled">
        <h2>
          <el-icon><DataLine /></el-icon>
          Recall v4 Information
          <el-tag type="success" size="small" style="margin-left: 12px;">
            {{ latestData?.recall_strategy || 'summary_with_fact_call' }}
          </el-tag>
        </h2>
        <div class="recall-info-grid">
          <div class="recall-stat">
            <div class="recall-stat-label">Recall Strategy</div>
            <div class="recall-stat-value">{{ latestData?.recall_strategy || 'N/A' }}</div>
          </div>
          <div class="recall-stat">
            <div class="recall-stat-label">Fact Call Rounds</div>
            <div class="recall-stat-value">{{ latestData?.recall_fact_rounds || 0 }}</div>
          </div>
          <div class="recall-stat">
            <div class="recall-stat-label">Summary Entries</div>
            <div class="recall-stat-value">{{ latestData?.recall_summary_count || 0 }}</div>
          </div>
          <div class="recall-stat">
            <div class="recall-stat-label">Original Messages</div>
            <div class="recall-stat-value">{{ latestData?.recall_message_count || 0 }}</div>
          </div>
          <div class="recall-stat" v-if="latestData?.recall_trace_ids?.length > 0">
            <div class="recall-stat-label">Trace IDs</div>
            <div class="recall-stat-value trace-ids">
              <el-tag
                v-for="tid in latestData.recall_trace_ids.slice(0, 10)"
                :key="tid"
                size="small"
                type="info"
                style="margin: 2px;"
              >
                {{ tid }}
              </el-tag>
              <span v-if="latestData.recall_trace_ids.length > 10" class="more-ids">
                +{{ latestData.recall_trace_ids.length - 10 }} more
              </span>
            </div>
          </div>
        </div>
      </section>

      <!-- Function Call Logs Section (v3.2) -->
      <section class="fc-logs-section" v-if="functionCallLogs.length > 0 || latestData?.session_id">
        <h2>
          <el-icon><DataLine /></el-icon>
          Function Call Logs
          <el-tag type="danger" size="small" style="margin-left: 12px;" v-if="functionCallLogs.length > 0">
            {{ functionCallLogs.length }} calls
          </el-tag>
          <el-button size="small" style="margin-left: auto;" @click="loadFunctionCalls" :loading="fcLoading">
            <el-icon><Refresh /></el-icon>
            Load
          </el-button>
        </h2>

        <div v-if="functionCallLogs.length === 0 && !fcLoading" class="fc-empty">
          <el-empty description="No Function Call records. Click 'Load' to fetch Function Call logs for the current session." :image-size="60" />
        </div>

        <div v-else class="fc-logs-list">
          <div
            v-for="(fc, idx) in functionCallLogs"
            :key="idx"
            class="fc-log-card"
            :class="[`fc-status-${fc.status || 'success'}`]"
          >
            <div class="fc-log-header">
              <div class="fc-round">
                <el-tag :type="fc.status === 'success' ? 'success' : fc.status === 'error' ? 'danger' : 'warning'" size="small">
                  #{{ fc.round_index !== undefined ? fc.round_index + 1 : idx + 1 }}
                </el-tag>
              </div>
              <div class="fc-name">{{ fc.function_name || 'retrieve_fact' }}</div>
              <div class="fc-status">
                <el-tag :type="fc.status === 'success' ? 'success' : fc.status === 'error' ? 'danger' : 'warning'" size="small" effect="plain">
                  {{ fc.status || 'success' }}
                </el-tag>
              </div>
              <div class="fc-latency">{{ (fc.latency_ms || 0).toFixed(1) }}ms</div>
              <el-button size="small" type="primary" link @click="toggleFCDetail(idx)">
                {{ expandedFC === idx ? 'Collapse' : 'Expand' }}
              </el-button>
            </div>

            <!-- Arguments -->
            <div class="fc-log-args">
              <span class="fc-label">Args:</span>
              <code>{{ JSON.stringify(fc.arguments || {}) }}</code>
            </div>

            <!-- Response preview -->
            <div class="fc-log-response" v-if="fc.response_text">
              <span class="fc-label">Response ({{ fc.response_tokens || 0 }} tokens):</span>
              <div class="fc-response-preview">
                {{ expandedFC === idx ? fc.response_text : truncateText(fc.response_text, 200) }}
              </div>
            </div>

            <!-- Error message -->
            <div class="fc-log-error" v-if="fc.error_message">
              <span class="fc-label">Error:</span>
              <span class="fc-error-text">{{ fc.error_message }}</span>
            </div>

            <!-- Expanded detail: prompt before/after -->
            <div class="fc-log-detail" v-if="expandedFC === idx">
              <div class="fc-detail-section" v-if="fc.model_output_before">
                <div class="fc-detail-title">Model output that triggered FC:</div>
                <el-input type="textarea" :value="fc.model_output_before" :rows="3" readonly resize="none" />
              </div>
              <div class="fc-detail-section" v-if="fc.prompt_before">
                <div class="fc-detail-title">Prompt Before Call:</div>
                <el-input type="textarea" :value="fc.prompt_before" :rows="4" readonly resize="none" />
              </div>
              <div class="fc-detail-section" v-if="fc.prompt_after">
                <div class="fc-detail-title">Prompt After Call:</div>
                <el-input type="textarea" :value="fc.prompt_after" :rows="4" readonly resize="none" />
              </div>
            </div>
          </div>
        </div>

        <!-- FC Stats -->
        <div class="fc-stats" v-if="fcStats">
          <el-descriptions :column="4" size="small" border>
            <el-descriptions-item label="Total Calls">{{ fcStats.total || 0 }}</el-descriptions-item>
            <el-descriptions-item label="Success">{{ fcStats.success || 0 }}</el-descriptions-item>
            <el-descriptions-item label="Errors">{{ fcStats.error || 0 }}</el-descriptions-item>
            <el-descriptions-item label="Budget Exceeded">{{ fcStats.budget_exceeded || 0 }}</el-descriptions-item>
          </el-descriptions>
        </div>
      </section>

      <!-- History Section -->
      <section class="history-section">
        <h2>
          <el-icon><Clock /></el-icon>
          Injection History
        </h2>
        <el-table :data="historyItems" stripe style="width: 100%">
          <el-table-column prop="timestamp" label="Time" width="180">
            <template #default="{ row }">
              {{ formatTimestamp(row.timestamp) }}
            </template>
          </el-table-column>
          <el-table-column prop="query_preview" label="Query" min-width="200" />
          <el-table-column prop="injection_enabled" label="Injected" width="80">
            <template #default="{ row }">
              <el-tag :type="row.injection_enabled ? 'success' : 'info'" size="small">
                {{ row.injection_enabled ? 'Yes' : 'No' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="alpha" label="Alpha" width="80">
            <template #default="{ row }">
              {{ row.alpha.toFixed(2) }}
            </template>
          </el-table-column>
          <el-table-column prop="preference_tokens" label="Pref Tokens" width="100" />
          <el-table-column prop="history_tokens" label="Hist Tokens" width="100" />
          <el-table-column prop="latency_ms" label="Latency" width="100">
            <template #default="{ row }">
              {{ row.latency_ms.toFixed(1) }}ms
            </template>
          </el-table-column>
          <el-table-column label="Action" width="80">
            <template #default="{ row }">
              <el-button type="primary" link size="small" @click="viewDetail(row.request_id)">
                Detail
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </section>
    </div>

    <!-- Detail Dialog -->
    <el-dialog
      v-model="showDetailDialog"
      title="Injection Detail"
      width="800px"
    >
      <div v-if="detailData" class="detail-content">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="Request ID">{{ detailData.request_id }}</el-descriptions-item>
          <el-descriptions-item label="Timestamp">{{ detailData.timestamp }}</el-descriptions-item>
          <el-descriptions-item label="User ID">{{ detailData.user_id }}</el-descriptions-item>
          <el-descriptions-item label="Session ID">{{ detailData.session_id }}</el-descriptions-item>
          <el-descriptions-item label="Total Latency">{{ detailData.total_latency_ms?.toFixed(1) }}ms</el-descriptions-item>
          <el-descriptions-item label="Injection Overhead">{{ detailData.injection_overhead_ms?.toFixed(1) }}ms</el-descriptions-item>
        </el-descriptions>

        <h4>Original Query</h4>
        <el-input
          type="textarea"
          :value="detailData.original_query"
          :rows="3"
          readonly
        />

        <h4>Final Input Preview</h4>
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

// Function Call Logs state (v3.2)
const functionCallLogs = ref<any[]>([])
const fcLoading = ref(false)
const fcStats = ref<any>(null)
const expandedFC = ref<number | null>(null)

// Computed
const flowSteps = computed(() => latestData.value?.flow_steps || defaultFlowSteps)

const historyMessageCount = computed(() => {
  const step = flowSteps.value.find((s: any) => s.step_name === 'History Suffix Injection')
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
          name: 'Query',
          itemStyle: { color: '#3b82f6' },
        },
        {
          value: latestData.value?.token_distribution?.preference ?? 0,
          name: 'Preference (K/V)',
          itemStyle: { color: '#10b981' },
        },
        {
          value: latestData.value?.token_distribution?.history ?? 0,
          name: 'History (Suffix)',
          itemStyle: { color: '#f59e0b' },
        },
      ],
    },
  ],
}))

// Default flow steps for demo
const defaultFlowSteps = [
  { step_id: 1, step_name: 'Receive Input', description: 'Receive user raw query and identifiers', status: 'completed', duration_ms: 0.1, details: {} },
  { step_id: 2, step_name: 'Memory Trigger Detection', description: 'Detect whether to trigger memory storage/update', status: 'skipped', duration_ms: 0.5, details: {} },
  { step_id: 3, step_name: 'Reference Resolver', description: 'Resolve anaphoric expressions and determine recall scope', status: 'skipped', duration_ms: 0.5, details: {} },
  { step_id: 4, step_name: 'Load External Data', description: 'Read user preferences and history messages via adapter', status: 'completed', duration_ms: 5.2, details: { preferences_count: 3, history_count: 8 } },
  { step_id: 5, step_name: 'Preference K/V Injection', description: 'Encode user preferences as K/V and inject at negative positions', status: 'completed', duration_ms: 2.1, details: { tokens: 45, cache_hit: true, alpha: 0.4 } },
  { step_id: 6, step_name: 'History Suffix Injection', description: 'Format relevant history as suffix and append to input', status: 'completed', duration_ms: 1.3, details: { tokens: 120, messages_count: 5 } },
  { step_id: 7, step_name: 'LLM Inference', description: 'Call LLM for inference generation', status: 'completed', duration_ms: 156.8, details: { input_tokens: 175 } },
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
    case 'completed': return 'Done'
    case 'running': return 'Running'
    case 'skipped': return 'Skipped'
    case 'pending': return 'Pending'
    default: return status
  }
}

function formatDetailKey(key: string | number) {
  const k = String(key)
  const keyMap: Record<string, string> = {
    query_length: 'Query Length',
    triggered: 'Triggered',
    trigger_type: 'Trigger Type',
    resolved: 'Resolved',
    reference_type: 'Reference Type',
    preferences_count: 'Preferences Count',
    history_count: 'History Count',
    tokens: 'Token Count',
    cache_hit: 'Cache Hit',
    alpha: 'Alpha',
    messages_count: 'Messages Count',
    input_tokens: 'Input Tokens',
  }
  return keyMap[k] || k
}

function formatDetailValue(value: any) {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
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
    ElMessage.warning('No data to copy')
    return
  }
  
  const mode = latestData.value.mode || 'dki'
  const lines = []
  lines.push('═══════════════════════════════════════════════════════')
  lines.push(`  ${mode === 'dki' ? 'DKI Injection Info' : mode === 'rag' ? 'RAG Prompt Info' : 'Baseline Mode Info'}`)
  lines.push('═══════════════════════════════════════════════════════')
  lines.push('')
  lines.push(`[Original Query]`)
  lines.push(latestData.value.original_query || '(None)')
  lines.push('')
  
  if (mode === 'dki') {
    if (latestData.value.preference_text) {
      lines.push(`[Preference Injection] (K/V Injection, α=${latestData.value.injection_layers?.[0]?.alpha?.toFixed(2) || '0.00'})`)
      lines.push('───────────────────────────────────────────────────────')
      lines.push(latestData.value.preference_text)
      lines.push('')
    }
    
    if (latestData.value.history_suffix_text) {
      lines.push(`[History Suffix] (${latestData.value.token_distribution?.history || 0} tokens)`)
      lines.push('───────────────────────────────────────────────────────')
      lines.push(latestData.value.history_suffix_text)
      lines.push('')
    }
  } else if (mode === 'rag') {
    if (latestData.value.rag_context_text) {
      lines.push(`[RAG Retrieved Context]`)
      lines.push('───────────────────────────────────────────────────────')
      lines.push(latestData.value.rag_context_text)
      lines.push('')
    }
    
    if (latestData.value.rag_prompt_text) {
      lines.push(`[RAG Full Prompt]`)
      lines.push('───────────────────────────────────────────────────────')
      lines.push(latestData.value.rag_prompt_text)
      lines.push('')
    }
  }
  
  if (latestData.value.history_messages?.length > 0) {
    lines.push(`[History Messages] (${latestData.value.history_messages.length} items)`)
    lines.push('───────────────────────────────────────────────────────')
    for (const msg of latestData.value.history_messages) {
      const role = msg.role === 'user' ? 'User' : 'Assistant'
      lines.push(`  [${role}] ${msg.content}`)
    }
    lines.push('')
  }
  
  lines.push(`[Final Input Preview]`)
  lines.push('───────────────────────────────────────────────────────')
  lines.push(latestData.value.final_input_preview || latestData.value.original_query || '(None)')
  lines.push('')
  lines.push('═══════════════════════════════════════════════════════')
  
  const text = lines.join('\n')
  navigator.clipboard.writeText(text).then(() => {
    ElMessage.success('Copied to clipboard')
  }).catch(() => {
    ElMessage.error('Copy failed')
  })
}

// ===== Function Call Logs Methods (v3.2) =====
async function loadFunctionCalls() {
  const sessionId = latestData.value?.session_id
  if (!sessionId) {
    ElMessage.warning('No session data available. Please send a message first.')
    return
  }
  fcLoading.value = true
  try {
    // Fetch function call logs and stats in parallel
    const [fcRes, statsRes] = await Promise.all([
      api.visualization.getFunctionCalls(sessionId, true).catch(() => null),
      api.visualization.getFunctionCallStats(sessionId).catch(() => null),
    ])
    functionCallLogs.value = fcRes?.function_calls || []
    fcStats.value = statsRes?.stats || null
    if (functionCallLogs.value.length === 0) {
      ElMessage.info('No function call records for this session')
    }
  } catch (e: any) {
    console.error('Failed to load function call logs:', e)
    ElMessage.error('Failed to load function call logs')
  } finally {
    fcLoading.value = false
  }
}

function toggleFCDetail(idx: number) {
  expandedFC.value = expandedFC.value === idx ? null : idx
}

function truncateText(text: string, maxLen: number): string {
  if (!text) return ''
  if (text.length <= maxLen) return text
  return text.substring(0, maxLen) + '...'
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

// ===== Recall v4 Section =====
.recall-section {
  h2 {
    display: flex;
    align-items: center;
  }
}

.recall-info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
}

.recall-stat {
  background: var(--bg-hover);
  border-radius: 8px;
  padding: 16px;
  text-align: center;

  .recall-stat-label {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .recall-stat-value {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);

    &.trace-ids {
      font-size: 12px;
      font-weight: 400;
      text-align: left;

      .more-ids {
        font-size: 11px;
        color: var(--text-muted);
        margin-left: 4px;
      }
    }
  }
}

// ===== Function Call Logs Section (v3.2) =====
.fc-logs-section {
  h2 {
    display: flex;
    align-items: center;
  }
}

.fc-empty {
  padding: 20px 0;
}

.fc-logs-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.fc-log-card {
  background: var(--bg-hover);
  border-radius: 8px;
  padding: 12px 16px;
  border-left: 4px solid #10b981;

  &.fc-status-error {
    border-left-color: #ef4444;
  }

  &.fc-status-timeout {
    border-left-color: #f59e0b;
  }

  &.fc-status-budget_exceeded {
    border-left-color: #f97316;
  }
}

.fc-log-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;

  .fc-round {
    flex-shrink: 0;
  }

  .fc-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    flex: 1;
  }

  .fc-status {
    flex-shrink: 0;
  }

  .fc-latency {
    font-size: 12px;
    color: var(--text-muted);
    flex-shrink: 0;
  }
}

.fc-log-args {
  margin-bottom: 6px;
  font-size: 12px;

  .fc-label {
    color: var(--text-muted);
    margin-right: 6px;
  }

  code {
    background: var(--bg-surface);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-secondary);
    word-break: break-all;
  }
}

.fc-log-response {
  margin-bottom: 6px;
  font-size: 12px;

  .fc-label {
    color: var(--text-muted);
    margin-right: 6px;
  }

  .fc-response-preview {
    background: var(--bg-surface);
    padding: 8px 12px;
    border-radius: 6px;
    margin-top: 4px;
    font-size: 12px;
    line-height: 1.5;
    color: var(--text-primary);
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'Fira Code', 'Consolas', monospace;
  }
}

.fc-log-error {
  margin-bottom: 6px;
  font-size: 12px;

  .fc-label {
    color: var(--text-muted);
    margin-right: 6px;
  }

  .fc-error-text {
    color: #ef4444;
    font-weight: 500;
  }
}

.fc-log-detail {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  gap: 12px;

  .fc-detail-section {
    .fc-detail-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-secondary);
      margin-bottom: 6px;
    }

    :deep(.el-textarea__inner) {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      font-family: 'Fira Code', 'Consolas', monospace;
      font-size: 11px;
      line-height: 1.5;
    }
  }
}

.fc-stats {
  margin-top: 16px;
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
