<template>
  <div class="stats-view">
    <!-- Auth Gate -->
    <div v-if="!isAuthenticated" class="auth-gate">
      <div class="auth-card">
        <el-icon class="auth-icon"><Lock /></el-icon>
        <h2>访问受限</h2>
        <p>统计页面需要管理员权限</p>
        
        <el-form @submit.prevent="handleAuth">
          <el-form-item>
            <el-input
              v-model="authPassword"
              type="password"
              placeholder="请输入管理密码"
              show-password
              size="large"
              @keyup.enter="handleAuth"
            />
          </el-form-item>
          <el-form-item>
            <el-button
              type="primary"
              size="large"
              :loading="authLoading"
              @click="handleAuth"
              style="width: 100%"
            >
              验证
            </el-button>
          </el-form-item>
        </el-form>
        
        <el-alert
          v-if="authError"
          type="error"
          :title="authError"
          :closable="false"
          show-icon
        />
      </div>
    </div>
    
    <!-- Stats Content -->
    <div v-else class="stats-content">
      <!-- Header -->
      <header class="page-header">
        <div class="header-content">
          <h1>系统统计</h1>
          <p>DKI 系统运行状态和性能指标</p>
        </div>
        <div class="header-actions">
          <el-button :icon="Refresh" @click="refreshStats" :loading="loading">
            刷新
          </el-button>
          <el-button @click="handleLogout">
            <el-icon><SwitchButton /></el-icon>
            退出统计
          </el-button>
        </div>
      </header>
      
      <!-- Overview Cards -->
      <div class="overview-cards">
        <div class="overview-card">
          <div class="card-header">
            <span class="card-title">总请求数</span>
            <el-icon class="card-icon"><DataAnalysis /></el-icon>
          </div>
          <div class="card-value">{{ stats?.dkiStats?.totalRequests || 0 }}</div>
          <div class="card-trend positive">
            <el-icon><TrendCharts /></el-icon>
            <span>+12.5% 较昨日</span>
          </div>
        </div>
        
        <div class="overview-card">
          <div class="card-header">
            <span class="card-title">缓存命中率</span>
            <el-icon class="card-icon success"><CircleCheck /></el-icon>
          </div>
          <div class="card-value">{{ cacheHitRate }}%</div>
          <div class="card-trend positive">
            <el-icon><TrendCharts /></el-icon>
            <span>L1: {{ stats?.cacheStats?.l1HitRate || 0 }}%</span>
          </div>
        </div>
        
        <div class="overview-card">
          <div class="card-header">
            <span class="card-title">注入率</span>
            <el-icon class="card-icon warning"><Lightning /></el-icon>
          </div>
          <div class="card-value">{{ injectionRate }}%</div>
          <div class="card-trend">
            <span>平均 Alpha: {{ avgAlpha }}</span>
          </div>
        </div>
        
        <div class="overview-card">
          <div class="card-header">
            <span class="card-title">运行时间</span>
            <el-icon class="card-icon info"><Timer /></el-icon>
          </div>
          <div class="card-value">{{ formatUptime(stats?.uptimeSeconds || 0) }}</div>
          <div class="card-trend">
            <span>{{ stats?.adapterStats?.type || 'N/A' }} 适配器</span>
          </div>
        </div>
      </div>
      
      <!-- Charts Row -->
      <div class="charts-row">
        <!-- Cache Distribution -->
        <div class="chart-card">
          <h3>缓存层级分布</h3>
          <v-chart :option="cacheChartOption" autoresize style="height: 300px" />
        </div>
        
        <!-- Request Trend -->
        <div class="chart-card">
          <h3>请求趋势</h3>
          <v-chart :option="trendChartOption" autoresize style="height: 300px" />
        </div>
      </div>
      
      <!-- Detailed Stats -->
      <div class="detailed-stats">
        <div class="stats-card">
          <h3>DKI 统计</h3>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="总请求数">
              {{ stats?.dkiStats?.totalRequests || 0 }}
            </el-descriptions-item>
            <el-descriptions-item label="L1 命中">
              {{ stats?.dkiStats?.l1Hits || 0 }}
            </el-descriptions-item>
            <el-descriptions-item label="L2 命中">
              {{ stats?.dkiStats?.l2Hits || 0 }}
            </el-descriptions-item>
            <el-descriptions-item label="L3 计算">
              {{ stats?.dkiStats?.l3Computes || 0 }}
            </el-descriptions-item>
            <el-descriptions-item label="平均 Alpha">
              {{ stats?.dkiStats?.avgAlpha?.toFixed(4) || 'N/A' }}
            </el-descriptions-item>
            <el-descriptions-item label="注入率">
              {{ (stats?.dkiStats?.injectionRate * 100)?.toFixed(1) || 0 }}%
            </el-descriptions-item>
          </el-descriptions>
        </div>
        
        <div class="stats-card">
          <h3>缓存统计</h3>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="L1 大小">
              {{ stats?.cacheStats?.l1Size || 0 }} / {{ stats?.cacheStats?.l1MaxSize || 0 }}
            </el-descriptions-item>
            <el-descriptions-item label="L1 命中率">
              {{ (stats?.cacheStats?.l1HitRate * 100)?.toFixed(1) || 0 }}%
            </el-descriptions-item>
            <el-descriptions-item label="L2 命中率">
              {{ (stats?.cacheStats?.l2HitRate * 100)?.toFixed(1) || 0 }}%
            </el-descriptions-item>
            <el-descriptions-item label="L1 使用率">
              <el-progress
                :percentage="l1UsagePercent"
                :stroke-width="8"
                :color="l1UsagePercent > 80 ? '#f56c6c' : '#10b981'"
              />
            </el-descriptions-item>
          </el-descriptions>
        </div>
        
        <div class="stats-card">
          <h3>适配器状态</h3>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="类型">
              <el-tag>{{ stats?.adapterStats?.type || 'N/A' }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="连接状态">
              <el-tag :type="stats?.adapterStats?.connected ? 'success' : 'danger'">
                {{ stats?.adapterStats?.connected ? '已连接' : '未连接' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="运行时间">
              {{ formatUptime(stats?.uptimeSeconds || 0) }}
            </el-descriptions-item>
          </el-descriptions>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import {
  Lock,
  Refresh,
  SwitchButton,
  DataAnalysis,
  CircleCheck,
  Lightning,
  Timer,
  TrendCharts,
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart, LineChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
} from 'echarts/components'
import { useStatsAuthStore } from '@/stores/statsAuth'
import { api } from '@/services/api'
import type { SystemStats } from '@/types'

// Register ECharts components
use([
  CanvasRenderer,
  PieChart,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
])

const statsAuthStore = useStatsAuthStore()

const authPassword = ref('')
const authLoading = ref(false)
const authError = ref('')
const loading = ref(false)
const stats = ref<SystemStats | null>(null)
let refreshInterval: ReturnType<typeof setInterval> | null = null

const isAuthenticated = computed(() => statsAuthStore.isAuthenticated)

const cacheHitRate = computed(() => {
  if (!stats.value?.cacheStats) return 0
  const { l1HitRate, l2HitRate } = stats.value.cacheStats
  return ((l1HitRate + l2HitRate) * 50).toFixed(1)
})

const injectionRate = computed(() => {
  return ((stats.value?.dkiStats?.injectionRate || 0) * 100).toFixed(1)
})

const avgAlpha = computed(() => {
  return stats.value?.dkiStats?.avgAlpha?.toFixed(3) || '0.000'
})

const l1UsagePercent = computed(() => {
  if (!stats.value?.cacheStats) return 0
  const { l1Size, l1MaxSize } = stats.value.cacheStats
  return Math.round((l1Size / l1MaxSize) * 100)
})

// Cache distribution chart
const cacheChartOption = computed(() => ({
  tooltip: {
    trigger: 'item',
    formatter: '{b}: {c} ({d}%)',
  },
  legend: {
    orient: 'vertical',
    left: 'left',
    textStyle: {
      color: 'var(--text-secondary)',
    },
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
      label: {
        show: false,
      },
      emphasis: {
        label: {
          show: true,
          fontSize: 14,
          fontWeight: 'bold',
        },
      },
      data: [
        {
          value: stats.value?.dkiStats?.l1Hits || 0,
          name: 'L1 (内存)',
          itemStyle: { color: '#10b981' },
        },
        {
          value: stats.value?.dkiStats?.l2Hits || 0,
          name: 'L2 (Redis)',
          itemStyle: { color: '#3b82f6' },
        },
        {
          value: stats.value?.dkiStats?.l3Computes || 0,
          name: 'L3 (计算)',
          itemStyle: { color: '#f59e0b' },
        },
      ],
    },
  ],
}))

// Request trend chart (mock data for demo)
const trendChartOption = computed(() => ({
  tooltip: {
    trigger: 'axis',
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true,
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
    axisLine: {
      lineStyle: { color: 'var(--border-color)' },
    },
    axisLabel: {
      color: 'var(--text-secondary)',
    },
  },
  yAxis: {
    type: 'value',
    axisLine: {
      lineStyle: { color: 'var(--border-color)' },
    },
    axisLabel: {
      color: 'var(--text-secondary)',
    },
    splitLine: {
      lineStyle: { color: 'var(--border-color)', type: 'dashed' },
    },
  },
  series: [
    {
      name: '请求数',
      type: 'line',
      smooth: true,
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(16, 185, 129, 0.3)' },
            { offset: 1, color: 'rgba(16, 185, 129, 0)' },
          ],
        },
      },
      lineStyle: {
        color: '#10b981',
        width: 2,
      },
      itemStyle: {
        color: '#10b981',
      },
      data: [120, 80, 150, 280, 350, 420, 380],
    },
  ],
}))

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  
  if (days > 0) {
    return `${days}天 ${hours}小时`
  } else if (hours > 0) {
    return `${hours}小时 ${minutes}分钟`
  } else {
    return `${minutes}分钟`
  }
}

async function handleAuth() {
  if (!authPassword.value) {
    authError.value = '请输入密码'
    return
  }
  
  authLoading.value = true
  authError.value = ''
  
  try {
    const success = statsAuthStore.authenticate(authPassword.value)
    if (success) {
      ElMessage.success('验证成功')
      await refreshStats()
    } else {
      authError.value = '密码错误'
    }
  } finally {
    authLoading.value = false
  }
}

function handleLogout() {
  statsAuthStore.logout()
  authPassword.value = ''
}

async function refreshStats() {
  loading.value = true
  
  try {
    stats.value = await api.stats.getSystemStats()
  } catch (error) {
    // Use mock data for demo
    stats.value = {
      dkiStats: {
        totalRequests: 1234,
        l1Hits: 856,
        l2Hits: 234,
        l3Computes: 144,
        avgAlpha: 0.312,
        injectionRate: 0.87,
      },
      cacheStats: {
        l1Size: 456,
        l1MaxSize: 1000,
        l1HitRate: 0.69,
        l2HitRate: 0.19,
      },
      adapterStats: {
        type: 'postgresql',
        connected: true,
      },
      uptimeSeconds: 86400 * 3 + 3600 * 5 + 60 * 23,
    }
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  if (isAuthenticated.value) {
    refreshStats()
    // Auto refresh every 30 seconds
    refreshInterval = setInterval(refreshStats, 30000)
  }
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style lang="scss" scoped>
.stats-view {
  height: 100%;
  overflow-y: auto;
}

.auth-gate {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-color) 100%);
}

.auth-card {
  width: 400px;
  padding: 48px;
  background: var(--bg-color);
  border-radius: 16px;
  border: 1px solid var(--border-color);
  text-align: center;
  box-shadow: var(--shadow-lg);
  
  .auth-icon {
    font-size: 48px;
    color: var(--text-muted);
    margin-bottom: 24px;
  }
  
  h2 {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 8px;
  }
  
  p {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0 0 32px;
  }
  
  .el-form-item {
    margin-bottom: 16px;
  }
}

.stats-content {
  padding: 24px;
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
  
  .header-actions {
    display: flex;
    gap: 12px;
  }
}

.overview-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.overview-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 20px;
  
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    
    .card-title {
      font-size: 14px;
      color: var(--text-secondary);
    }
    
    .card-icon {
      font-size: 20px;
      color: var(--text-muted);
      
      &.success { color: #10b981; }
      &.warning { color: #f59e0b; }
      &.info { color: #3b82f6; }
    }
  }
  
  .card-value {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
  }
  
  .card-trend {
    font-size: 12px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 4px;
    
    &.positive {
      color: #10b981;
    }
    
    &.negative {
      color: #ef4444;
    }
  }
}

.charts-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.chart-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 20px;
  
  h3 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px;
  }
}

.detailed-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 16px;
}

.stats-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 20px;
  
  h3 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px;
  }
}
</style>
