<template>
  <div class="login-page">
    <!-- Background decoration -->
    <div class="bg-decoration">
      <div class="circle circle-1"></div>
      <div class="circle circle-2"></div>
      <div class="circle circle-3"></div>
    </div>
    
    <div class="login-container">
      <!-- Left side - Branding -->
      <div class="branding-section">
        <div class="brand-content">
          <img src="/logo.svg" alt="DKI" class="brand-logo" />
          <h1 class="brand-title">DKI Chat</h1>
          <p class="brand-subtitle">Dynamic KV Injection System</p>
          <div class="brand-features">
            <div class="feature-item">
              <el-icon><Lightning /></el-icon>
              <span>注意力级记忆增强</span>
            </div>
            <div class="feature-item">
              <el-icon><User /></el-icon>
              <span>个性化用户偏好</span>
            </div>
            <div class="feature-item">
              <el-icon><Connection /></el-icon>
              <span>会话历史智能注入</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Right side - Login form -->
      <div class="form-section">
        <div class="form-container">
          <h2 class="form-title">欢迎回来</h2>
          <p class="form-subtitle">登录您的账户以继续</p>
          
          <el-form
            ref="formRef"
            :model="form"
            :rules="rules"
            class="login-form"
            @submit.prevent="handleLogin"
          >
            <el-form-item prop="username">
              <el-input
                v-model="form.username"
                placeholder="用户名"
                size="large"
                :prefix-icon="User"
              />
            </el-form-item>
            
            <el-form-item prop="password">
              <el-input
                v-model="form.password"
                type="password"
                placeholder="密码"
                size="large"
                :prefix-icon="Lock"
                show-password
                @keyup.enter="handleLogin"
              />
            </el-form-item>
            
            <div class="form-options">
              <el-checkbox v-model="form.remember">记住我</el-checkbox>
              <el-link type="primary" :underline="false">忘记密码？</el-link>
            </div>
            
            <el-form-item>
              <el-button
                type="primary"
                size="large"
                class="login-btn"
                :loading="loading"
                @click="handleLogin"
              >
                登录
              </el-button>
            </el-form-item>
          </el-form>
          
          <div class="form-footer">
            <span>还没有账户？</span>
            <el-link type="primary" :underline="false" @click="showRegister = true">
              立即注册
            </el-link>
          </div>
          
          <!-- Demo login hint -->
          <div class="demo-hint">
            <el-alert
              title="演示模式"
              type="info"
              :closable="false"
              show-icon
            >
              <template #default>
                使用任意用户名和密码登录（演示模式）
              </template>
            </el-alert>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Register Dialog -->
    <el-dialog
      v-model="showRegister"
      title="创建账户"
      width="400px"
      :close-on-click-modal="false"
    >
      <el-form
        ref="registerFormRef"
        :model="registerForm"
        :rules="registerRules"
        label-position="top"
      >
        <el-form-item label="用户名" prop="username">
          <el-input v-model="registerForm.username" placeholder="请输入用户名" />
        </el-form-item>
        <el-form-item label="邮箱" prop="email">
          <el-input v-model="registerForm.email" placeholder="请输入邮箱（可选）" />
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input
            v-model="registerForm.password"
            type="password"
            placeholder="请输入密码"
            show-password
          />
        </el-form-item>
        <el-form-item label="确认密码" prop="confirmPassword">
          <el-input
            v-model="registerForm.confirmPassword"
            type="password"
            placeholder="请再次输入密码"
            show-password
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showRegister = false">取消</el-button>
        <el-button type="primary" :loading="registerLoading" @click="handleRegister">
          注册
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { User, Lock, Lightning, Connection } from '@element-plus/icons-vue'
import { ElMessage, FormInstance, FormRules } from 'element-plus'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const formRef = ref<FormInstance>()
const registerFormRef = ref<FormInstance>()
const loading = ref(false)
const registerLoading = ref(false)
const showRegister = ref(false)

const form = reactive({
  username: '',
  password: '',
  remember: false,
})

const registerForm = reactive({
  username: '',
  email: '',
  password: '',
  confirmPassword: '',
})

const rules: FormRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 2, max: 20, message: '用户名长度为 2-20 个字符', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 4, max: 50, message: '密码长度为 4-50 个字符', trigger: 'blur' },
  ],
}

const registerRules: FormRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 2, max: 20, message: '用户名长度为 2-20 个字符', trigger: 'blur' },
  ],
  email: [
    { type: 'email', message: '请输入有效的邮箱地址', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 50, message: '密码长度为 6-50 个字符', trigger: 'blur' },
  ],
  confirmPassword: [
    { required: true, message: '请确认密码', trigger: 'blur' },
    {
      validator: (_rule, value, callback) => {
        if (value !== registerForm.password) {
          callback(new Error('两次输入的密码不一致'))
        } else {
          callback()
        }
      },
      trigger: 'blur',
    },
  ],
}

async function handleLogin() {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    loading.value = true
    
    try {
      // Demo mode: accept any credentials
      // In production, this would call the actual API
      const success = await authStore.login({
        username: form.username,
        password: form.password,
        remember: form.remember,
      })
      
      if (success) {
        ElMessage.success('登录成功')
        const redirect = route.query.redirect as string || '/'
        router.push(redirect)
      } else {
        // Demo mode fallback
        authStore.user = {
          id: `user-${Date.now()}`,
          username: form.username,
        }
        authStore.token = `demo-token-${Date.now()}`
        ElMessage.success('登录成功（演示模式）')
        const redirect = route.query.redirect as string || '/'
        router.push(redirect)
      }
    } catch (error) {
      // Demo mode fallback
      authStore.user = {
        id: `user-${Date.now()}`,
        username: form.username,
      }
      authStore.token = `demo-token-${Date.now()}`
      ElMessage.success('登录成功（演示模式）')
      const redirect = route.query.redirect as string || '/'
      router.push(redirect)
    } finally {
      loading.value = false
    }
  })
}

async function handleRegister() {
  if (!registerFormRef.value) return
  
  await registerFormRef.value.validate(async (valid) => {
    if (!valid) return
    
    registerLoading.value = true
    
    try {
      // Demo mode: just close dialog and show success
      ElMessage.success('注册成功，请登录')
      showRegister.value = false
      form.username = registerForm.username
    } finally {
      registerLoading.value = false
    }
  })
}
</script>

<style lang="scss" scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  position: relative;
  overflow: hidden;
}

.bg-decoration {
  position: absolute;
  inset: 0;
  pointer-events: none;
  
  .circle {
    position: absolute;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
    
    &.circle-1 {
      width: 600px;
      height: 600px;
      top: -200px;
      right: -200px;
    }
    
    &.circle-2 {
      width: 400px;
      height: 400px;
      bottom: -100px;
      left: -100px;
    }
    
    &.circle-3 {
      width: 200px;
      height: 200px;
      top: 50%;
      left: 30%;
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05));
    }
  }
}

.login-container {
  display: flex;
  width: 900px;
  max-width: 95vw;
  background: var(--bg-surface);
  border-radius: 24px;
  overflow: hidden;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
  position: relative;
  z-index: 1;
}

.branding-section {
  flex: 1;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  padding: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  
  @media (max-width: 768px) {
    display: none;
  }
}

.brand-content {
  text-align: center;
  color: white;
  
  .brand-logo {
    width: 80px;
    height: 80px;
    margin-bottom: 24px;
    filter: brightness(0) invert(1);
  }
  
  .brand-title {
    font-size: 32px;
    font-weight: 700;
    margin: 0 0 8px;
  }
  
  .brand-subtitle {
    font-size: 14px;
    opacity: 0.9;
    margin: 0 0 32px;
  }
}

.brand-features {
  display: flex;
  flex-direction: column;
  gap: 16px;
  text-align: left;
  
  .feature-item {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    
    .el-icon {
      font-size: 20px;
    }
  }
}

.form-section {
  flex: 1;
  padding: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-color);
}

.form-container {
  width: 100%;
  max-width: 320px;
}

.form-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 8px;
}

.form-subtitle {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0 0 32px;
}

.login-form {
  .el-form-item {
    margin-bottom: 20px;
  }
  
  .el-input {
    --el-input-border-radius: 12px;
  }
}

.form-options {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.login-btn {
  width: 100%;
  height: 48px;
  font-size: 16px;
  border-radius: 12px;
}

.form-footer {
  text-align: center;
  margin-top: 24px;
  font-size: 14px;
  color: var(--text-secondary);
  
  .el-link {
    margin-left: 4px;
  }
}

.demo-hint {
  margin-top: 24px;
  
  :deep(.el-alert) {
    border-radius: 12px;
  }
}
</style>
