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
              <span>Attention-Level Memory Enhancement</span>
            </div>
            <div class="feature-item">
              <el-icon><User /></el-icon>
              <span>Personalized User Preferences</span>
            </div>
            <div class="feature-item">
              <el-icon><Connection /></el-icon>
              <span>Smart History Injection</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Right side - Login form -->
      <div class="form-section">
        <div class="form-container">
          <h2 class="form-title">Welcome Back</h2>
          <p class="form-subtitle">Sign in to your account to continue</p>
          
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
                placeholder="Username"
                size="large"
                :prefix-icon="User"
              />
            </el-form-item>
            
            <el-form-item prop="password">
              <el-input
                v-model="form.password"
                type="password"
                placeholder="Password"
                size="large"
                :prefix-icon="Lock"
                show-password
                @keyup.enter="handleLogin"
              />
            </el-form-item>
            
            <div class="form-options">
              <el-checkbox v-model="form.remember">Remember me</el-checkbox>
              <el-link type="primary" :underline="false">Forgot password?</el-link>
            </div>
            
            <el-form-item>
              <el-button
                type="primary"
                size="large"
                class="login-btn"
                :loading="loading"
                @click="handleLogin"
              >
                Sign In
              </el-button>
            </el-form-item>
          </el-form>
          
          <div class="form-footer">
            <span>Don't have an account?</span>
            <el-link type="primary" :underline="false" @click="showRegister = true">
              Register Now
            </el-link>
          </div>
          
          <!-- Demo login hint -->
          <div class="demo-hint">
            <el-alert
              title="Demo Mode"
              type="info"
              :closable="false"
              show-icon
            >
              <template #default>
                Use any username and password to log in (demo mode)
              </template>
            </el-alert>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Register Dialog -->
    <el-dialog
      v-model="showRegister"
      title="Create Account"
      width="400px"
      :close-on-click-modal="false"
    >
      <el-form
        ref="registerFormRef"
        :model="registerForm"
        :rules="registerRules"
        label-position="top"
      >
        <el-form-item label="Username" prop="username">
          <el-input v-model="registerForm.username" placeholder="Enter username" />
        </el-form-item>
        <el-form-item label="Email" prop="email">
          <el-input v-model="registerForm.email" placeholder="Enter email (optional)" />
        </el-form-item>
        <el-form-item label="Password" prop="password">
          <el-input
            v-model="registerForm.password"
            type="password"
            placeholder="Enter password"
            show-password
          />
        </el-form-item>
        <el-form-item label="Confirm Password" prop="confirmPassword">
          <el-input
            v-model="registerForm.confirmPassword"
            type="password"
            placeholder="Re-enter password"
            show-password
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showRegister = false">Cancel</el-button>
        <el-button type="primary" :loading="registerLoading" @click="handleRegister">
          Register
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
    { required: true, message: 'Please enter username', trigger: 'blur' },
    { min: 2, max: 20, message: 'Username must be 2-20 characters', trigger: 'blur' },
  ],
  password: [
    { required: true, message: 'Please enter password', trigger: 'blur' },
    { min: 4, max: 50, message: 'Password must be 4-50 characters', trigger: 'blur' },
  ],
}

const registerRules: FormRules = {
  username: [
    { required: true, message: 'Please enter username', trigger: 'blur' },
    { min: 2, max: 20, message: 'Username must be 2-20 characters', trigger: 'blur' },
  ],
  email: [
    { type: 'email', message: 'Please enter a valid email address', trigger: 'blur' },
  ],
  password: [
    { required: true, message: 'Please enter password', trigger: 'blur' },
    { min: 6, max: 50, message: 'Password must be 6-50 characters', trigger: 'blur' },
  ],
  confirmPassword: [
    { required: true, message: 'Please confirm password', trigger: 'blur' },
    {
      validator: (_rule, value, callback) => {
        if (value !== registerForm.password) {
          callback(new Error('Passwords do not match'))
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
        ElMessage.success('Login successful')
        const redirect = route.query.redirect as string || '/'
        router.push(redirect)
      } else {
        // Demo mode fallback
        authStore.user = {
          id: `user-${Date.now()}`,
          username: form.username,
        }
        authStore.token = `demo-token-${Date.now()}`
        ElMessage.success('Login successful (demo mode)')
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
      ElMessage.success('Login successful (demo mode)')
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
      ElMessage.success('Registration successful, please sign in')
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
