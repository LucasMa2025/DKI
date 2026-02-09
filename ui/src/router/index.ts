import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useStatsAuthStore } from '@/stores/statsAuth'

const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/LoginView.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/',
    component: () => import('@/layouts/MainLayout.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        name: 'Chat',
        component: () => import('@/views/ChatView.vue'),
      },
      {
        path: 'sessions',
        name: 'Sessions',
        component: () => import('@/views/SessionsView.vue'),
      },
      {
        path: 'preferences',
        name: 'Preferences',
        component: () => import('@/views/PreferencesView.vue'),
      },
      {
        path: 'stats',
        name: 'Stats',
        component: () => import('@/views/StatsView.vue'),
        meta: { requiresStatsAuth: true },
      },
    ],
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/',
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// Navigation guards
router.beforeEach((to, _from, next) => {
  const authStore = useAuthStore()
  const statsAuthStore = useStatsAuthStore()
  
  // Check if route requires authentication
  if (to.meta.requiresAuth !== false && !authStore.isAuthenticated) {
    next({ name: 'Login', query: { redirect: to.fullPath } })
    return
  }
  
  // Check if route requires stats authentication
  if (to.meta.requiresStatsAuth && !statsAuthStore.isAuthenticated) {
    // Will be handled by the StatsView component
    next()
    return
  }
  
  // Redirect to home if already logged in and trying to access login
  if (to.name === 'Login' && authStore.isAuthenticated) {
    next({ name: 'Chat' })
    return
  }
  
  next()
})

export default router
