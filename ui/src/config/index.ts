// Application configuration
export const config = {
    // API settings
    // 注意: baseUrl 应该是后端服务器的根路径，不带 /api 后缀
    // 各 API 端点会自行添加正确的前缀 (/api/*, /v1/*)
    api: {
        baseUrl: import.meta.env.VITE_API_BASE_URL || "",
        timeout: 30000,
    },

    // Stats page authentication
    // In production, this should be handled by proper backend authentication
    statsAuth: {
        enabled: true,
        // Simple password protection for stats page (non-production standard)
        password: import.meta.env.VITE_STATS_PASSWORD || "dki_admin_2024",
    },

    // Default settings
    defaults: {
        model: "dki-default",
        temperature: 0.7,
        maxTokens: 2048,
        dkiAlpha: 0.3,
    },

    // Feature flags
    features: {
        streaming: true,
        hybridInjection: true,
        debugMode: import.meta.env.DEV,
    },
};

export default config;
