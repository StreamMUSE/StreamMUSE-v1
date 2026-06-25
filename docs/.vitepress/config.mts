import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'StreamMUSE',
  description: '实时 AI 音乐伴奏生成系统',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '快速开始', link: '/getting-started/installation' },
      { text: '用户指南', link: '/user-guide/running-realtime' },
      { text: '架构', link: '/architecture/overview' },
      { text: '开发者指南', link: '/developer-guide/adding-input-source' },
      { text: '参考', link: '/reference/cli-reference' },
    ],

    sidebar: {
      '/getting-started/': [
        {
          text: '快速开始',
          items: [
            { text: '安装', link: '/getting-started/installation' },
            { text: '快速上手', link: '/getting-started/quickstart' },
            { text: '配置项', link: '/getting-started/configuration' },
          ],
        },
      ],

      '/user-guide/': [
        {
          text: '用户指南',
          items: [
            { text: '运行实时系统', link: '/user-guide/running-realtime' },
            { text: '输入模式', link: '/user-guide/input-modes' },
            { text: '输出类型', link: '/user-guide/output-types' },
            { text: 'Session 日志', link: '/user-guide/session-logging' },
            { text: '音乐注入', link: '/user-guide/music-injection' },
            { text: 'Lekai Mac 本地运行', link: '/user-guide/lekai-mac-local' },
          ],
        },
      ],

      '/architecture/': [
        { text: '架构总览', link: '/architecture/overview' },
        {
          text: 'Presentation 层',
          collapsed: false,
          items: [
            { text: '概览', link: '/architecture/presentation/overview' },
            { text: 'CLI', link: '/architecture/presentation/cli' },
            { text: '配置解析', link: '/architecture/presentation/config_parser' },
          ],
        },
        {
          text: 'Application 层',
          collapsed: false,
          items: [
            { text: '概览', link: '/architecture/application/overview' },
            { text: '配置模型', link: '/architecture/application/config' },
            { text: '工厂', link: '/architecture/application/factories' },
            { text: 'RealTimeMusicService', link: '/architecture/application/service' },
          ],
        },
        {
          text: 'Domain 层',
          collapsed: false,
          items: [
            { text: '概览', link: '/architecture/domain/overview' },
            { text: '接口协议', link: '/architecture/domain/interfaces' },
            { text: '音乐模型', link: '/architecture/domain/musical' },
            { text: '时序', link: '/architecture/domain/timing' },
            { text: '日志', link: '/architecture/domain/logging' },
          ],
        },
        {
          text: 'Infrastructure 层',
          collapsed: false,
          items: [
            { text: '概览', link: '/architecture/infrastructure/overview' },
            {
              text: '输入',
              collapsed: true,
              items: [
                { text: '概览', link: '/architecture/infrastructure/input/overview' },
                { text: '键盘', link: '/architecture/infrastructure/input/keyboard' },
                { text: 'MIDI 设备', link: '/architecture/infrastructure/input/midi_device' },
                { text: 'MIDI 文件', link: '/architecture/infrastructure/input/midi_file' },
                { text: 'List 输入', link: '/architecture/infrastructure/input/list_input' },
              ],
            },
            {
              text: '输出',
              collapsed: true,
              items: [
                { text: '概览', link: '/architecture/infrastructure/output/overview' },
                { text: '音频', link: '/architecture/infrastructure/output/audio' },
                { text: '控制台', link: '/architecture/infrastructure/output/console' },
                { text: 'MIDI 文件', link: '/architecture/infrastructure/output/midi_file' },
                { text: 'WebSocket', link: '/architecture/infrastructure/output/websocket' },
                { text: 'JSON 日志', link: '/architecture/infrastructure/output/json_logger' },
                { text: 'Session 日志', link: '/architecture/infrastructure/output/session_logger' },
                { text: '组合输出', link: '/architecture/infrastructure/output/composite' },
                { text: '节拍器', link: '/architecture/infrastructure/output/metronome' },
              ],
            },
            {
              text: '推理',
              collapsed: true,
              items: [
                { text: '概览', link: '/architecture/infrastructure/inference/overview' },
                { text: 'HTTP 客户端', link: '/architecture/infrastructure/inference/http_client' },
                { text: '序列化', link: '/architecture/infrastructure/inference/serialization' },
                { text: 'Stanley 引擎', link: '/architecture/infrastructure/inference/stanley_engine' },
                { text: 'Stanley Legacy', link: '/architecture/infrastructure/inference/stanley_legacy' },
              ],
            },
          ],
        },
      ],

      '/developer-guide/': [
        {
          text: '开发者指南',
          items: [
            { text: '新增输入源', link: '/developer-guide/adding-input-source' },
            { text: '新增输出 Sink', link: '/developer-guide/adding-output-sink' },
            { text: '新增推理引擎', link: '/developer-guide/adding-inference-engine' },
            { text: 'Fake 推理服务器', link: '/developer-guide/fake-inference-server' },
            { text: '生成日志', link: '/developer-guide/generation-logging' },
            { text: '测试策略', link: '/developer-guide/testing-strategy' },
            { text: '一致性测试', link: '/developer-guide/consistency-test' },
          ],
        },
      ],

      '/reference/': [
        {
          text: '参考',
          items: [
            { text: 'CLI 参考', link: '/reference/cli-reference' },
            { text: '术语表', link: '/reference/glossary' },
          ],
        },
      ],
    },

    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一页', next: '下一页' },
    lastUpdated: { text: '最后更新' },
    darkModeSwitchLabel: '深色模式',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '回到顶部',

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '未找到结果',
            resetButtonTitle: '清除搜索',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    socialLinks: [],
  },
})
