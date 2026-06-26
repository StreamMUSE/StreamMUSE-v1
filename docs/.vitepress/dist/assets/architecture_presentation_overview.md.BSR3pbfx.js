import{_ as n,o as e,c as s,a2 as p}from"./chunks/framework.DnrXFDHb.js";const u=JSON.parse('{"title":"presentation 层总览","description":"CLI 入口点与配置解析的职责与流程","frontmatter":{"title":"presentation 层总览","description":"CLI 入口点与配置解析的职责与流程"},"headers":[],"relativePath":"architecture/presentation/overview.md","filePath":"architecture/presentation/overview.md","lastUpdated":1775717144000}'),t={name:"architecture/presentation/overview.md"};function i(o,a,c,l,r,d){return e(),s("div",null,[...a[0]||(a[0]=[p(`<h1 id="presentation-层总览" tabindex="-1">presentation 层总览 <a class="header-anchor" href="#presentation-层总览" aria-label="Permalink to &quot;presentation 层总览&quot;">​</a></h1><p><strong>源文件</strong>：<code>src/streammuse/presentation/cli/</code></p><p>Presentation 层是系统的<strong>入口点</strong>，负责将命令行参数转换为 <code>ApplicationConfig</code>，并将所有组件组装为可运行的服务。</p><hr><h2 id="职责" tabindex="-1">职责 <a class="header-anchor" href="#职责" aria-label="Permalink to &quot;职责&quot;">​</a></h2><ul><li>解析 CLI 参数（<code>argparse</code>）</li><li>从环境变量读取配置（<code>env_to_config()</code>）</li><li>构建 <code>ApplicationConfig</code>（<code>args_to_config()</code>）</li><li>创建 <code>SessionManager</code>（如有日志需求）</li><li>通过三个 Factory 创建组件</li><li>创建 <code>Tempo</code>、<code>PlaybackScheduler</code></li><li>注册清理（<code>atexit</code>）和信号处理（<code>SIGINT</code>、<code>SIGTERM</code>）</li><li>启动 <code>RealTimeMusicService</code></li></ul><hr><h2 id="启动流程" tabindex="-1">启动流程 <a class="header-anchor" href="#启动流程" aria-label="Permalink to &quot;启动流程&quot;">​</a></h2><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>uv run streammuse-cli --input-mode keyboard ...</span></span>
<span class="line"><span>    │</span></span>
<span class="line"><span>    ▼ main()</span></span>
<span class="line"><span>parse_args()</span></span>
<span class="line"><span>    │</span></span>
<span class="line"><span>    ├── env_to_config()     # 读取环境变量（当前仅返回 None）</span></span>
<span class="line"><span>    └── args_to_config()    # CLI 参数 → ApplicationConfig</span></span>
<span class="line"><span>    │</span></span>
<span class="line"><span>    ▼</span></span>
<span class="line"><span>SessionManager（如 output_type 为 json_log/session/composite）</span></span>
<span class="line"><span>    │</span></span>
<span class="line"><span>    ▼ 三个 Factory</span></span>
<span class="line"><span>InputSourceFactory.create()   → InputSource</span></span>
<span class="line"><span>OutputSinkFactory.create()    → OutputSink</span></span>
<span class="line"><span>InferenceEngineFactory.create() → InferenceEngine</span></span>
<span class="line"><span>    │</span></span>
<span class="line"><span>    ▼</span></span>
<span class="line"><span>Tempo + PlaybackScheduler</span></span>
<span class="line"><span>    │</span></span>
<span class="line"><span>    ▼</span></span>
<span class="line"><span>RealTimeMusicService.start()</span></span>
<span class="line"><span>    │</span></span>
<span class="line"><span>    ▼</span></span>
<span class="line"><span>while service.running:</span></span>
<span class="line"><span>    sleep(0.1)</span></span></code></pre></div><hr><h2 id="组件" tabindex="-1">组件 <a class="header-anchor" href="#组件" aria-label="Permalink to &quot;组件&quot;">​</a></h2><table tabindex="0"><thead><tr><th>文件</th><th>说明</th></tr></thead><tbody><tr><td><code>cli.py</code></td><td><code>main()</code> 入口函数，生命周期管理</td></tr><tr><td><code>config_parser.py</code></td><td><code>parse_args()</code>、<code>args_to_config()</code>、<code>env_to_config()</code></td></tr></tbody></table><hr><h2 id="详细文档" tabindex="-1">详细文档 <a class="header-anchor" href="#详细文档" aria-label="Permalink to &quot;详细文档&quot;">​</a></h2><ul><li><a href="./cli.html">cli.md</a> — <code>main()</code> 流程、清理与信号处理</li><li><a href="./config_parser.html">config_parser.md</a> — CLI 参数说明与配置构建</li></ul>`,15)])])}const _=n(t,[["render",i]]);export{u as __pageData,_ as default};
