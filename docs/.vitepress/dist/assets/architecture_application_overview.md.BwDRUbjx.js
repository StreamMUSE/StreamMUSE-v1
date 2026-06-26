import{_ as n,o as s,c as e,a2 as p}from"./chunks/framework.DnrXFDHb.js";const h=JSON.parse('{"title":"Application 层总览","description":"StreamMUSE Application 层的职责、组件构成与组装流程","frontmatter":{"title":"Application 层总览","description":"StreamMUSE Application 层的职责、组件构成与组装流程"},"headers":[],"relativePath":"architecture/application/overview.md","filePath":"architecture/application/overview.md","lastUpdated":1775717144000}'),i={name:"architecture/application/overview.md"};function c(t,a,l,o,r,d){return s(),e("div",null,[...a[0]||(a[0]=[p(`<h1 id="application-层总览" tabindex="-1">Application 层总览 <a class="header-anchor" href="#application-层总览" aria-label="Permalink to &quot;Application 层总览&quot;">​</a></h1><p><strong>源文件</strong>：<code>src/streammuse/application/</code></p><p>Application 层是系统的<strong>编排层</strong>，负责将各组件组装为可运行的服务。</p><p>当前实现中：</p><ol><li><code>services/</code> 主要依赖 Domain 协议与类型。</li><li><code>factories/</code> 作为组合入口，会直接导入 Infrastructure 的具体实现类并完成装配。</li></ol><hr><h2 id="职责" tabindex="-1">职责 <a class="header-anchor" href="#职责" aria-label="Permalink to &quot;职责&quot;">​</a></h2><ul><li>定义配置数据模型（<code>ApplicationConfig</code> 及其子配置）</li><li>通过 Factory 将配置转换为 Domain 接口的具体实现</li><li>运行核心实时服务（<code>RealTimeMusicService</code>）</li></ul><hr><h2 id="组件构成" tabindex="-1">组件构成 <a class="header-anchor" href="#组件构成" aria-label="Permalink to &quot;组件构成&quot;">​</a></h2><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>application/</span></span>
<span class="line"><span>├── config/</span></span>
<span class="line"><span>│   └── models.py          # TempoConfig、InputConfig、OutputConfig、InferenceConfig、ApplicationConfig</span></span>
<span class="line"><span>├── factories/</span></span>
<span class="line"><span>│   ├── input_factory.py   # InputSourceFactory</span></span>
<span class="line"><span>│   ├── output_factory.py  # OutputSinkFactory</span></span>
<span class="line"><span>│   └── inference_factory.py  # InferenceEngineFactory</span></span>
<span class="line"><span>└── services/</span></span>
<span class="line"><span>    └── real_time_music_service.py  # RealTimeMusicService</span></span></code></pre></div><hr><h2 id="组装流程" tabindex="-1">组装流程 <a class="header-anchor" href="#组装流程" aria-label="Permalink to &quot;组装流程&quot;">​</a></h2><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>CLI args</span></span>
<span class="line"><span>    │</span></span>
<span class="line"><span>    ▼</span></span>
<span class="line"><span>ApplicationConfig（由 config_parser 构建）</span></span>
<span class="line"><span>    │</span></span>
<span class="line"><span>    ├──▶ InputSourceFactory.create(config)  ──▶ InputSource（实现类）</span></span>
<span class="line"><span>    ├──▶ OutputSinkFactory.create(config)   ──▶ OutputSink（实现类）</span></span>
<span class="line"><span>    └──▶ InferenceEngineFactory.create(config) ──▶ InferenceEngine（实现类）</span></span>
<span class="line"><span>                                  │</span></span>
<span class="line"><span>                                  ▼</span></span>
<span class="line"><span>                    RealTimeMusicService(</span></span>
<span class="line"><span>                        input_source=...,</span></span>
<span class="line"><span>                        output_sink=...,</span></span>
<span class="line"><span>                        inference_engine=...,</span></span>
<span class="line"><span>                        tempo=...,</span></span>
<span class="line"><span>                        scheduler=...</span></span>
<span class="line"><span>                    )</span></span>
<span class="line"><span>                                  │</span></span>
<span class="line"><span>                                  ▼</span></span>
<span class="line"><span>                          service.start()</span></span></code></pre></div><p><strong>关键原则</strong>：<code>RealTimeMusicService</code> 持有的是 Domain 接口（<code>InputSource</code>、<code>OutputSink</code>、<code>InferenceEngine</code>），而非任何具体类。这使得在测试中可以轻松用 mock 替换任何组件。</p><hr><h2 id="详细文档" tabindex="-1">详细文档 <a class="header-anchor" href="#详细文档" aria-label="Permalink to &quot;详细文档&quot;">​</a></h2><ul><li><a href="./config.html">config.md</a> — 配置数据模型</li><li><a href="./factories.html">factories.md</a> — 三个 Factory 的实现细节</li><li><a href="./service.html">service.md</a> — <code>RealTimeMusicService</code> 三线程架构</li></ul>`,18)])])}const f=n(i,[["render",c]]);export{h as __pageData,f as default};
