## 文档类型：技术文档

### 结构要求
1. **标题**：H1 标题，简洁明确
2. **概述**：1-2 段说明文档目的和背景
3. **架构设计**：使用文字描述系统架构，如有必要使用 ASCII 图或 Mermaid 语法
4. **核心模块**：按模块分节描述，每节包含职责、接口、关键逻辑
5. **数据结构**：关键数据结构定义（代码块）
6. **API 接口**：接口定义，包含请求/响应格式
7. **注意事项**：性能、安全、兼容性等需要注意的点

### 写作规范
- 使用 Markdown 标题层级（H1 -> H2 -> H3）
- 代码示例使用代码块，标注语言类型
- 技术术语首次出现时给出解释
- 语气：technical，准确、简洁、无冗余
- 保持客观，不使用主观评价词

### 示例
<example>
# API 网关模块技术文档

## 概述

API 网关模块负责统一处理所有外部 API 请求，提供路由转发、鉴权、限流和监控能力。

## 架构设计

```
Client -> [Nginx] -> [API Gateway] -> [Backend Services]
                      |-> Auth Middleware
                      |-> Rate Limiter
                      |-> Request Logger
```

## 核心模块

### 路由引擎

**职责**：根据请求路径和 HTTP 方法将请求转发到对应的后端服务。

**关键接口**：
- `route(request: HttpRequest) -> ServiceEndpoint`
- `match(path: str, method: str) -> RouteRule`

## 注意事项

- 限流默认阈值为 100 QPS，可通过配置文件调整
- 鉴权中间件支持 JWT 和 API Key 两种方式
</example>
