# 旅游实时工具 MCP Server

提供四个 MCP 工具：

- `geocode_location`：地点名称解析为经纬度
- `get_current_weather`：实时天气
- `get_weather_forecast`：未来 1～14 天天气预报
- `plan_route`：直线距离、驾车、步行和公交路线

## 数据源

- 天气和默认地理编码：Open-Meteo，无需 API Key
- 默认驾车路线：OSRM，无需 API Key
- 国内地点与驾车/步行/公交路线：高德地图，配置 `AMAP_API_KEY` 后优先启用

## 启动

```powershell
cd "D:\agentic rag\AgentiRAG-master"
..\.venv\Scripts\python.exe -m mcp_server.travel_tools
```

该命令使用 stdio 传输，启动后没有普通命令行提示，等待 MCP 客户端连接属于正常行为。

## MCP 客户端配置

```json
{
  "mcpServers": {
    "agentic-rag-travel-tools": {
      "command": "D:\\agentic rag\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server.travel_tools"],
      "cwd": "D:\\agentic rag\\AgentiRAG-master"
    }
  }
}
```

若需要国内步行或公交规划，在项目 `.env` 中加入：

```dotenv
AMAP_API_KEY="你的高德Web服务Key"
MCP_HTTP_TIMEOUT=20
```
