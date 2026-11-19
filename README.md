# Checkpoint Reshard API

训练进程数变化时，把同一份 float32 检查点（**权重 + 动量**）从只读源卷重切到新的进程数，
输出每个进程的全局下标区间和原始步数，供训练作业直接续跑。
训练损失出现 NaN 时，还可以对比**先后两份等长检查点**，按全局下标定位
**刚从有限值变成 NaN/Inf** 的参数（仅权重 / 仅动量 / 两者三类）。

- 源清单（`manifest.json`）声明步数、总长度、每个 float32 文件覆盖的全局下标区间；
  权重与动量的切分点**可以不同**。
- 源区间有**缺口或重叠**（或文件大小不符）→ 拒绝转换（HTTP 422）。
- 按新进程数把全局下标 `[0, total_length)` 均分为连续区间（大小差 ≤ 1 个元素），
  分块流式读写（默认 4 MiB，可用 `CHUNK_BYTES` 调整）。
- 每个 rank 目录同时产出该区间的 `weights.f32` 和 `momentum.f32`，外加
  `shard.json`（区间 + 原始步数）。
- 中断/重启后续做：**已完整落盘的分片原样复用**，只补未完成的；
  **整组齐备前下载一律 409**。
- 检查点对比（`/diagnose`）：四组（新/旧 × 权重/动量）分片边界再不一样，
  也按**全局下标**分块对齐后逐元素比较；只统计**有限 → NaN/Inf** 的新异常，
  每类给出总数和下标最小的至多 5 处。
- 诊断与重分片的分块读**共用同一个并发上限**（`READ_CONCURRENCY`，默认 4）；
  有重分片在等待读名额时，空出的名额**优先给重分片**；读失败名额照样释放。

## 快速开始（Docker Compose）

仓库自带两份本地样例，分片边界互不相同：

- `sample/source/` —— 较早的检查点（步数 7459、总长 10007、权重 3 段 / 动量 3 段不同切分）
- `sample/next/` —— 较晚的检查点（步数 7460、权重 3 段 / 动量 4 段另样切分，
  并注入 5 处 NaN/Inf 供 `/diagnose` 演示）

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

如需重新生成样例：

```bash
python3 sample/make_sample.py sample/source
python3 sample/make_next.py   sample/next
```

不用 Docker 本地跑：

```bash
pip install -r requirements.txt
SOURCE_DIR=$PWD/sample/source NEXT_DIR=$PWD/sample/next OUTPUT_ROOT=$PWD/outputs \
  uvicorn app.main:create_app --factory --port 8000
```

## API（均可在 /docs 中操作）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/reshard` | 发起转换：`{"new_world_size": 6}`；相同请求幂等返回同一任务 |
| GET | `/jobs` / `/jobs/{job_id}` | 任务列表 / 进度（含各 rank 区间、步数、完成数） |
| POST | `/jobs/{job_id}/resume` | 失败/重启后续做；已完成分片自动跳过 |
| GET | `/jobs/{job_id}/shards/{rank}/meta` | 该 rank 的区间 `[start, end)` 与原始步数 |
| GET | `/jobs/{job_id}/shards/{rank}/download?part=weights\|momentum` | 下载分片；任务未全部完成时 409 |
| POST | `/diagnose` | 对比两份等长检查点：`{}` 用默认挂载（SOURCE_DIR vs NEXT_DIR），也可 `{"old_dir": ..., "new_dir": ...}` 覆盖 |

```bash
# 发起（source_dir 缺省用挂载的 SOURCE_DIR，也可显式覆盖）
curl -X POST localhost:8000/reshard -H 'Content-Type: application/json' \
     -d '{"new_world_size": 6}'
# 查询 → 拿到 job_id 与每个 rank 的 [start, end) 和 step
curl localhost:8000/jobs/<job_id>
# 中断后续做
curl -X POST localhost:8000/jobs/<job_id>/resume
# 整组完成后下载
curl -OJ "localhost:8000/jobs/<job_id>/shards/0/download?part=weights"
# 对比先后两份检查点，定位新出现的 NaN/Inf
curl -X POST localhost:8000/diagnose -H 'Content-Type: application/json' -d '{}'
```

`/diagnose` 的响应（对自带样例）：

```json
{
  "step_old": 7459, "step_new": 7460, "total_length": 10007,
  "weights_only":  {"count": 2, "first_indices": [1234, 9876]},
  "momentum_only": {"count": 2, "first_indices": [55, 7000]},
  "both":          {"count": 1, "first_indices": [5000]}
}
```

- 三类互斥：**仅权重**、**仅动量**、**两者都有**；`first_indices` 是该类下标最小的至多 5 处。
- 旧检查点里已是 NaN/Inf 的位置**不算**新异常（只统计有限 → 非有限的跳变）。
- 两份检查点 `total_length` 不一致 → 422；清单非法（缺口/重叠/大小不符）→ 422；
  读取中途失败 → 500 且**只返回错误**，不给部分计数。

## 源清单格式

```json
{
  "step": 7459,
  "total_length": 10007,
  "weights":  [{"path": "weights/part-0.f32",  "start": 0,    "end": 3500}, ...],
  "momentum": [{"path": "momentum/part-0.f32", "start": 0,    "end": 2000}, ...]
}
```

- 区间为**左闭右开**的全局元素下标；每类的所有区间必须**无缝无重叠**地铺满
  `[0, total_length)`，否则 422。
- 每个文件必须是**原始 little-endian float32**，大小恰好 `(end-start)*4` 字节。
- `path` 相对源目录，拒绝绝对路径与 `..`。

## 输出布局

```
outputs/<job_id>/
  job.json                 # 任务状态，每完成一个分片落盘一次
  rank_00000/
    weights.f32            # 区间 [start, end) 的权重
    momentum.f32           # 同一区间的动量
    shard.json             # {"rank","start","end","step",...} —— 最后写入的完成标记
```

## 设计要点

- **均分**：rank `i` 得 `[i*N//W, (i+1)*N//W)`，连续、全覆盖、大小差 ≤ 1。
- **崩溃安全**：数据先写 `*.tmp` 再 `os.replace` 原子改名；`shard.json` 最后写，
  作为分片完成标记；`job.json` 同样原子替换。服务重启时把 `running` 任务标记为
  可续做的 `failed`。
- **复用判定**：分片目录里标记文件存在且两个数据文件大小精确匹配 → 视为完整，
  续做时直接跳过（不重写）。
- **源变更防护**：续做前重新校验清单，步数/总长/区间与任务不一致则失败。
- **下载闸门**：仅当任务状态为 `done`（全部 rank 齐备）才放行下载，否则 409。
- **幂等任务**：`job_id` 由（源目录、清单内容、新进程数）哈希得出，重复发起
  同一请求返回同一任务。
- **全局下标对齐**：诊断按 `CHUNK_BYTES` 大小的窗口扫描，窗口内用与重分片相同的
  区间数学（`plan_copy_ops`）把四组不同分片布局各自拼回全局下标，再逐元素比较，
  分片边界不影响结果。
- **共享读名额**：`ReadGate` 限制诊断 + 重分片的在途分块读总数（`READ_CONCURRENCY`）；
  重分片等待时空出的名额先给重分片（诊断让路）；名额用 `try/finally` 释放，
  读失败不泄漏。
- **诊断全或无**：扫描中途任何读失败都使整个请求失败，只返回错误，不返回部分计数。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

覆盖：均分数学、清单校验（缺口/重叠/大小/路径穿越）、端到端字节级一致性、
中断续做（完整分片不重写）、重启后复用、源变更检测、下载闸门、API 全流程、
诊断三分类与下标截断、等长校验、读失败释放名额（诊断与重分片两侧）、
交错读盘共享并发上限、重分片优先唤醒。

## 已知限制

- 单进程服务：任务以线程方式跑在一个 uvicorn worker 内，请勿开多 worker。
- 下载按文件提供（weights/momentum 分开），未提供打包归档接口。
- 源文件在任务创建后被**同尺寸同步数**篡改无法识别（未做内容校验和）。
- `new_world_size > total_length` 会被拒绝（每个进程至少分到一个元素）。
- `/diagnose` 是同步全量扫描：超大检查点会让该请求长时间占用连接；
  逐元素比较用纯 Python（未引入 numpy），字节一致的窗口会整体跳过。
- 重分片持续繁忙时诊断读会长期让路（重分片优先是刻意设计）。
