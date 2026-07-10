# τ 网格搜索标定报告 — 刚柔分层阈值敏感性

时间: 2026-07-09T12:05:33.284047

τ 训练值: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
τ 诊断值: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]

## 训练 τ 扫描 (体制D, 300故事 × 8 epoch)

| τ_train | rf_loss | phys_curv | flex_pos | flex_neg | H-rigid | H-flex | flex_SNR |
|---------|---------|-----------|----------|----------|---------|--------|----------|
| 0.1 | 0.1613 | 0.0395 | 0.0392 | 0.0397 | **SUPPORT** | **INCONCLUSIVE** | 0.011 |
| 0.2 | 0.1631 | 0.0410 | 0.0414 | 0.0412 | **SUPPORT** | **INCONCLUSIVE** | 0.004 |
| 0.3 | 0.1616 | 0.0358 | 0.0362 | 0.0366 | **SUPPORT** | **INCONCLUSIVE** | 0.009 |
| 0.4 | 0.1607 | 0.0418 | 0.0420 | 0.0420 | **SUPPORT** | **INCONCLUSIVE** | 0.001 |
| 0.5 | 0.1610 | 0.0370 | 0.0369 | 0.0370 | **SUPPORT** | **INCONCLUSIVE** | 0.003 |
| 0.6 | 0.1632 | 0.0406 | 0.0403 | 0.0400 | **SUPPORT** | **INCONCLUSIVE** | 0.007 |
| 0.7 | 0.1756 | 0.0398 | 0.0398 | 0.0399 | **SUPPORT** | **INCONCLUSIVE** | 0.003 |
| 0.8 | 0.1746 | 0.0433 | 0.0426 | 0.0423 | **SUPPORT** | **INCONCLUSIVE** | 0.009 |
| 1.0 | 0.1799 | 0.0418 | 0.0418 | 0.0424 | **SUPPORT** | **INCONCLUSIVE** | 0.013 |
| 1.5 | 0.1800 | 0.0435 | 0.0435 | 0.0438 | **SUPPORT** | **INCONCLUSIVE** | 0.008 |
| 2.0 | 0.1967 | 0.0387 | 0.0387 | 0.0390 | **SUPPORT** | **INCONCLUSIVE** | 0.007 |

## 品质因数 (Figure of Merit)
- 总扫描: 11 个 τ 值
- H-rigid SUPPORT: 11 ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0])
- H-flex SUPPORT: 0 ([])
- SNR 中位: 0.0070208374168519005

## 最佳推荐
- τ_train = **1.0**
- H-rigid: SUPPORT (phys_curv=0.0418)
- H-flex: INCONCLUSIVE (flex+=0.0418 / f-=0.0424)

### 对最佳模型 (τ_train=1.0) 的诊断阈值后扫描
| τ_diag | H-rigid | H-flex | phys | flex+ | flex- |
|--------|---------|--------|------|-------|-------|
| 0.1 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 0.2 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 0.3 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 0.4 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 0.5 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 0.6 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 0.7 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 0.8 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 1.0 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 1.5 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |
| 2.0 | SUPPORT | INCONCLUSIVE | 0.0418 | 0.0418 | 0.0424 |