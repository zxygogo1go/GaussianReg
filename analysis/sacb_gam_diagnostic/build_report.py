"""Build the portable SACB-Net vs GAM-SACB-Net diagnostic report artifact."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data"
GENERATED_AT = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nested(record, path):
    try:
        value = record
        for key in path:
            value = value[key]
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")


def bootstrap_ci(values, samples=10000, seed=2026):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))


baseline_eval = read_json(SOURCE / "baseline_evaluation.json")
gam_eval = read_json(SOURCE / "gam_evaluation_epoch403.json")
baseline_metrics = read_jsonl(SOURCE / "baseline_metrics.jsonl")
gam_metrics = read_jsonl(SOURCE / "gam_metrics.jsonl")

baseline_cases = {str(case["patient_id"]): case for case in baseline_eval["patients"]}
gam_cases = {str(case["patient_id"]): case for case in gam_eval["patients"]}
if set(baseline_cases) != set(gam_cases):
    raise RuntimeError("patient sets differ")

paths = {
    "ncc": ("image", "ncc_after"),
    "dice": ("segmentation_after", "mean_dice"),
    "dice_l1": ("segmentation_after", "dice_per_class", "1"),
    "dice_l2": ("segmentation_after", "dice_per_class", "2"),
    "hd95": ("segmentation_after", "mean_hd95"),
    "assd": ("segmentation_after", "mean_assd"),
    "fold": ("jacobian", "negative_jacobian_ratio"),
}

patient_rows = []
for patient_id in sorted(baseline_cases, key=int):
    baseline_case = baseline_cases[patient_id]
    gam_case = gam_cases[patient_id]
    row = {"patient_id": patient_id}
    for metric, path in paths.items():
        baseline_value = nested(baseline_case, path)
        gam_value = nested(gam_case, path)
        row["baseline_" + metric] = baseline_value if math.isfinite(baseline_value) else None
        row["gam_" + metric] = gam_value if math.isfinite(gam_value) else None
        row[metric + "_delta"] = (
            gam_value - baseline_value
            if math.isfinite(baseline_value) and math.isfinite(gam_value)
            else None
        )
    patient_rows.append(row)

dice_rows = [row for row in patient_rows if row["dice_delta"] is not None]
dice_delta = np.asarray([row["dice_delta"] for row in dice_rows], dtype=np.float64)
dice_ci = bootstrap_ci(dice_delta)
dice_without_94 = np.asarray(
    [row["dice_delta"] for row in dice_rows if row["patient_id"] != "94"],
    dtype=np.float64,
)

metric_definitions = [
    ("NCC", "ncc", 1),
    ("Dice", "dice", 1),
    ("Dice-L1", "dice_l1", 1),
    ("Dice-L2", "dice_l2", 1),
    ("HD95 (mm)", "hd95", -1),
    ("ASSD (mm)", "assd", -1),
    ("Negative Jacobian ratio", "fold", -1),
]
comparison_rows = []
for label, key, direction in metric_definitions:
    eligible = [
        row for row in patient_rows
        if row["baseline_" + key] is not None and row["gam_" + key] is not None
    ]
    base_values = np.asarray([row["baseline_" + key] for row in eligible])
    gam_values = np.asarray([row["gam_" + key] for row in eligible])
    benefit = direction * (gam_values - base_values)
    low, high = bootstrap_ci(benefit)
    comparison_rows.append(
        {
            "metric": label,
            "n": len(eligible),
            "baseline": float(base_values.mean()),
            "gam": float(gam_values.mean()),
            "gam_benefit": float(benefit.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "gam_wins": int((benefit > 0).sum()),
        }
    )

base_by_epoch = {int(row["epoch"]): row for row in baseline_metrics}
gam_by_epoch = {int(row["epoch"]): row for row in gam_metrics}
common_epochs = sorted(set(base_by_epoch) & set(gam_by_epoch))
curve_rows = []
ncc_curve_rows = []
dice_curve_rows = []
for epoch in common_epochs:
    base_val = base_by_epoch[epoch]["validation"]
    gam_val = gam_by_epoch[epoch]["validation"]
    curve_rows.append(
        {
            "epoch": epoch,
            "baseline_ncc": base_val.get("ncc_after"),
            "gam_ncc": gam_val.get("ncc_after"),
            "baseline_dice": base_val.get("mean_dice"),
            "gam_dice": gam_val.get("mean_dice"),
            "baseline_fold": base_val.get("negative_jacobian_ratio"),
            "gam_fold": gam_val.get("negative_jacobian_ratio"),
        }
    )
    ncc_curve_rows.extend(
        (
            {
                "epoch": epoch,
                "model": "Original SACB-Net",
                "ncc": base_val.get("ncc_after"),
            },
            {
                "epoch": epoch,
                "model": "GAM-SACB-Net",
                "ncc": gam_val.get("ncc_after"),
            },
        )
    )
    dice_curve_rows.extend(
        (
            {
                "epoch": epoch,
                "model": "Original SACB-Net",
                "dice": base_val.get("mean_dice"),
            },
            {
                "epoch": epoch,
                "model": "GAM-SACB-Net",
                "dice": gam_val.get("mean_dice"),
            },
        )
    )

gate_rows = []
gate_curve_rows = []
selected_gate_epochs = {1, 50, 100, 219, 300, 403, 441, 500}
for row in gam_metrics:
    epoch = int(row["epoch"])
    if epoch % 10 != 0 and epoch not in selected_gate_epochs:
        continue
    train = row["train"]
    gate_rows.append(
        {
            "epoch": epoch,
            "gate_l5": train.get("gate5"),
            "gate_l4": train.get("gate4"),
            "visibility_l5": train.get("gacm5_visibility"),
            "visibility_l4": train.get("gacm4_visibility"),
        }
    )
    gate_curve_rows.extend(
        (
            {"epoch": epoch, "mechanism": "Gate L5", "value": train.get("gate5")},
            {"epoch": epoch, "mechanism": "Gate L4", "value": train.get("gate4")},
            {
                "epoch": epoch,
                "mechanism": "Visibility L5",
                "value": train.get("gacm5_visibility"),
            },
            {
                "epoch": epoch,
                "mechanism": "Visibility L4",
                "value": train.get("gacm4_visibility"),
            },
        )
    )

def best_validation(rows, metric, maximize=True):
    valid = [
        (int(row["epoch"]), float(row["validation"][metric]))
        for row in rows
        if row.get("validation") and row["validation"].get(metric) is not None
    ]
    return (max if maximize else min)(valid, key=lambda item: item[1])


baseline_best_ncc = best_validation(baseline_metrics, "ncc_after")
gam_best_ncc = best_validation(gam_metrics, "ncc_after")
baseline_best_dice = best_validation(baseline_metrics, "mean_dice")
gam_best_dice = best_validation(gam_metrics, "mean_dice")

checkpoint_rows = []
for model_name, rows, epoch in (
    ("Original SACB-Net", baseline_metrics, baseline_best_ncc[0]),
    ("GAM-SACB-Net", gam_metrics, gam_best_ncc[0]),
):
    validation = next(row["validation"] for row in rows if int(row["epoch"]) == epoch)
    checkpoint_rows.append(
        {
            "model": model_name,
            "selected_epoch": epoch,
            "validation_ncc": validation["ncc_after"],
            "validation_dice": validation["mean_dice"],
            "validation_hd95": validation["mean_hd95"],
            "validation_assd": validation["mean_assd"],
            "validation_fold": validation["negative_jacobian_ratio"],
        }
    )

gam_epoch_441 = next(row["train"] for row in gam_metrics if int(row["epoch"]) == 441)
loss_weights = {
    "similarity": 1.0,
    "smoothness": 0.3,
    "deep_similarity": 1.0,
    "token": 0.01,
    "transport": 0.02,
    "anchor": 0.05,
}
loss_rows = [
    {
        "term": term,
        "raw_value": float(gam_epoch_441[term]),
        "weight": weight,
        "weighted_value": float(gam_epoch_441[term]) * weight,
    }
    for term, weight in loss_weights.items()
]

derived_sources = {
    "patient_deltas": patient_rows,
    "metric_comparison": comparison_rows,
    "validation_curves": curve_rows,
    "validation_ncc_long": ncc_curve_rows,
    "validation_dice_long": dice_curve_rows,
    "gate_diagnostics": gate_rows,
    "gate_diagnostics_long": gate_curve_rows,
    "checkpoint_comparison": checkpoint_rows,
    "loss_contributions": loss_rows,
}
with (SOURCE / "derived_diagnostics.json").open("w", encoding="utf-8") as handle:
    json.dump(derived_sources, handle, indent=2, ensure_ascii=False, allow_nan=False)

sql_queries = {
    "patient_deltas_source": (
        "SELECT * FROM patient_deltas "
        "WHERE dice_delta IS NOT NULL ORDER BY CAST(patient_id AS INTEGER)"
    ),
    "metric_comparison_source": "SELECT * FROM metric_comparison ORDER BY metric",
    "validation_ncc_source": "SELECT * FROM validation_ncc_long ORDER BY epoch, model",
    "validation_dice_source": "SELECT * FROM validation_dice_long ORDER BY epoch, model",
    "gate_diagnostics_source": "SELECT * FROM gate_diagnostics_long ORDER BY epoch, mechanism",
    "checkpoint_source": "SELECT * FROM checkpoint_comparison ORDER BY model",
    "loss_source": "SELECT * FROM loss_contributions ORDER BY term",
}


def sqlite_type(values):
    values = [value for value in values if value is not None]
    if values and all(isinstance(value, (bool, int)) for value in values):
        return "INTEGER"
    if values and all(isinstance(value, (bool, int, float)) for value in values):
        return "REAL"
    return "TEXT"


connection = sqlite3.connect(":memory:")
for dataset_name, rows in derived_sources.items():
    if not rows:
        continue
    fields = list(rows[0])
    definitions = ", ".join(
        '"%s" %s' % (field, sqlite_type([row.get(field) for row in rows]))
        for field in fields
    )
    connection.execute('CREATE TABLE "%s" (%s)' % (dataset_name, definitions))
    placeholders = ", ".join("?" for _ in fields)
    connection.executemany(
        'INSERT INTO "%s" VALUES (%s)' % (dataset_name, placeholders),
        [[row.get(field) for field in fields] for row in rows],
    )
for query in sql_queries.values():
    connection.execute(query).fetchall()
connection.close()


def source_spec(source_id, label, table, description, filters, definitions):
    return {
        "id": source_id,
        "label": label,
        "path": "source_data/derived_diagnostics.json",
        "query": {
            "engine": "SQLite",
            "sql": sql_queries[source_id],
            "description": description,
            "executed_at": GENERATED_AT,
            "language": "sql",
            "tables_used": [table],
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


test_filters = [
    "Same patient-disjoint HNTS-MRG24 test manifest",
    "Finite metric values in both models",
    "Response-aware labels",
    "GAM evaluation is the earlier pre-final-best checkpoint and is marked provisional",
]
training_filters = ["Epochs 1-500", "Seed 2026", "Same validation manifest"]
sources = [
    source_spec(
        "patient_deltas_source",
        "Patient-paired held-out test deltas",
        "patient_deltas",
        "Patient-paired metrics derived from the two evaluation JSON files.",
        test_filters,
        ["Dice delta = GAM Dice - original SACB-Net Dice"],
    ),
    source_spec(
        "metric_comparison_source",
        "Held-out test metric summary",
        "metric_comparison",
        "Metric means, paired bootstrap confidence intervals, and win counts.",
        test_filters,
        [
            "GAM benefit is sign-adjusted so positive always indicates better performance",
            "Confidence intervals use 10,000 paired patient bootstrap samples",
        ],
    ),
    source_spec(
        "validation_ncc_source",
        "Validation NCC curves",
        "validation_ncc_long",
        "Epoch-level validation NCC from both metrics.jsonl files.",
        training_filters,
        ["Best checkpoint is selected by maximum validation NCC"],
    ),
    source_spec(
        "validation_dice_source",
        "Validation Dice curves",
        "validation_dice_long",
        "Epoch-level response-aware validation Dice from both metrics.jsonl files.",
        training_filters,
        ["Validation labels are not used by the training objective"],
    ),
    source_spec(
        "gate_diagnostics_source",
        "GCDR gate and GACM visibility diagnostics",
        "gate_diagnostics_long",
        "Epoch-level GAM mechanism means sampled from metrics.jsonl.",
        training_filters,
        [
            "Gate is the GCDR Gaussian-flow mixture weight",
            "Visibility is the mean GACM token visibility",
        ],
    ),
    source_spec(
        "checkpoint_source",
        "Validation-NCC-selected checkpoint comparison",
        "checkpoint_comparison",
        "Validation metrics at each run's final maximum-NCC checkpoint.",
        training_filters,
        ["Best checkpoint is selected by maximum validation NCC"],
    ),
    source_spec(
        "loss_source",
        "GAM loss contribution diagnostics",
        "loss_contributions",
        "Raw and weighted training-loss values at GAM epoch 441.",
        ["GAM epoch 441", "Configured loss weights"],
        ["Weighted value = raw value multiplied by configured loss weight"],
    ),
]

charts = [
    {
        "id": "dice_delta_by_patient",
        "title": "逐患者 Dice 差值",
        "subtitle": "GAM-SACB-Net − 原始 SACB-Net；正值代表 GAM 更好，n=15",
        "intent": "comparison",
        "question": "Dice 改善是普遍存在，还是集中在少数病例？",
        "rationale": "逐患者差值柱图能显示胜负分布和异常贡献病例。",
        "comparisonContext": {
            "baseline": "Original SACB-Net",
            "grain": "patient",
            "unit": "Dice",
        },
        "type": "bar",
        "dataset": "patient_deltas",
        "sourceId": "patient_deltas_source",
        "encodings": {
            "x": {"field": "patient_id", "type": "nominal", "label": "患者"},
            "y": {"field": "dice_delta", "type": "quantitative", "label": "Dice 差值"},
        },
        "valueFormat": "number",
        "layout": "full",
        "palette": {"kind": "diverging", "name": "signed-delta", "midpoint": 0},
        "referenceLines": [
            {
                "axis": "y",
                "value": 0,
                "label": "无差异",
                "color": "neutral",
                "lineStyle": "dashed",
            }
        ],
        "settings": {"showValues": True, "sort": "ascending"},
        "surface": {"surface": "card", "viewMode": "both"},
    },
    {
        "id": "validation_ncc_curve",
        "title": "验证集 NCC 曲线",
        "subtitle": "相同划分、训练策略与 500 个 epoch",
        "intent": "trend",
        "question": "GAM 的 NCC 优势是否在训练中持续出现？",
        "rationale": "完整 epoch 曲线可区分稳定增益与单点波动。",
        "comparisonContext": {
            "baseline": "Original SACB-Net",
            "grain": "epoch",
            "unit": "NCC",
        },
        "type": "line",
        "dataset": "validation_ncc_long",
        "sourceId": "validation_ncc_source",
        "encodings": {
            "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
            "y": {"field": "ncc", "type": "quantitative", "label": "NCC"},
            "color": {"field": "model", "type": "nominal", "label": "模型"},
            "lineStyle": {"field": "model", "type": "nominal", "label": "模型"},
        },
        "valueFormat": "number",
        "layout": "full",
        "palette": {"kind": "identity", "name": "model-identity"},
        "legend": {"position": "bottom", "sort": "spec", "title": "模型"},
        "labels": {"values": "endpoints"},
        "surface": {"surface": "card", "viewMode": "both"},
    },
    {
        "id": "validation_dice_curve",
        "title": "验证集 Dice 曲线",
        "subtitle": "标签仅用于验证，不参与训练",
        "intent": "trend",
        "question": "NCC 继续提高时，Dice 是否同步提高？",
        "rationale": "并列模型曲线揭示优化目标与解剖指标的分离。",
        "comparisonContext": {
            "baseline": "Original SACB-Net",
            "grain": "epoch",
            "unit": "Dice",
        },
        "type": "line",
        "dataset": "validation_dice_long",
        "sourceId": "validation_dice_source",
        "encodings": {
            "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
            "y": {"field": "dice", "type": "quantitative", "label": "Dice"},
            "color": {"field": "model", "type": "nominal", "label": "模型"},
            "lineStyle": {"field": "model", "type": "nominal", "label": "模型"},
        },
        "valueFormat": "number",
        "layout": "full",
        "palette": {"kind": "identity", "name": "model-identity"},
        "legend": {"position": "bottom", "sort": "spec", "title": "模型"},
        "labels": {"values": "endpoints"},
        "surface": {"surface": "card", "viewMode": "both"},
    },
    {
        "id": "gate_visibility_curve",
        "title": "GCDR 门控与 GACM visibility",
        "subtitle": "训练均值；1 表示接近完全采用 Gaussian flow 或所有 token 可见",
        "intent": "trend",
        "question": "自适应门控和可见性机制是否保持有效动态范围？",
        "rationale": "多序列曲线直接展示两个机制是否发生饱和。",
        "comparisonContext": {
            "grain": "epoch",
            "normalization": "mean over training samples",
            "unit": "ratio",
        },
        "type": "line",
        "dataset": "gate_diagnostics_long",
        "sourceId": "gate_diagnostics_source",
        "encodings": {
            "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
            "y": {"field": "value", "type": "quantitative", "label": "均值"},
            "color": {"field": "mechanism", "type": "nominal", "label": "诊断量"},
            "lineStyle": {"field": "mechanism", "type": "nominal", "label": "诊断量"},
        },
        "valueFormat": "number",
        "layout": "full",
        "palette": {"kind": "identity", "name": "mechanism-identity"},
        "legend": {"position": "bottom", "sort": "spec", "title": "诊断量"},
        "referenceLines": [
            {
                "axis": "y",
                "value": 1,
                "label": "饱和",
                "color": "neutral",
                "lineStyle": "dashed",
            }
        ],
        "surface": {"surface": "card", "viewMode": "both"},
    },
]

tables = [
    {
        "id": "test_metric_table",
        "title": "测试集汇总指标",
        "subtitle": "旧 GAM 测试 checkpoint 与最终原始 SACB-Net；GAM benefit 正值代表更好",
        "dataset": "metric_comparison",
        "sourceId": "metric_comparison_source",
        "defaultSort": {"field": "metric", "direction": "asc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "metric", "label": "指标", "type": "text"},
            {"field": "n", "label": "n", "format": "number"},
            {"field": "baseline", "label": "原始 SACB", "format": "number"},
            {"field": "gam", "label": "GAM-SACB", "format": "number"},
            {
                "field": "gam_benefit",
                "label": "GAM benefit",
                "format": "number",
                "movement": True,
            },
            {"field": "ci95_low", "label": "CI low", "format": "number"},
            {"field": "ci95_high", "label": "CI high", "format": "number"},
            {"field": "gam_wins", "label": "GAM 胜例", "format": "number"},
        ],
    },
    {
        "id": "checkpoint_table",
        "title": "验证 NCC 所选最终 checkpoint",
        "subtitle": "两个模型均完成 500 epoch，checkpoint 规则相同",
        "dataset": "checkpoint_comparison",
        "sourceId": "checkpoint_source",
        "defaultSort": {"field": "model", "direction": "asc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "model", "label": "模型", "type": "text"},
            {"field": "selected_epoch", "label": "Epoch", "format": "number"},
            {"field": "validation_ncc", "label": "Val NCC", "format": "number"},
            {"field": "validation_dice", "label": "Val Dice", "format": "number"},
            {"field": "validation_hd95", "label": "Val HD95", "format": "number"},
            {"field": "validation_assd", "label": "Val ASSD", "format": "number"},
            {"field": "validation_fold", "label": "Val Fold", "format": "number"},
        ],
    },
    {
        "id": "loss_weight_table",
        "title": "GAM epoch 441 训练损失贡献",
        "subtitle": "raw value × configured weight；几何专用正则项贡献很小",
        "dataset": "loss_contributions",
        "sourceId": "loss_source",
        "defaultSort": {"field": "term", "direction": "asc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "term", "label": "损失项", "type": "text"},
            {"field": "raw_value", "label": "Raw", "format": "number"},
            {"field": "weight", "label": "Weight", "format": "number"},
            {"field": "weighted_value", "label": "Weighted", "format": "number"},
        ],
    },
]

blocks = [
    {
        "id": "title",
        "type": "markdown",
        "body": "# SACB-Net 与 GAM-SACB-Net 性能差异诊断",
        "layout": "full",
    },
    {
        "id": "technical_summary",
        "type": "markdown",
        "body": (
            "## 技术结论：模块提高了图像相似性，但没有形成稳定的解剖重叠增益\n\n"
            f"- **当前测试表不是最终公平对比。** GAM 测试结果生成于最终 epoch {gam_best_ncc[0]} "
            "checkpoint 出现之前，必须用最终 `best_validation_ncc.pt` 重新评估。\n"
            f"- **已有测试结果中，Dice 平均只增加 {dice_delta.mean():.4f}，"
            f"95% CI [{dice_ci[0]:.4f}, {dice_ci[1]:.4f}]，仅 5/15 病例获胜。** "
            f"中位数为 {np.median(dice_delta):.4f}；去掉贡献最大的病例 94 后，"
            f"平均差值变为 {dice_without_94.mean():.4f}。\n"
            "- **NCC 增益稳定而形变拓扑变差。** 这说明模型更擅长追逐强度匹配，"
            "但没有把额外自由度可靠转化为肿瘤/靶区边界对齐。\n"
            "- **最强机制线索是门控与 visibility 饱和。** GCDR 从偏向原 dense flow "
            "快速转为几乎完全采用 Gaussian flow，visibility 也接近全 1，"
            "使原设想中的自适应融合与治疗响应可见性失去区分能力。"
        ),
        "layout": "full",
    },
    {
        "id": "dice_only_decision",
        "type": "markdown",
        "body": (
            "## Dice-only 决策：当前完整方案应判定为 no-go\n\n"
            "如果唯一成功标准是 Dice，那么当前 GACM+GCDR 完整组合没有达到继续扩展的门槛，"
            "不应再通过增加新模块掩盖负结果。建议只保留一次拆分消融机会：分别验证 "
            "GACM-only，以及 GACM+GCDR 但仅使用共同损失的版本。由于 GCDR 依赖 GACM "
            "产生的 Gaussian flow 和 context，不能构造严格独立的 GCDR-only。"
            "预先规定继续门槛为验证集平均 "
            "Dice 至少提高 0.01、患者级 bootstrap 95% CI 不跨 0，并且多数可评估患者获胜。"
            "若没有任何单模块满足门槛，就停止该方向；若只有一个模块满足，则保留该模块并删除"
            "或重构另一个模块。"
        ),
        "layout": "full",
    },
    {
        "id": "dice_distribution_intro",
        "type": "markdown",
        "body": (
            "## Dice 均值被单个病例拉高，提升不是普遍现象\n\n"
            "病例 94 的 Dice 差值约为 +0.1263，超过总体净增益；多数患者的差值接近 0 "
            "或为负。因而当前 +0.0068 更像少数病例收益与多数病例轻微退化的混合，"
            "不是可稳定复现的结构性提升。"
        ),
        "layout": "full",
        "sourceId": "patient_deltas_source",
    },
    {"id": "dice_delta_chart", "type": "chart", "chartId": "dice_delta_by_patient", "layout": "full"},
    {
        "id": "test_table_intro",
        "type": "markdown",
        "body": (
            "## 指标方向出现分裂：NCC 更好，但拓扑代价明确\n\n"
            "现有测试中 NCC 对全部 16 个患者都有提升，而 Dice、HD95 和 ASSD 均未形成"
            "稳定优势；负 Jacobian 比例则显著恶化。这个组合与“更激进的强度配准”一致。"
        ),
        "layout": "full",
        "sourceId": "metric_comparison_source",
    },
    {"id": "test_metric_table_block", "type": "table", "tableId": "test_metric_table", "layout": "full"},
    {
        "id": "checkpoint_mismatch",
        "type": "markdown",
        "body": (
            "## 先修正 checkpoint 时间错配，再下最终结论\n\n"
            f"原始 SACB-Net 的最终验证 NCC 最优点为 epoch {baseline_best_ncc[0]}；"
            f"GAM 的最终最优点为 epoch {gam_best_ncc[0]}。当前 GAM 测试 JSON 是最终"
            "最优点生成之前的结果，因此应新建输出目录重新评估，不能覆盖旧结果。"
        ),
        "layout": "full",
        "sourceId": "checkpoint_source",
    },
    {"id": "checkpoint_table_block", "type": "table", "tableId": "checkpoint_table", "layout": "full"},
    {
        "id": "ncc_curve_intro",
        "type": "markdown",
        "body": (
            "## 训练目标持续改善 NCC，却没有同步改善 Dice\n\n"
            "GAM 的验证 NCC 后期高于基线，证明新增模块确实参与并优化了图像匹配；"
            "但对应 Dice 曲线没有同向拉开，说明 NCC 与解剖边界重叠在该纵向头颈任务上并不等价。"
        ),
        "layout": "full",
        "sourceId": "validation_ncc_source",
    },
    {"id": "ncc_curve_chart", "type": "chart", "chartId": "validation_ncc_curve", "layout": "full"},
    {
        "id": "dice_curve_intro",
        "type": "markdown",
        "body": (
            f"## 即使按验证 Dice 观察，GAM 也未超过基线峰值\n\n"
            f"基线验证 Dice 峰值为 {baseline_best_dice[1]:.4f}（epoch {baseline_best_dice[0]}），"
            f"GAM 峰值为 {gam_best_dice[1]:.4f}（epoch {gam_best_dice[0]}）。"
            "这不是单纯由验证 NCC checkpoint 规则造成的；当前结构和损失组合本身没有展示"
            "明确的 Dice 上限提升。"
        ),
        "layout": "full",
        "sourceId": "validation_dice_source",
    },
    {"id": "dice_curve_chart", "type": "chart", "chartId": "validation_dice_curve", "layout": "full"},
    {
        "id": "mechanism_intro",
        "type": "markdown",
        "body": (
            "## 门控和可见性机制发生饱和，削弱了模块原本的设计意图\n\n"
            "GCDR 门控初始约 0.02，本应在 dense flow 与 Gaussian flow 间按局部可靠性自适应选择；"
            "训练后 L5/L4 均值约为 0.99/0.92，接近长期采用 Gaussian flow。"
            "GACM visibility 也接近 1，意味着几乎所有 token 都被视为可见，"
            "难以表达放疗前后肿瘤缩小、消失或新出现的对应不确定性。"
        ),
        "layout": "full",
        "sourceId": "gate_diagnostics_source",
    },
    {"id": "gate_chart", "type": "chart", "chartId": "gate_visibility_curve", "layout": "full"},
    {
        "id": "loss_alignment",
        "type": "markdown",
        "body": (
            "## 几何专用约束相对过弱，模型主要受多尺度 NCC 驱动\n\n"
            "在 epoch 441，token、transport 和 anchor 三项的加权贡献合计远小于"
            "similarity 与 deep similarity；目标中也没有 Jacobian 或逆一致性惩罚。"
            "因此额外形变自由度可以通过局部压缩/折叠换取 NCC，而不会受到直接拓扑约束。"
        ),
        "layout": "full",
        "sourceId": "loss_source",
    },
    {"id": "loss_table_block", "type": "table", "tableId": "loss_weight_table", "layout": "full"},
    {
        "id": "scope_definitions",
        "type": "markdown",
        "body": (
            "## 比较范围与指标定义\n\n"
            "- 人群：同一 patient-disjoint HNTS-MRG24 测试集，16 位患者。\n"
            "- Dice：response-aware；仅统计两个时间点均有有效标签的病例，因此 mean Dice n=15。\n"
            "- NCC：全图局部归一化互相关，越高越好。\n"
            "- HD95/ASSD：毫米，越低越好。\n"
            "- Fold ratio：负 Jacobian 行列式体素比例，越低越好。\n"
            "- 训练：两者都不使用解剖标签，采用相同数据划分、优化器、日程和验证 NCC checkpoint 规则。"
        ),
        "layout": "full",
    },
    {
        "id": "methodology",
        "type": "markdown",
        "body": (
            "## 诊断方法\n\n"
            "测试比较使用相同患者的配对差值，置信区间通过 10,000 次患者级 bootstrap 计算；"
            "训练诊断复现两个 run 的 500 个 epoch 验证曲线，并检查 checkpoint 元数据、"
            "GCDR gate、GACM visibility 和各损失项的实际加权贡献。"
        ),
        "layout": "full",
    },
    {
        "id": "limitations",
        "type": "markdown",
        "body": (
            "## 不确定性与当前结论边界\n\n"
            "- 当前测试 GAM 结果来自较早 checkpoint，最终数值必须重新评估后更新。\n"
            "- 只有一个随机种子，不能区分小幅架构效应与训练随机性。\n"
            "- 未做 GACM-only、GCDR-only 和共同损失消融，因此不能把退化单独归因于某个模块。\n"
            "- 测试集较小，且标签可评估病例数随类别变化；不过 Dice 效应量很小且中位数为负，"
            "问题并非仅仅是统计功效不足。"
        ),
        "layout": "full",
    },
    {
        "id": "next_steps",
        "type": "markdown",
        "body": (
            "## 推荐的下一步实验顺序\n\n"
            "1. 用最终 epoch 441 的 `best_validation_ncc.pt` 重新评估 GAM，并保留新的输出目录。\n"
            "2. 固定相同 seed，补两组训练：SACB+GACM（共同损失）与 "
            "SACB+GACM+GCDR（共同损失）；与已有 SACB 和完整辅助损失结果构成递进消融。\n"
            "3. 对完整模型增加 gate/visibility 防饱和约束，并加入 Jacobian 或逆一致性正则，"
            "优先修复 Fold ratio，再观察 Dice 是否同步改善。\n"
            "4. 最终候选配置运行至少 3 个随机种子，报告患者配对结果与跨 seed 均值±标准差。"
        ),
        "layout": "full",
    },
    {
        "id": "further_questions",
        "type": "markdown",
        "body": (
            "## 仍需回答的问题\n\n"
            "- Dice 退化主要来自 GACM 的对应错误，还是 GCDR 的 Gaussian flow 过度主导？\n"
            "- visibility 饱和是损失权重过小、参数化下限不合适，还是纵向响应数据缺少可辨识信号？\n"
            "- 在不使用训练标签的前提下，能否通过边缘、互信息或局部结构一致性让优化目标更贴近肿瘤边界？"
        ),
        "layout": "full",
    },
]

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "SACB-Net 与 GAM-SACB-Net 性能差异诊断",
        "description": "HNTS-MRG24 配准实验的逐患者、训练曲线与机制诊断。",
        "generatedAt": GENERATED_AT,
        "sources": sources,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
    },
    "snapshot": {
        "version": 1,
        "generatedAt": GENERATED_AT,
        "status": "ready",
        "datasets": derived_sources,
    },
    "sources": sources,
}

with (ROOT / "artifact.json").open("w", encoding="utf-8") as handle:
    json.dump(artifact, handle, indent=2, ensure_ascii=False, allow_nan=False)

print(ROOT / "artifact.json")
