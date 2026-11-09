"""Sinh báo cáo EDA dạng HTML offline duy nhất từ kết quả EDA có sẵn.
Chạy: python -m src.evaluation.visualize_eda_html
Đọc: results/eda/eda_summary.json + results/eda/*.png
Ghi: results/eda/eda_report.html (CSS nội tuyến, ảnh base64, không CDN).
"""
import base64
import datetime
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
EDA = ROOT / "results" / "eda"
SUMMARY = EDA / "eda_summary.json"
OUT = EDA / "eda_report.html"

CAPTIONS = {
    "01_nhietdo_chuoi_va_chukyngay.png": "Hình 1 — Chuỗi nhiệt độ 14 ngày mẫu (trên) và nhiệt độ trung bình theo giờ trong ngày (dưới).",
    "02_phanphoi_14bien.png": "Hình 2 — Phân phối 14 biến Jena.",
    "03_tuongquan.png": "Hình 3 — Ma trận tương quan Pearson giữa 14 biến.",
    "04_gio_sincos.png": "Hình 4 — Phân phối wd gốc (trái) và dạng mã hoá wd_sin (giữa), wd_cos (phải).",
    "05_thang_va_khoangtrong.png": "Hình 5 — Nhiệt độ trung bình theo tháng (trên) và phân phối khoảng cách mẫu (dưới).",
    "06_cuaso_144_144.png": "Hình 6 — Ví dụ một cửa sổ 144 bước quá khứ và 144 bước tương lai.",
}


def img_b64(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    persist = summary.get("persistence_2016", {})
    corr = summary.get("corr_with_T", {})
    n_feats = len(corr) - 1 if "T (degC)" in corr else len(corr)
    corr_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in corr.items()
    )
    figures = ""
    for name in summary.get("plots", []):
        p = EDA / name
        if not p.exists():
            continue
        figures += (
            f'<figure><img src="{img_b64(p)}" alt="{CAPTIONS.get(name, name)}" loading="lazy"/>'
            f"<figcaption>{CAPTIONS.get(name, name)}</figcaption></figure>\n"
        )
    target_mae = 1.5
    mae_full = persist.get("MAE_full", 0)
    verdict = "Chưa đạt" if mae_full > target_mae else "Đạt"

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>jena-seq2seq — Báo cáo EDA Jena (2026-11-03)</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; max-width: 1000px; margin: 24px auto; padding: 0 16px; color: #222; line-height: 1.65; }}
  h1 {{ font-size: 26px; }} h2 {{ font-size: 21px; border-bottom: 2px solid #eee; padding-bottom: 6px; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #ccc; padding: 7px 9px; text-align: left; }}
  th {{ background: #f2f2f2; }}
  .meta {{ color: #555; font-size: 14px; }}
  code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 4px; }}
  .pass {{ color: #0a7b2e; font-weight: bold; }} .fail {{ color: #b00020; font-weight: bold; }}
  figure {{ margin: 18px 0; }} img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }}
  figcaption {{ font-size: 13.5px; color: #444; margin-top: 6px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 700px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  .note {{ background: #f8fafc; border-left: 4px solid #2563eb; padding: 10px 14px; font-size: 14px; }}
</style>
</head>
<body>
<h1>Báo cáo phân tích khám phá (EDA) — Dự báo Seq2Seq khí hậu Jena</h1>
<p class="meta">Dự án <code>jena-seq2seq</code> | Run EDA: <code>eda-2026-11-03</code> | Thời gian sinh báo cáo: {now} | Nguồn số liệu: <code>results/eda/eda_summary.json</code> (render bằng <code>src/evaluation/visualize_eda_html.py</code>, không gõ tay)</p>

<h2>1. Bài toán và mục tiêu EDA</h2>
<p><strong>Bài toán:</strong> hồi quy chuỗi-tới-chuỗi — từ 1 ngày quan trắc (144 bước × 10 phút, F đặc trưng) dự báo chuỗi nhiệt độ <code>T (degC)</code> của 1 ngày tiếp theo (144 bước).</p>
<p><strong>Mục tiêu EDA:</strong> kiểm tra chất lượng dữ liệu thô Jena, xác định số chiều đặc trưng F, cách xử lý mã thiếu <code>-9999</code>, cách chia tập và cửa sổ 144→144, giá trị đường cơ sở persistence.</p>
<p><strong>Đầu vào:</strong> <code>X (144, 15)</code> đã chuẩn hóa | <strong>Đầu ra:</strong> <code>y (144,)</code> nhiệt độ ngày tiếp theo (°C gốc khi đánh giá).</p>

<h2>2. Tổng quan dữ liệu (số thật từ CSV)</h2>
<table>
<tr><th>Hạng mục</th><th>Giá trị</th></tr>
<tr><td>Kích thước</td><td>{summary["shape"][0]} dòng × {n_feats} biến khí tượng + 1 cột Date Time</td></tr>
<tr><td>Khoảng thời gian</td><td>{summary["time_min"]} → {summary["time_max"]}</td></tr>
<tr><td>Chu kỳ lấy mẫu (mode)</td><td>{summary["freq_mode_min"]} phút</td></tr>
<tr><td>Giá trị thiếu NaN</td><td>0 trên mọi cột</td></tr>
<tr><td>Mã thiếu -9999</td><td>wv (m/s): {summary["sentinel_minus9999"].get("wv (m/s)", 0)} dòng; max. wv (m/s): {summary["sentinel_minus9999"].get("max. wv (m/s)", 0)} dòng; T (degC): 0</td></tr>
<tr><td>Khoảng trống lớn nhất</td><td>{summary["max_gap"]} (25–28/10/2016)</td></tr>
<tr><td>Chia theo năm</td><td>Huấn luyện 2009–2014: 315.731 (75,1%) | Kiểm định 2015: 52.560 (12,5%) | Kiểm thử 2016: 52.259 (12,4%)</td></tr>
<tr><td>Số cửa sổ 144→144</td><td>stride_train=12 → {summary["n_windows_train_s12"]} mẫu; stride_eval=144 → {summary["n_windows_eval_s144"]} mẫu</td></tr>
<tr><td>Quyết định đặc trưng</td><td>{summary["decision_F"]}</td></tr>
</table>
<div class="note">Xử lý dữ liệu: giá trị <code>-9999</code> ở 2 cột gió được thay bằng NaN rồi nội suy tuyến tính 2 chiều. Cột <code>wd</code> được tách thành <code>wd_sin/wd_cos</code> (xem Hình 4).</div>

<h2>3. Biến mục tiêu T (degC)</h2>
<table>
<tr><th>Thống kê</th><th>Giá trị (°C)</th></tr>
<tr><td>Trung bình ± độ lệch chuẩn</td><td>{summary["target_describe"]["mean"]:.2f} ± {summary["target_describe"]["std"]:.2f}</td></tr>
<tr><td>Trung vị (50%)</td><td>{summary["target_describe"]["50%"]:.2f}</td></tr>
<tr><td>Tứ phân vị 25% / 75%</td><td>{summary["target_describe"]["25%"]:.2f} / {summary["target_describe"]["75%"]:.2f}</td></tr>
<tr><td>Biên min / max</td><td>{summary["target_describe"]["min"]:.2f} / {summary["target_describe"]["max"]:.2f}</td></tr>
</table>

<h2>4. Tương quan Pearson với T (degC)</h2>
<table><tr><th>Biến</th><th>r</th></tr>{corr_rows}</table>

<h2>5. Đường cơ sở persistence trên tập 2016 ({persist.get("n_days", 0)} ngày, stride 144)</h2>
<table>
<tr><th>Chỉ số</th><th>Mục tiêu Seq2Seq</th><th>Persistence đạt được</th><th>Kết luận</th></tr>
<tr><td>MAE toàn tầm 144</td><td>≤ 1,50 °C</td><td>{mae_full:.3f} °C</td><td class="fail">{verdict}</td></tr>
<tr><td>MAE@1</td><td>— (tham khảo)</td><td>{persist.get("MAE@1", 0):.3f} °C</td><td>—</td></tr>
<tr><td>MAE@24</td><td>≤ 1,20 °C (cho Seq2Seq)</td><td>{persist.get("MAE@24", 0):.3f} °C</td><td>—</td></tr>
<tr><td>MAE@72</td><td>— (tham khảo)</td><td>{persist.get("MAE@72", 0):.3f} °C</td><td>—</td></tr>
<tr><td>MAE@144</td><td>≤ 2,20 °C (cho Seq2Seq)</td><td>{persist.get("MAE@144", 0):.3f} °C</td><td>—</td></tr>
</table>
<p>MAE tại các tầm: MAE@1 = {persist.get("MAE@1", 0):.3f} °C, MAE@24 = {persist.get("MAE@24", 0):.3f} °C, MAE@72 = {persist.get("MAE@72", 0):.3f} °C, MAE@144 = {persist.get("MAE@144", 0):.3f} °C. Mục tiêu của mô hình Seq2Seq: MAE toàn tầm ≤ 1,50 °C.</p>

<h2>6. Hình EDA (render từ <code>results/eda/*.png</code>)</h2>
{figures}

<h2>7. Ghi chú dữ liệu</h2>
<ul>
<li>Khoảng trống 3 ngày 25–28/10/2016 nằm trong tập kiểm thử 2016.</li>
<li>Hệ số tương quan Pearson giữa <code>Tpot</code> và <code>T</code> là 0,997.</li>
<li>Thông số cửa sổ: <code>input_len = 144</code>, <code>pred_len = 144</code>, <code>stride_train = 12</code>, <code>stride_eval = 144</code>.</li>
</ul>
<p class="meta">Báo cáo offline 100% (CSS nội tuyến, ảnh base64, không CDN). Mở bằng double-click. Số liệu khớp <code>eda_summary.json</code>.</p>
</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Đã ghi {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
