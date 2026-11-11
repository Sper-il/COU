"""
Module xây dựng báo cáo HTML offline cho kết quả đánh giá mô hình Seq2Seq Jena Climate.
Tuân thủ API: build_report(run_dir: str) -> str
- Tự chứa (CSS nội tuyến, ảnh Base64, không CDN/Internet).
- Kiểm tra ngưỡng ĐẠT / KHÔNG ĐẠT theo tiêu chí tại docs/description.md.
"""

import os
import json
import base64
from pathlib import Path


def _image_to_base64(img_path: Path) -> str:
    """Đọc ảnh và chuyển thành chuỗi Base64 Data URL."""
    if not img_path.exists():
        return ""
    ext = img_path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else f"image/{ext}"
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _get_status_badge(val: float, threshold: float, lower_is_better: bool = True) -> str:
    """Trả về thẻ HTML badge đánh giá ĐẠT / KHÔNG ĐẠT."""
    if val is None:
        return '<span class="badge badge-na">N/A</span>'
    passed = (val <= threshold) if lower_is_better else (val >= threshold)
    if passed:
        return '<span class="badge badge-pass">🟢 ĐẠT</span>'
    return '<span class="badge badge-fail">🔴 KHÔNG ĐẠT</span>'


def build_report(run_dir: str) -> str:
    """
    Đọc dữ liệu từ run_dir và tạo file report.html.
    
    Args:
        run_dir: Đường dẫn đến thư mục lượt chạy (chứa metrics.json, config.yaml, plots/)
        
    Returns:
        Đường dẫn tuyệt đối đến file report.html vừa tạo.
    """
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {run_path}")

    # 1. Đọc metrics.json
    metrics_file = run_path / "metrics.json"
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    # 2. Đọc config.yaml (dưới dạng raw text nếu không có thư viện yaml)
    config_file = run_path / "config.yaml"
    config_content = ""
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config_content = f.read()

    # 3. Quét và mã hóa Base64 các biểu đồ trong thư mục plots
    plots_dir = run_path / "plots"
    plot_names = [
        ("loss.png", "Hàm mất mát qua các Epoch (Loss Curve)"),
        ("mae_horizon.png", "Sai số MAE theo tầm dự báo (1, 24, 72, 144 bước)"),
        ("best_prediction.png", "Ví dụ dự báo tốt nhất (Best Prediction)"),
        ("worst_prediction.png", "Ví dụ dự báo xấu nhất (Worst Prediction)"),
        ("worst_case.png", "Phân tích trường hợp biên / sai lệch lớn (Worst Case)")
    ]
    encoded_plots = {}
    for fname, desc in plot_names:
        img_path = plots_dir / fname
        encoded_plots[fname] = {
            "title": desc,
            "b64": _image_to_base64(img_path),
            "exists": img_path.exists()
        }

    # 4. Trích xuất chỉ số chính & so sánh ngưỡng (description.md)
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    mae_h1 = metrics.get("mae_h1")
    mae_h24 = metrics.get("mae_h24")
    mae_h72 = metrics.get("mae_h72")
    mae_h144 = metrics.get("mae_h144")
    pers_mae = metrics.get("persistence_mae")
    latency = metrics.get("inference_latency_ms")

    # Kiểm tra Persistence
    pers_passed = (mae < pers_mae) if (mae is not None and pers_mae is not None) else None
    pers_badge = '<span class="badge badge-pass">🟢 ĐẠT (Vượt Baseline)</span>' if pers_passed else '<span class="badge badge-fail">🔴 KHÔNG ĐẠT</span>'

    # Bảng so sánh 3 mô hình nếu có trong metrics
    comparisons = metrics.get("comparison", [])
    comp_rows = ""
    for item in comparisons:
        comp_rows += f"""
        <tr>
            <td><strong>{item.get('model', '-')}</strong></td>
            <td>{item.get('mae', '-')} °C</td>
            <td>{item.get('rmse', '-')} °C</td>
            <td>{item.get('mae_h24', '-')} °C</td>
            <td>{item.get('mae_h144', '-')} °C</td>
        </tr>
        """

    # 5. Tạo nội dung HTML
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo đánh giá mô hình Seq2Seq Jena Climate</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --success: #15803d;
            --danger: #b91c1c;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        h1 {{ margin: 0 0 8px 0; font-size: 24px; color: var(--primary); }}
        .meta {{ color: var(--text-muted); font-size: 14px; }}
        .card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        h2 {{ font-size: 18px; margin-top: 0; margin-bottom: 16px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: #f1f5f9; color: var(--text-main); font-weight: 600; }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-pass {{ background: #dcfce7; color: var(--success); }}
        .badge-fail {{ background: #fee2e2; color: var(--danger); }}
        .badge-na {{ background: #f1f5f9; color: var(--text-muted); }}
        .grid-plots {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 20px;
        }}
        .plot-box {{
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            background: #ffffff;
            text-align: center;
        }}
        .plot-box h3 {{ font-size: 14px; margin: 0 0 10px 0; color: var(--text-muted); text-align: left; }}
        .plot-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .no-img {{ padding: 40px 0; color: var(--text-muted); font-style: italic; background: #f8fafc; }}
        pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 14px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 1. Header & Thông tin chung -->
        <div class="header">
            <h1>Báo cáo Đánh giá Mô hình Jena-Seq2Seq</h1>
            <div class="meta">
                <strong>Thư mục:</strong> {run_path.name} | 
                <strong>Mô hình:</strong> {metrics.get('model_name', 'LSTM S2S')} |
                <strong>Độ trễ suy luận (CPU):</strong> {f"{latency} ms/mẫu (≤ 100 ms: " + ('🟢 ĐẠT' if latency and latency <= 100 else '🔴 KHÔNG ĐẠT') + ")" if latency is not None else "N/A"}
            </div>
        </div>

        <!-- 2. Bảng chỉ số đánh giá & Tiêu chuẩn nghiệm thu -->
        <div class="card">
            <h2>1. Chỉ số Đánh giá & Tiêu chuẩn Nghiệm thu (Tập Test 2016, Tầm 144)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Chỉ số</th>
                        <th>Giá trị thực nghiệm</th>
                        <th>Ngưỡng yêu cầu (description.md)</th>
                        <th>Trạng thái</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>MAE toàn tầm</strong></td>
                        <td>{f"{mae:.3f} °C" if mae is not None else "N/A"}</td>
                        <td>≤ 1.50 °C</td>
                        <td>{_get_status_badge(mae, 1.50)}</td>
                    </tr>
                    <tr>
                        <td><strong>RMSE toàn tầm</strong></td>
                        <td>{f"{rmse:.3f} °C" if rmse is not None else "N/A"}</td>
                        <td>≤ 2.00 °C</td>
                        <td>{_get_status_badge(rmse, 2.00)}</td>
                    </tr>
                    <tr>
                        <td><strong>MAE @ Step 1</strong></td>
                        <td>{f"{mae_h1:.3f} °C" if mae_h1 is not None else "N/A"}</td>
                        <td>Tham khảo (khởi đầu)</td>
                        <td><span class="badge badge-na">Tham khảo</span></td>
                    </tr>
                    <tr>
                        <td><strong>MAE @ Step 24 (4 giờ)</strong></td>
                        <td>{f"{mae_h24:.3f} °C" if mae_h24 is not None else "N/A"}</td>
                        <td>≤ 1.20 °C</td>
                        <td>{_get_status_badge(mae_h24, 1.20)}</td>
                    </tr>
                    <tr>
                        <td><strong>MAE @ Step 72 (12 giờ)</strong></td>
                        <td>{f"{mae_h72:.3f} °C" if mae_h72 is not None else "N/A"}</td>
                        <td>Tham khảo</td>
                        <td><span class="badge badge-na">Tham khảo</span></td>
                    </tr>
                    <tr>
                        <td><strong>MAE @ Step 144 (24 giờ)</strong></td>
                        <td>{f"{mae_h144:.3f} °C" if mae_h144 is not None else "N/A"}</td>
                        <td>≤ 2.20 °C</td>
                        <td>{_get_status_badge(mae_h144, 2.20)}</td>
                    </tr>
                    <tr>
                        <td><strong>Đường cơ sở Persistence (MAE)</strong></td>
                        <td>{f"{pers_mae:.3f} °C" if pers_mae is not None else "N/A"}</td>
                        <td>Seq2Seq phải vượt Baseline (MAE < Persistence)</td>
                        <td>{pers_badge}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- 3. Bảng so sánh 3 mô hình -->
        {f'''
        <div class="card">
            <h2>2. Đối sánh 3 Mô hình (Cùng tầm 144, tập Test 2016)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Mô hình</th>
                        <th>MAE toàn tầm</th>
                        <th>RMSE toàn tầm</th>
                        <th>MAE @ Step 24</th>
                        <th>MAE @ Step 144</th>
                    </tr>
                </thead>
                <tbody>
                    {comp_rows}
                </tbody>
            </table>
        </div>
        ''' if comp_rows else ''}

        <!-- 4. Biểu đồ trực quan -->
        <div class="card">
            <h2>3. Biểu đồ Đánh giá & Huấn luyện</h2>
            <div class="grid-plots">
        """

    for fname, info in encoded_plots.items():
        if info["exists"] and info["b64"]:
            img_tag = f'<img src="{info["b64"]}" alt="{info["title"]}">'
        else:
            img_tag = f'<div class="no-img">Chưa có biểu đồ {fname}</div>'
            
        html_content += f"""
                <div class="plot-box">
                    <h3>{info['title']}</h3>
                    {img_tag}
                </div>
        """

    html_content += f"""
            </div>
        </div>

        <!-- 5. Cấu hình thực thi -->
        {f'''
        <div class="card">
            <h2>4. Cấu hình Lượt chạy (config.yaml)</h2>
            <pre><code>{config_content}</code></pre>
        </div>
        ''' if config_content else ''}
    </div>
</body>
</html>
"""

    out_file = run_path / "report.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return str(out_file)


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "results/runs/demo"
    print(f"[*] Đang tạo báo cáo cho thư mục: {target_dir}")
    try:
        out_path = build_report(target_dir)
        print(f"[OK] Đã xuất báo cáo tại: {out_path}")
    except Exception as err:
        print(f"[LỖI]: {err}")