"""Kiến trúc mô hình cho bài toán Seq2Seq dự báo nhiệt độ Jena.

Cung cấp 3 lớp đúng theo đề bài và `configs/model_config.yaml`:

1. :class:`LSTMS2S` — Encoder–Decoder LSTM cơ sở (mức **Must**).
2. :class:`LSTMS2SAttention` — Encoder–Decoder LSTM + Attention Bahdanau (mức **Should**).
3. :class:`MiniTransformer` — Transformer mini chỉ có encoder (mức **Should**, đối chiếu).

Chữ ký chung::

    forward(x: (B, 144, 15), y_in: (B, 144) | None) -> (B, 144)

* Lúc **huấn luyện**: ``y_in`` là mục tiêu *dịch phải 1 bước* (teacher forcing),
  kèm ``teacher_forcing_ratio`` để trộn dần giữa mục tiêu thật và dự báo của mô hình.
* Lúc **suy luận**: ``y_in = None`` → bộ giải mã tự hồi tiếp đủ 144 bước.

Ba quyết định thiết kế quan trọng (đều ghi trong `docs/progress_status.md`)
-------------------------------------------------------------------------------
**1. Bộ giải mã không biết trước pha giờ.** Cửa sổ huấn luyện dùng
``stride_train = 12`` nên điểm bắt đầu rơi vào bất kỳ pha nào trong ngày.
Bộ giải mã vì thế phải *suy ra* pha từ chính lịch sử 1 ngày quá khứ (đường cong
nhiệt độ 24 giờ vừa vào luôn mang thông tin pha). Trạng thái ẩn cuối của bộ mã
hóa được nối thẳng vào bộ giải mã: làm trạng thái khởi tạo *và* làm một phần
đầu vào của **mọi** bước giải mã.

**2. Cửa sổ vào và cửa sổ ra lệch nhau đúng 144 bước (24 giờ).** Đây là đặc
tính cấu trúc mạnh nhất của bài toán: ``y[t]`` xảy ra **cùng giờ của hôm
qua** với ``x[t]``. Vì vậy:

* *Neo theo cùng giờ* (time-aligned anchor): bước giải mã thứ ``t`` dùng
  ``x[t]`` làm giá trị tham chiếu, và trạng thái ẩn bộ mã hóa **cùng vị trí**
  ``enc_out[:, t]`` được đưa vào bộ giải mã. Bộ giải mã vì thế không phải đoán
  "mình đang ở giờ nào" mà được *đưa cho* đúng ngữ cảnh của giờ đó.
* *Đầu ra dạng phần dư* (residual): ``y_t = x_t + delta_t``. Vì ``y_t - x_t``
  chính là **thay đổi nhiệt độ sau 24 giờ** tại một giờ cố định trong ngày,
  đây là đại lượng "trạng thái tĩnh" hơn nhiều so với ``y_t - x_143`` (lệch pha
  theo giờ). Đường cơ sở seasonal-naive (``y[t] = x[t]``) đạt MAE 2,74 °C
  trên 2011, thấp hơn persistence 3,21 °C — xác nhận neo theo cùng giờ là
  đường cơ sở đúng.

**3. Huấn luyện bằng scheduled sampling về 0.** Tỉ lệ teacher forcing được
giảm **tuyến tính về 0** trong vài epoch đầu, để mô hình học cách "tự chữa"
khi chạy tự do — đúng trạng thái lúc suy luận. Nếu giữ teacher forcing ở mức
0,5 suốt 30 epoch, mô hình học được hàm mất mát huấn luyện rất thấp nhưng
hỏng khi tự hồi tiếp (exposure bias): đo thử trên tập con cho thấy MAE kiểm
định 3,60 °C khi TF còn ở 0,48, và chỉ giảm còn 3,03 °C (vượt persistence) sau
khi TF về 0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Vị trí cột nhiệt độ trong FINAL_FEATURES của src/data/transforms.py
TEMP_FEATURE_INDEX = 1


# --------------------------------------------------------------------------- #
# Thành phần dùng chung
# --------------------------------------------------------------------------- #
class BahdanauAttention(nn.Module):
    """Attention cộng (Bahdanau / additive) trên trạng thái ẩn của bộ mã hóa.

    ``score_t = v^T tanh(W_h * h_dec + W_e * enc_out)``; trọng số được chuẩn hóa
    theo trục thời gian của bộ mã hóa, rồi lấy trọng số trung bình của các
    trạng thái ẩn làm ngữ cảnh.

    **Tối ưu:** ``W_e * enc_out`` không đổi theo bước giải mã (đầu ra bộ mã
    hóa là hằng trong suốt 144 bước), nên được chiếu **một lần** và tái sử
    dụng. Nhờ vậy chi phí attention giảm từ O(T · S · H²) xuống
    O(S · H² + T · S · H) — đây là điểm nghẽn khiến mô hình vượt ngưỡng
    100 ms/mẫu trước đây.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def project_encoder(self, enc_out: torch.Tensor) -> torch.Tensor:
        """Chiếu đầu ra bộ mã hóa **một lần**: (B, S, H) -> (B, S, H)."""
        return self.w_encoder(enc_out)

    def forward(
        self,
        dec_hidden: torch.Tensor,
        enc_out: torch.Tensor,
        enc_proj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dec_hidden: (B, H) trạng thái ẩn bộ giải mã (bước trước).
            enc_out: (B, S, H) đầu ra bộ mã hóa.
            enc_proj: (B, S, H) kết quả ``project_encoder(enc_out)``. Nếu
                ``None`` thì tự tính (đường chạy đơn lẻ, không tối ưu).

        Returns:
            context: (B, H) ngữ cảnh đã gộp.
            weights: (B, S) trọng số attention.
        """
        if enc_proj is None:
            enc_proj = self.project_encoder(enc_out)
        scores = self.v(
            torch.tanh(self.w_hidden(dec_hidden).unsqueeze(1) + enc_proj)
        ).squeeze(-1)  # (B, S)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)  # (B, H)
        return context, weights


class _PositionalEncoding(nn.Module):
    """Mã hoá vị trí kiểu sinusoidal (đăng ký dạng buffer, không học tham số)."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def _init_gate(pred_len: int, lo: float = -2.94, hi: float = 2.94) -> nn.Parameter:
    """Khởi tạo cổng trộn hai neo theo vật lý của bài toán.

    Ở tầm ngắn (vài phút–vài giờ sau thời điểm quan sát cuối), neo
    **persistence** (``x[143]``) chính xác hơn nhiều (MAE@1 của persistence chỉ
    0,135 °C vì ``y[0]`` chỉ cách ``x[143]`` 10 phút). Ở tầm dài (cả ngày), neo
    **cùng giờ hôm qua** (``x[t]``) tốt hơn (MAE@144: 2,74 °C so với 3,21 °C).

    Vì vậy khởi tạo ``sigmoid(gate)`` đi từ ~0,05 ở bước đầu lên ~0,95 ở bước
    cuối, để mô hình bắt đầu gần đúng cả hai vùng và tự điều chỉnh tiếp.
    """
    if pred_len <= 1:
        return nn.Parameter(torch.tensor([lo]))
    ramp = torch.linspace(lo, hi, pred_len)
    return nn.Parameter(ramp)


class _Seq2SBase(nn.Module):
    """Lớp cơ sở chung cho 2 mô hình Encoder–Decoder LSTM.

    Chia sẻ: bộ mã hóa, cơ chế trạng thái ẩn khởi tạo, **neo theo cùng giờ**
    (``x[t]`` cho bước giải mã ``t``) và vòng giải mã tự hồi tiếp có hỗ trợ
    teacher forcing.

    Vì ``input_len == pred_len == 144`` và hai cửa sổ lệch nhau đúng 24 giờ,
    ``y[t]`` xảy ra cùng giờ của hôm qua với ``x[t]``. Vì vậy:
      * giá trị neo của bước ``t`` là ``x[t]`` (cùng giờ), không phải ``x[143]``;
      * đầu ra là phần dư: ``y_t = x[t] + delta_t``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pred_len: int,
        temp_index: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.temp_index = temp_index

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    # -- tiện ích ----------------------------------------------------------
    def _anchor_series(self, x: torch.Tensor) -> torch.Tensor:
        """Chuỗi neo theo cùng giờ: ``x[:, :pred_len, temp_index]`` (B, T).

        Nếu cửa sổ vào ngắn hơn cửa sổ ra, phần thiếu được lấp bằng giá trị
        cuối cùng có sẵn (hiếm gặp với cấu hình 144→144).
        """
        temp = x[:, :, self.temp_index]
        if temp.size(1) >= self.pred_len:
            return temp[:, : self.pred_len]
        pad = temp[:, -1:].expand(-1, self.pred_len - temp.size(1))
        return torch.cat([temp, pad], dim=1)

    def _blend_base(
        self, anchor_t: torch.Tensor, persist: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """Trộn hai neo theo cổng học được, rồi cộng phần dư.

        ``base_t = w_t * x[t] + (1 - w_t) * x[143]`` với ``w_t = sigmoid(gate)``.

        * ``x[t]`` là neo **theo cùng giờ** (cùng đồng hồ hôm qua) — hợp với
          tầm dài (24 giờ).
        * ``x[143]`` là neo **persistence** (giá trị quan sát cuối cùng) — hợp với
          tầm ngắn, vì ``y[0]`` chỉ cách ``x[143]`` 10 phút.

        Ở tầm 1 bước, mô hình học ``w ≈ 0`` (dùng persistence, MAE@1 ≈ 0,14 °C);
        ở tầm 144 bước, ``w ≈ 1`` (dùng neo cùng giờ). Phần dư ``delta_t`` là
        phần tự do để mô hình học thay đổi thực tế từ 15 đặc trưng.
        """
        w = torch.sigmoid(gate)
        return w * anchor_t + (1.0 - w) * persist

    def _start_token(self, x: torch.Tensor) -> torch.Tensor:
        """Ký hiệu bắt đầu cho bộ giải mã = nhiệt độ cuối ngày hôm qua."""
        return x[:, -1, self.temp_index].unsqueeze(-1)  # (B, 1)

    def _init_decoder_state(
        self, enc_h: torch.Tensor, enc_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trạng thái ẩn khởi tạo cho bộ giải mã lấy từ bộ mã hóa."""
        h = enc_h[-1].contiguous().unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = enc_c[-1].contiguous().unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h, c

    def _teacher_input(self, y_in: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """Chuỗi mục tiêu *dịch phải 1 bước*, dài ``T + 1``.

        ``teacher_input[:, 0] = start`` (nhiệt độ cuối ngày hôm qua),
        ``teacher_input[:, k] = y[:, k-1]`` với ``k = 1..T``.

        Vì chuỗi dài ``T + 1``, bước giải mã ``t`` (0 ≤ t < T) luôn tìm được
        mục tiêu thật cho **bước kế tiếp** tại ``teacher_input[:, t + 1] =
        y[:, t]``. Trước đây chuỗi chỉ dài ``T`` và được tra đúng chỗ ``t``,
        khiến bộ giải mã ăn nhầm giá trị lệch 1 bước (off-by-one) — lỗi này
        làm hiệu quả teacher forcing sai lệch so với lúc suy luận.
        """
        return torch.cat([start, y_in], dim=1)  # (B, T+1)

    def _select_next_input(
        self,
        teacher_input: torch.Tensor,
        index: int,
        y_t: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        """Chọn đầu vào bước tiếp theo: mục tiêu thật hay dự báo của mô hình.

        Args:
            teacher_input: chuỗi mục tiêu dịch phải 1 bước, dài ``T + 1``.
            index: chỉ số bước **kế tiếp** trong ``teacher_input`` (tức ``t + 1``).
            y_t: dự báo của bước hiện tại, dùng khi không dùng teacher forcing.
            teacher_forcing_ratio: xác suất dùng mục tiêu thật.
        """
        rand = torch.rand(y_t.size(0), device=y_t.device)
        chosen = torch.where(
            rand < teacher_forcing_ratio, teacher_input[:, index], y_t
        )
        return chosen.unsqueeze(-1)  # (B, 1)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"x phải có shape (B, S, F), nhận được {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"x phải có {self.input_dim} đặc trưng ở chiều cuối "
                f"(F = 15 sau khi tách wd), nhận được {x.size(-1)}"
            )

    def _check_target(self, x: torch.Tensor, y_in: torch.Tensor | None) -> None:
        """Kiểm tra shape của mục tiêu dịch phải 1 bước."""
        if y_in is None:
            return
        if y_in.dim() != 2:
            raise ValueError(
                f"y_in phải có shape (B, T), nhận được {tuple(y_in.shape)}"
            )
        if y_in.size(0) != x.size(0):
            raise ValueError(
                f"y_in có batch {y_in.size(0)} khác batch của x {x.size(0)}"
            )
        if y_in.size(1) != self.pred_len:
            raise ValueError(
                f"y_in phải có {self.pred_len} bước (pred_len), "
                f"nhận được {y_in.size(1)}"
            )


# --------------------------------------------------------------------------- #
# 1. Encoder–Decoder LSTM cơ sở
# --------------------------------------------------------------------------- #
class LSTMS2S(_Seq2SBase):
    """Encoder–Decoder LSTM cơ sở — mô hình mức **Must**, đúng tên trong đề.

    * Bộ mã hóa: LSTM 2 lớp, ``hidden_dim = 128``, nhận (B, 144, 15).
    * Bộ giải mã: LSTM 2 lớp, nhận ``[T_bước_trước, T_neo, trạng thái ẩn encoder]``
      cho mỗi bước, sinh 144 giá trị nhiệt độ (đã chuẩn hóa).
    * Đầu ra residual: ``y_t = T_neo + delta_t``.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        pred_len: int = 144,
        temp_index: int = TEMP_FEATURE_INDEX,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, dropout, pred_len, temp_index)

        # đầu vào bộ giải mã: [T bước trước (1), T neo (1), ngữ cảnh encoder (H)]
        self.decoder = nn.LSTM(
            input_size=2 + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, 1)
        # cổng trộn hai neo (cùng giờ vs persistence), khởi tạo ưu tiên neo cùng giờ
        self.gate = _init_gate(pred_len)

    def forward(
        self,
        x: torch.Tensor,
        y_in: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, F) chuỗi quá khứ đã chuẩn hóa.
            y_in: (B, T) mục tiêu *dịch phải 1 bước*; ``None`` = tự hồi tiếp.
            teacher_forcing_ratio: xác suất dùng mục tiêu thật ở mỗi bước.

        Returns:
            (B, T) chuỗi nhiệt độ dự báo (đơn vị chuẩn hóa).
        """
        self._check_input(x)
        self._check_target(x, y_in)
        batch = x.size(0)
        steps = self.pred_len if y_in is None else y_in.size(1)

        enc_out, (enc_h, enc_c) = self.encoder(x)
        enc_last_h = enc_h[-1]  # (B, H)

        dec_h, dec_c = self._init_decoder_state(enc_h, enc_c)

        anchor = self._anchor_series(x)  # (B, T) — cùng giờ với từng bước đích
        persist = x[:, -1, self.temp_index].unsqueeze(1).expand(-1, self.pred_len)
        start = self._start_token(x)  # (B, 1)
        teacher_input = self._teacher_input(y_in, start) if y_in is not None else None

        prev = start
        use_tf = y_in is not None and teacher_forcing_ratio > 0.0 and self.training
        outputs: list[torch.Tensor] = []

        for t in range(steps):
            # ngữ cảnh của đúng giờ t: trạng thái ẩn bộ mã hóa cùng vị trí
            ctx = enc_out[:, t] if t < enc_out.size(1) else enc_last_h
            dec_in = torch.cat(
                [prev, anchor[:, t:t + 1], ctx], dim=-1
            ).unsqueeze(1)  # (B, 1, 2+H)
            dec_out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))
            delta = self.output_proj(dec_out.squeeze(1)).squeeze(-1)  # (B,)
            base = self._blend_base(anchor[:, t], persist[:, t], self.gate[t])
            y_t = base + delta  # đầu ra = trộn hai neo + phần dư học được
            outputs.append(y_t)

            if use_tf:
                # mục tiêu thật cho bước kế tiếp nằm ở teacher_input[:, t + 1]
                prev = self._select_next_input(
                    teacher_input, t + 1, y_t, teacher_forcing_ratio
                )
            else:
                prev = y_t.unsqueeze(-1)

        return torch.stack(outputs, dim=1)


# --------------------------------------------------------------------------- #
# 2. Encoder–Decoder LSTM + Attention Bahdanau
# --------------------------------------------------------------------------- #
class LSTMS2SAttention(_Seq2SBase):
    """Encoder–Decoder LSTM kèm Attention Bahdanau — mức **Should**.

    Khác :class:`LSTMS2S` ở chỗ mỗi bước giải mã, bộ giải mã "nhìn lại" toàn bộ
    144 trạng thái ẩn của bộ mã hóa qua cơ chế attention cộng, nhờ vậy giữ được
    thông tin về các đợt thay đổi đột ngột (ví dụ trước bão) ở nửa sau ngày.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        pred_len: int = 144,
        temp_index: int = TEMP_FEATURE_INDEX,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_layers, dropout, pred_len, temp_index)

        self.attention = BahdanauAttention(hidden_dim)
        # đầu vào bộ giải mã: [T bước trước (1), T neo (1), ngữ cảnh attention (H)]
        self.decoder = nn.LSTM(
            input_size=2 + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(2 * hidden_dim, 1)
        # cổng trộn hai neo (cùng giờ vs persistence)
        self.gate = _init_gate(pred_len)

    def forward(
        self,
        x: torch.Tensor,
        y_in: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        self._check_input(x)
        self._check_target(x, y_in)
        batch = x.size(0)
        steps = self.pred_len if y_in is None else y_in.size(1)

        enc_out, (enc_h, enc_c) = self.encoder(x)  # (B, S, H)
        dec_h, dec_c = self._init_decoder_state(enc_h, enc_c)

        anchor = self._anchor_series(x)  # (B, T)
        persist = x[:, -1, self.temp_index].unsqueeze(1).expand(-1, self.pred_len)
        start = self._start_token(x)
        teacher_input = self._teacher_input(y_in, start) if y_in is not None else None

        prev = start
        use_tf = y_in is not None and teacher_forcing_ratio > 0.0 and self.training
        # chiếu đầu ra bộ mã hóa MỘT LẦN (hằng trong suốt 144 bước giải mã)
        enc_proj = self.attention.project_encoder(enc_out)
        # ngữ cảnh attention của bước trước, khởi tạo từ trạng thái ẩn encoder
        context, _ = self.attention(enc_h[-1], enc_out, enc_proj)
        outputs: list[torch.Tensor] = []

        for t in range(steps):
            dec_in = torch.cat(
                [prev, anchor[:, t:t + 1], context], dim=-1
            ).unsqueeze(1)  # (B, 1, 2+H)
            dec_out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))
            h_last = dec_h[-1]  # (B, H)
            # attention của bước hiện tại dùng trạng thái ẩn vừa cập nhật
            context, _ = self.attention(h_last, enc_out, enc_proj)
            delta = self.output_proj(torch.cat([h_last, context], dim=-1)).squeeze(-1)
            base = self._blend_base(anchor[:, t], persist[:, t], self.gate[t])
            y_t = base + delta
            outputs.append(y_t)

            if use_tf:
                # mục tiêu thật cho bước kế tiếp nằm ở teacher_input[:, t + 1]
                prev = self._select_next_input(
                    teacher_input, t + 1, y_t, teacher_forcing_ratio
                )
            else:
                prev = y_t.unsqueeze(-1)

        return torch.stack(outputs, dim=1)


# --------------------------------------------------------------------------- #
# 3. Transformer mini (chỉ encoder)
# --------------------------------------------------------------------------- #
class MiniTransformer(nn.Module):
    """Transformer mini chỉ có encoder — mức **Should**, dùng để đối chiếu.

    Kiến trúc: chiếu đặc trưng ``15 → d_model``, cộng mã hoá vị trí, 2 lớp
    Transformer encoder (``nhead = 4``), rồi đầu hồi quy ``d_model → 1`` cho
    từng bước trong 144 bước. Mô hình này **không** tự hồi tiếp: một lượt
    forward cho ra đủ 144 bước, nên không có hiện tượng exposure bias.

    Để đối chiếu **cùng cấu trúc đầu ra** với hai mô hình LSTM (yêu cầu
    "compare against Transformer" trong đề), Transformer cũng dùng chung cách
    viết đầu ra **neo theo cùng giờ + phần dư**:

        ``y_t = w_t * x[t] + (1 - w_t) * x[143] + delta_t``

    với ``w_t = sigmoid(gate_t)``. Nhờ vậy cả ba mô hình học cùng một bài toán
    (phần dư nhiệt độ so với neo) nên bảng so sánh MAE tại 1/24/72/144 là
    công bằng, và Transformer không phải "học từ đầu" cả đường cong nhiệt độ
    chỉ để bị so với LSTM đã có sẵn ưu thế cấu trúc.
    """

    def __init__(
        self,
        input_dim: int = 15,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pred_len: int = 144,
        dim_feedforward: int | None = None,
        temp_index: int = TEMP_FEATURE_INDEX,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.temp_index = temp_index
        ff_dim = dim_feedforward if dim_feedforward is not None else 4 * d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # cổng trộn hai neo (cùng giờ vs persistence) — giống hai mô hình LSTM
        self.gate = _init_gate(pred_len)

    def _anchor_series(self, x: torch.Tensor) -> torch.Tensor:
        """Chuỗi neo theo cùng giờ: ``x[:, :pred_len, temp_index]`` (B, T)."""
        temp = x[:, :, self.temp_index]
        if temp.size(1) >= self.pred_len:
            return temp[:, : self.pred_len]
        pad = temp[:, -1:].expand(-1, self.pred_len - temp.size(1))
        return torch.cat([temp, pad], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        y_in: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """``y_in`` được chấp nhận để thống nhất chữ ký với 2 mô hình Seq2Seq
        (mô hình này không dùng teacher forcing)."""
        if x.dim() != 3:
            raise ValueError(f"x phải có shape (B, S, F), nhận được {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"x phải có {self.input_dim} đặc trưng ở chiều cuối "
                f"(F = 15 sau khi tách wd), nhận được {x.size(-1)}"
            )
        h = self.input_proj(x)  # (B, S, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h = self.norm(h)
        delta = self.head(h).squeeze(-1)  # (B, S) — phần dư
        if delta.size(1) != self.pred_len:
            if delta.size(1) >= self.pred_len:
                delta = delta[:, : self.pred_len]
            else:
                pad = delta[:, -1:].expand(-1, self.pred_len - delta.size(1))
                delta = torch.cat([delta, pad], dim=1)

        # đầu ra dạng neo + phần dư (giống LSTMS2S / LSTMS2SAttention)
        anchor = self._anchor_series(x)  # (B, T)
        persist = x[:, -1, self.temp_index].unsqueeze(1).expand(-1, self.pred_len)
        w = torch.sigmoid(self.gate).unsqueeze(0)  # (1, T)
        base = w * anchor + (1.0 - w) * persist
        return base + delta


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
ARCH_REGISTRY = {
    "lstm_s2s": LSTMS2S,
    "lstm_attention": LSTMS2SAttention,
    "bahdanau": LSTMS2SAttention,
    "mini_transformer": MiniTransformer,
    "transformer": MiniTransformer,
}


# Tham số hợp lệ cho từng kiến trúc (dùng để lọc kwargs trước khi dựng mô hình)
ARCH_PARAMS: dict[str, set[str]] = {
    "lstm_s2s": {
        "input_dim", "hidden_dim", "num_layers", "dropout", "pred_len", "temp_index",
    },
    "lstm_attention": {
        "input_dim", "hidden_dim", "num_layers", "dropout", "pred_len", "temp_index",
    },
    "bahdanau": {
        "input_dim", "hidden_dim", "num_layers", "dropout", "pred_len", "temp_index",
    },
    "mini_transformer": {
        "input_dim", "d_model", "nhead", "num_layers", "dropout", "pred_len",
        "dim_feedforward", "temp_index",
    },
    "transformer": {
        "input_dim", "d_model", "nhead", "num_layers", "dropout", "pred_len",
        "dim_feedforward", "temp_index",
    },
}


def build_model(arch: str, **kwargs) -> nn.Module:
    """Dựng mô hình theo tên kiến trúc trong ``configs/model_config.yaml``.

    Tham số không thuộc kiến trúc đó bị **bỏ qua có chủ đích** (ví dụ
    ``hidden_dim`` không có ý nghĩa với Transformer). Nhờ vậy cùng một bộ
    cấu hình chung vẫn dựng được cả ba kiến trúc mà không cần rà nhánh.
    """
    key = arch.lower().replace("-", "_")
    if key not in ARCH_REGISTRY:
        raise ValueError(
            f"Kiến trúc '{arch}' không hợp lệ. Chọn một trong: {sorted(ARCH_REGISTRY)}"
        )
    cls = ARCH_REGISTRY[key]
    valid = ARCH_PARAMS[key]
    params = {k: v for k, v in kwargs.items() if k in valid}
    return cls(**params)


def count_parameters(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện (không tính buffer)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimated_checkpoint_mb(model: nn.Module) -> float:
    """Ước lượng kích thước checkpoint (float32) theo MB."""
    total = sum(p.numel() for p in model.parameters())
    total += sum(b.numel() for b in model.buffers())
    return total * 4 / (1024 ** 2)
