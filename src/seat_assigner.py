import numpy as np
from scipy.optimize import linear_sum_assignment


class SeatIDAssigner:
    """Gán ID theo chỗ ngồi.

    - Mỗi 'ghế ảo' có một ID nội bộ ổn định (seat_id, dùng làm track_id).
    - Hàng/Cột do NGƯỜI DÙNG xác nhận (set_manual_layout). Nếu chưa có thì ghế chỉ hiển thị
      dạng "#ID". suggest_layout() chỉ là GỢI Ý ban đầu (chia hàng theo toạ độ y), không đáng
      tin với camera đặt chéo góc.
    """

    def __init__(self, match_ratio=0.6, confirm_frames=10, ema=0.03, max_seats=None,
                 row_ratio=0.3, row_from_bottom=True):
        self.match_ratio = match_ratio      # khoảng cách tối đa = match_ratio * bề rộng người
        self.confirm_frames = confirm_frames
        self.ema = ema
        self.max_seats = max_seats          # vd 45; None = không giới hạn
        self.row_ratio = row_ratio          # chỉ dùng cho suggest_layout()
        self.row_from_bottom = row_from_bottom
        self.locked = False                 # True: ngừng tạo ghế mới
        self.seats = {}                     # seat_id -> {"pos", "w", "h"}
        self.candidates = []
        self.next_id = 1
        self.manual = None                  # seat_id -> (hàng, cột) do người dùng xác nhận

    # ---------- chốt / lưu / nạp ghế ----------
    def lock(self):
        self.locked = True
        self.candidates = []

    def export_seats(self):
        return {int(sid): {"pos": [float(v) for v in s["pos"]], "w": float(s["w"]), "h": float(s["h"])}
                for sid, s in self.seats.items()}

    def load_seats(self, seats):
        """Nạp các ghế đã hiệu chỉnh trước đó và chốt luôn (không tạo ghế mới)."""
        self.seats = {int(sid): {"pos": np.array(s["pos"], dtype=float),
                                 "w": float(s["w"]), "h": float(s["h"])}
                      for sid, s in seats.items()}
        self.next_id = max(self.seats, default=0) + 1
        self.lock()

    # ---------- sơ đồ hàng - cột ----------
    @property
    def has_layout(self):
        return bool(self.manual)

    def set_manual_layout(self, layout):
        self.manual = {int(k): (int(v[0]), int(v[1])) for k, v in layout.items()} or None

    def position(self, seat_id):
        """(hàng, cột) do người dùng xác nhận, hoặc None."""
        if not self.manual:
            return None
        return self.manual.get(int(seat_id))

    def label(self, seat_id):
        pos = self.position(seat_id)
        return f"H{pos[0]}-C{pos[1]}" if pos else f"#{int(seat_id)}"

    def long_label(self, seat_id):
        pos = self.position(seat_id)
        return f"Hàng {pos[0]}, Cột {pos[1]}" if pos else f"Ghế #{int(seat_id)}"

    def grid_size(self):
        if not self.manual:
            return 0, 0
        return (max(r for r, _ in self.manual.values()),
                max(c for _, c in self.manual.values()))

    def suggest_layout(self):
        """Gợi ý hàng/cột bằng cách gom theo toạ độ y rồi sắp theo x. Chỉ để điền sẵn bảng."""
        if not self.seats:
            return {}
        items = [(sid, s["pos"][0], s["pos"][1], s["h"]) for sid, s in self.seats.items()]
        items.sort(key=lambda t: t[2], reverse=self.row_from_bottom)
        rows = []
        for it in items:
            if rows:
                mean_y = np.mean([m[2] for m in rows[-1]])
                if abs(it[2] - mean_y) <= self.row_ratio * max(it[3], 1):
                    rows[-1].append(it)
                    continue
            rows.append([it])
        layout = {}
        for r, row in enumerate(rows, start=1):
            row.sort(key=lambda t: t[1])
            for c, it in enumerate(row, start=1):
                layout[it[0]] = (r, c)
        return layout

    # ---------- gán ghế ----------
    @staticmethod
    def _anchor(p):
        x1, y1, x2, y2 = p["box"]
        return np.array([(x1 + x2) / 2, y1], dtype=float), float(x2 - x1), float(y2 - y1)

    def update(self, persons):
        if persons:
            self._match(persons)
        return persons

    def _match(self, persons):
        anchors = [self._anchor(p) for p in persons]
        pts = np.array([a[0] for a in anchors])
        assigned = {}

        ids = list(self.seats.keys())
        if ids:
            S = np.array([self.seats[i]["pos"] for i in ids])
            W = np.array([self.seats[i]["w"] for i in ids])
            cost = np.linalg.norm(pts[:, None] - S[None], axis=2) / W[None, :]
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < self.match_ratio:
                    assigned[r] = ids[c]

        for i, p in enumerate(persons):
            pos, w, h = anchors[i]
            if i in assigned:
                s = self.seats[assigned[i]]
                s["pos"] = (1 - self.ema) * s["pos"] + self.ema * pos
                s["w"] = (1 - self.ema) * s["w"] + self.ema * w
                s["h"] = (1 - self.ema) * s["h"] + self.ema * h
                p["track_id"] = assigned[i]
                continue

            p["track_id"] = None            # chưa xác nhận -> BehaviorEngine bỏ qua
            if self.locked or (self.max_seats and len(self.seats) >= self.max_seats):
                continue
            for cand in self.candidates:
                if np.linalg.norm(cand["pos"] - pos) < 0.5 * max(cand["w"], 1):
                    cand["pos"] = 0.7 * cand["pos"] + 0.3 * pos
                    cand["hits"] += 1
                    if cand["hits"] >= self.confirm_frames:
                        self.seats[self.next_id] = {"pos": cand["pos"], "w": cand["w"], "h": cand["h"]}
                        p["track_id"] = self.next_id
                        self.next_id += 1
                        # So sánh theo identity (is): list.remove() dùng == trên dict chứa mảng numpy sẽ lỗi
                        self.candidates = [c for c in self.candidates if c is not cand]
                    break
            else:
                self.candidates.append({"pos": pos, "w": w, "h": h, "hits": 1})