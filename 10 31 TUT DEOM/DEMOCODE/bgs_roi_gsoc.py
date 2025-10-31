# bgs_roi_gsoc.py —— 背景减除 ROI 模块（同步 bgs_best_roi_out1 流程）
# 依赖：pip install opencv-contrib-python
from pathlib import Path
import cv2, numpy as np

__all__ = ["BGSROI", "BGSParams"]

class BGSParams:
    """
    可调参数（默认与 bgs_best_roi_out1.py 的“新版本”一致）
    """
    def __init__(
        self,
        # 预热 & BGS
        warmup=30,
        use_auto_thresh=True,     # 使用 Otsu 自适应阈值
        bin_thresh=140,           # 备用：固定阈值（当 use_auto_thresh=False 时使用）
        thr_min=80, thr_max=230,  # 自适应阈值安全夹

        # 帧差门控
        use_three_frame=True,
        diff_tau=24,
        diff_blur_ksize=5,
        motion_dilate_k=7,

        # 形态学重建
        recon_kernel=5,
        recon_max_iter=50,

        # 最终形态学清理
        final_open_iters=1,
        final_close_iters=3,

        # 衰减记忆（去拖影）：默认关闭（=0）
        decay=0.0,
        mem_thresh=0.5,

        # ROI 选择与平滑
        min_area=800,
        max_area_ratio=0.5,
        pad_ratio=0.30,
        ema_alpha=0.6,

        # ROI 组合策略
        roi_topk=3,               # 取前 K 个连通域并集
        use_convex_hull=False,    # True=用 K 个连通域的凸包外接矩形
        debug_draw_hull=False,    # 仅可视化：在 vis 上画出凸包
    ):
        self.warmup = int(warmup)

        self.use_auto_thresh = bool(use_auto_thresh)
        self.bin_thresh = int(bin_thresh)
        self.thr_min = int(thr_min)
        self.thr_max = int(thr_max)

        self.use_three_frame = bool(use_three_frame)
        self.diff_tau = int(diff_tau)
        self.diff_blur_ksize = int(diff_blur_ksize)
        self.motion_dilate_k = int(motion_dilate_k)

        self.recon_kernel = int(recon_kernel)
        self.recon_max_iter = int(recon_max_iter)

        self.final_open_iters = int(final_open_iters)
        self.final_close_iters = int(final_close_iters)

        self.decay = float(decay)
        self.mem_thresh = float(mem_thresh)

        self.min_area = int(min_area)
        self.max_area_ratio = float(max_area_ratio)
        self.pad_ratio = float(pad_ratio)
        self.ema_alpha = float(ema_alpha)

        self.roi_topk = int(roi_topk)
        self.use_convex_hull = bool(use_convex_hull)
        self.debug_draw_hull = bool(debug_draw_hull)

class BGSROI:
    """
    用法：
        from bgs_roi_gsoc import BGSROI, BGSParams
        bgs = BGSROI(BGSParams(...))
        roi, vis, mask = bgs.apply(frame)    # roi=(x1,y1,x2,y2) 或 None
    """
    def __init__(self, params: BGSParams = None):
        self.p = params or BGSParams()
        # GSOC 背景减除器（需 opencv-contrib-python）
        self.bgs = cv2.bgsegm.createBackgroundSubtractorGSOC()

        # 状态
        self.frames_seen = 0
        self.prev_gray = None
        self.prev2_gray = None
        self.mem_state = None             # 衰减记忆状态（float32 0..1）
        self.prev_box = None              # EMA (cx,cy,w,h)

        # 结构元素
        self.k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mk = max(1, self.p.motion_dilate_k)
        self.k_motion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))

    # ------------ 小工具 ------------
    def reset(self):
        self.__init__(self.p)

    @staticmethod
    def _clamp(x1,y1,x2,y2,W,H):
        return max(0,x1), max(0,y1), min(W,x2), min(H,y2)

    def _ema_box(self, prev, cur):
        a = self.p.ema_alpha
        if prev is None or a <= 0: return cur
        return tuple(int(a*pv + (1-a)*cv) for pv,cv in zip(prev,cur))

    def _auto_threshold(self, raw):
        # Otsu 自适应阈值 + 安全夹
        thr, _ = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(thr, self.p.thr_min, self.p.thr_max))
        _, out = cv2.threshold(raw, thr, 255, cv2.THRESH_BINARY)
        return out

    def _motion_gate(self, gray_t, gray_t1, gray_t2=None):
        if gray_t1 is None:
            # 首帧：全部通过
            return np.full_like(gray_t, 255, dtype=np.uint8)

        g0, g1 = gray_t, gray_t1
        k = self.p.diff_blur_ksize
        if k and k > 1:
            g0 = cv2.GaussianBlur(g0, (k,k), 0)
            g1 = cv2.GaussianBlur(g1, (k,k), 0)

        diff = cv2.absdiff(g0, g1)
        if self.p.use_three_frame and gray_t2 is not None:
            g2 = gray_t2
            if k and k > 1:
                g2 = cv2.GaussianBlur(g2, (k,k), 0)
            diff = cv2.max(diff, cv2.absdiff(g0, g2))

        _, mot = cv2.threshold(diff, self.p.diff_tau, 255, cv2.THRESH_BINARY)
        if self.p.motion_dilate_k > 1:
            mot = cv2.dilate(mot, self.k_motion, 1)
        return mot

    def _morph_reconstruct(self, seed, support):
        ksize = max(1, int(self.p.recon_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        prev = seed.copy()
        for _ in range(self.p.recon_max_iter):
            dil = cv2.dilate(prev, kernel, 1)
            rec = cv2.bitwise_and(dil, support)
            if np.array_equal(rec, prev): break
            prev = rec
        return prev

    def _decay_memory(self, cur_bin):
        # decay<=0 表示关闭记忆：直接返回当前帧
        if self.p.decay <= 0.0:
            return cur_bin
        cur_f = (cur_bin.astype(np.float32) / 255.0)
        if self.mem_state is None:
            self.mem_state = cur_f
        else:
            self.mem_state = np.maximum(cur_f, self.mem_state * self.p.decay)
        out = (self.mem_state >= self.p.mem_thresh).astype(np.uint8) * 255
        return out

    # —— ROI：前 K 连通域并集 —— #
    def _roi_from_top_components(self, mask, W, H):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: 
            return None, None
        valid = []
        frame_max = self.p.max_area_ratio * W * H
        for c in cnts:
            a = cv2.contourArea(c)
            if a >= self.p.min_area and a <= frame_max:
                valid.append((a, c))
        if not valid:
            return None, None
        valid.sort(key=lambda x: x[0], reverse=True)
        valid = [c for _,c in valid[:max(1,self.p.roi_topk)]]

        # 并集外接矩形
        x1=y1=10**9; x2=y2=-10**9
        for c in valid:
            x,y,w,h = cv2.boundingRect(c)
            x1 = min(x1, x);   y1 = min(y1, y)
            x2 = max(x2, x+w); y2 = max(y2, y+h)
        rect = (x1,y1,x2,y2)
        return rect, valid

    # —— ROI：凸包外接矩形（可选） —— #
    def _roi_from_convex_hull(self, mask, W, H):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None
        frame_max = self.p.max_area_ratio * W * H
        valid = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a >= self.p.min_area and a <= frame_max:
                valid.append((a, c))
        if not valid:
            return None, None
        valid.sort(key=lambda x: x[0], reverse=True)
        valid = [c for _,c in valid[:max(1,self.p.roi_topk)]]
        pts = np.vstack([c.reshape(-1,1,2) for c in valid])  # (N,1,2)
        hull = cv2.convexHull(pts)
        x,y,w,h = cv2.boundingRect(hull)
        rect = (x, y, x+w, y+h)
        return rect, hull

    # ------------ 主入口 ------------
    def apply(self, frame):
        """
        返回：
          roi_rect : (x1,y1,x2,y2) 或 None
          vis      : 在原图上画出 ROI 的可视化
          mask     : 黑白前景掩膜（二值，0/255）
        """
        self.frames_seen += 1
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 背景响应
        raw = self.bgs.apply(frame)

        # 预热阶段：不输出 ROI（与脚本版一致）
        if self.frames_seen <= self.p.warmup:
            vis = frame.copy()
            # 如需提示可在此写文字；此处保持安静输出
            self.prev2_gray = self.prev_gray
            self.prev_gray = gray
            return None, vis, raw

        # 1) BGS 阈值
        if self.p.use_auto_thresh:
            bgs_bin = self._auto_threshold(raw)
        else:
            _, bgs_bin = cv2.threshold(raw, self.p.bin_thresh, 255, cv2.THRESH_BINARY)

        # 2) 帧差门控
        mot = self._motion_gate(gray, self.prev_gray, self.prev2_gray)
        mot_wide = cv2.dilate(mot, self.k_motion, 1) if self.p.motion_dilate_k>1 else mot

        # 3) 形态学重建
        seed   = cv2.bitwise_and(bgs_bin, mot_wide)
        filled = self._morph_reconstruct(seed, bgs_bin)

        # 4) 最终开/闭
        if self.p.final_close_iters:
            filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, self.k3, iterations=self.p.final_close_iters)
        if self.p.final_open_iters:
            filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN,  self.k3, iterations=self.p.final_open_iters)

        # 5) 记忆（默认关闭=当前帧直出）
        mask_bin = self._decay_memory(filled)

        # 6) ROI：并集 or 凸包
        vis = frame.copy()
        rect = None
        hull = None
        if self.p.use_convex_hull:
            rect, hull = self._roi_from_convex_hull(mask_bin, W, H)
            if rect is None:
                rect, _ = self._roi_from_top_components(mask_bin, W, H)  # 回退
        else:
            rect, _ = self._roi_from_top_components(mask_bin, W, H)

        roi_rect = None
        if rect is not None:
            x1, y1, x2, y2 = rect
            # 外扩
            w, h = x2 - x1, y2 - y1
            padw, padh = int(w * self.p.pad_ratio), int(h * self.p.pad_ratio)
            x1, y1, x2, y2 = self._clamp(x1 - padw, y1 - padh, x2 + padw, y2 + padh, W, H)

            # EMA 平滑
            cx, cy, ww, hh = ((x1+x2)//2, (y1+y2)//2, x2-x1, y2-y1)
            cx, cy, ww, hh = self._ema_box(self.prev_box, (cx,cy,ww,hh))
            self.prev_box = (cx,cy,ww,hh)
            x1, y1, x2, y2 = self._clamp(cx - ww//2, cy - hh//2, cx + ww//2, cy + hh//2, W, H)

            roi_rect = (x1,y1,x2,y2)
            # 画 ROI
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, "ROI", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            # 可选：画凸包
            if self.p.debug_draw_hull and hull is not None:
                cv2.polylines(vis, [hull], True, (0,255,0), 1)

        # 更新灰度历史
        self.prev2_gray = self.prev_gray
        self.prev_gray  = gray

        return roi_rect, vis, mask_bin
