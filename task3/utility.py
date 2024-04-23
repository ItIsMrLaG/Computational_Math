import argparse
import time
from pathlib import Path
from random import normalvariate

import PIL.Image
import numpy as np
from PIL import Image
from typing import Protocol, Any
import abc
from dataclasses import dataclass

COLOR_NUMBERS = 3
COLOR_SIZE = 1
PIXEL_SIZE = COLOR_NUMBERS * COLOR_SIZE
BMP_HEADER_SIZE = 54
FLOAT_SIZE = 4
COMPRESSED_HEADER = 2 + 4 * 3 + 3


@dataclass
class SVD:
    U: np.matrix
    S: np.ndarray[np.float32]
    VT: np.matrix

    def full_matrix(self) -> np.matrix:
        return self.U @ np.diag(self.S) @ self.VT


class ToSVD(Protocol):
    @abc.abstractmethod
    def get_svd(self, k: int, mt: np.matrix) -> SVD:
        ...


class Limited:
    ms_limit: float | None
    limit_start: float
    working_time: float
    MSG: str = "Time limit in milliseconds too small :("

    def __init__(self, ms_limit=None):
        self.ms_limit = ms_limit
        self.fix_limit_start()

    def fix_limit_start(self) -> None:
        self.limit_start = time.time() * 1000

    @staticmethod
    def get_time_ms() -> float:
        return time.time() * 1000

    def is_time_out(self) -> bool:
        self.working_time = self.get_time_ms()
        if self.ms_limit is not None and (self.working_time - self.limit_start) >= self.ms_limit:
            return True
        return False


class StandardSVD(ToSVD):
    lm: Limited

    def __init__(self, ms_limit=None):
        self.lm = Limited(ms_limit)

    def get_svd(self, k: int, mt: np.matrix) -> SVD:
        self.lm.fix_limit_start()
        u, s, vt = np.linalg.svd(mt, full_matrices=False)

        if self.lm.is_time_out():
            print(self.lm.MSG + "\n" + f"working_time: {self.lm.working_time}")

        u = np.matrix(u.compress([True] * k, axis=1))
        vt = np.matrix(vt.compress([True] * k, axis=0))
        return SVD(u, s[:k], vt)

        return SVD(np.matrix(u), s, np.matrix(vt))


class PowerMethodSVD(ToSVD):
    EPSILON: float

    def __init__(self, ms_limit=None):
        self.EPSILON = 1e-10
        self.lm = Limited(ms_limit)

    @staticmethod
    def random_unit_vector(n: int) -> np.ndarray:
        un_norm = np.array([normalvariate(0, 1) for _ in range(n)])
        norm = np.linalg.norm(un_norm)
        return un_norm / norm

    def _get_singular_vector(self, mt: np.matrix):
        _, m = mt.shape
        curV: np.ndarray = self.random_unit_vector(m)
        AtA: np.matrix = mt.T @ mt
        i: int = 0

        self.lm.fix_limit_start()

        while True:
            if self.lm.is_time_out():
                if i != 0:
                    return curV
            i += 1

            lastV = curV
            curV = np.array(AtA @ lastV)[0]
            curV = curV / np.linalg.norm(curV)

            if abs(curV @ lastV) > 1 - self.EPSILON:
                return curV

    def get_svd(self, k: int, mt: np.matrix) -> SVD:
        _, m = mt.shape
        svd_decomposition: list[tuple[float, np.ndarray, np.ndarray]] = []

        if self.lm.ms_limit is not None:
            self.lm.ms_limit = self.lm.ms_limit / k

        for i in range(k):
            iter_mt = mt.copy().astype(np.float32)

            for s_i, u, v in svd_decomposition[:i]:
                iter_mt -= s_i * np.outer(u, v)

            v = self._get_singular_vector(iter_mt)
            u_un_norm = mt @ v
            sigma_i: float = np.linalg.norm(u_un_norm)
            u = u_un_norm / sigma_i

            svd_decomposition.append((sigma_i, u, v))

        s, u, vt = [np.array(x) for x in zip(*svd_decomposition)]

        return SVD(np.matrix(u.T), s, vt)


class PowerAdvancedMethodSVD(PowerMethodSVD):

    def __init__(self, ms_limit=None):
        super().__init__(ms_limit)
        self.EPSILON = 0.01

    def get_svd(self, k: int, mt: np.matrix) -> SVD:
        n, m = mt.shape

        V0: np.matrix = np.matrix(
            [self.random_unit_vector(k) for _ in range(m)]
        )

        self.lm.fix_limit_start()
        err: float = self.EPSILON + 1
        i: int = 0

        V: np.matrix = V0
        U: np.matrix
        S: np.ndarray

        while err > self.EPSILON:
            if self.lm.is_time_out():
                break

            i += 1
            qr = np.linalg.qr(mt @ V)
            U = np.matrix(qr.Q[:, 0:k])

            qr = np.linalg.qr(mt.T @ U)
            V = np.matrix(qr.Q[:, 0:k])
            S = np.matrix(qr.R[0:k, 0:k])

            err = np.linalg.norm(mt @ V - U @ S)
        if i == 0:
            print(self.lm.MSG)
            exit(1)
        else:
            return SVD(U, np.diagonal(S).astype(np.float32), V.T)


class BMPPixels24:
    R: np.matrix
    G: np.matrix
    B: np.matrix
    RGB: tuple[np.matrix, ...]

    def __getitem__(self, item: tuple[int, int]) -> tuple[int, ...]:
        return tuple([color[item] for color in self.RGB])

    def from_air(self, r: np.matrix, g: np.matrix, b: np.matrix):
        self.RGB = r, g, b
        self.R, self.G, self.B = self.RGB

    def from_img(self, img: PIL.Image.Image):
        n, m = img.size
        self.RGB = tuple([np.matrix(np.zeros((n, m), dtype=int)) for _ in range(COLOR_NUMBERS)])
        self.R, self.G, self.B = self.RGB

        for i in range(n):
            for j in range(m):
                self.R[i, j], self.G[i, j], self.B[i, j] = img.getpixel((i, j))

    def get_img(self) -> PIL.Image.Image:
        sizes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = self.R.shape, self.G.shape, self.B.shape

        if not all(map(lambda el: len(el) == 2, sizes)):
            raise Exception("The image cannot be created because the color matrix dimensions do not match")

        m, n = sizes[0]
        if any(map(lambda el: el != (m, n), sizes)):
            raise Exception("The image cannot be created because the color matrix sizes do not match")

        image: PIL.Image.Image = Image.new("RGB", (m, n))
        for i in range(m):
            for j in range(n):
                r, g, b = self[i, j]
                image.putpixel((i, j), (int(abs(r)), int(abs(g)), int(abs(b))))

        return image


@dataclass
class SVDPixels:
    R: SVD
    G: SVD
    B: SVD

    def __getitem__(self, item: str) -> np.matrix:
        if item == 'R':
            return self.R.full_matrix()
        elif item == 'G':
            return self.G.full_matrix()
        elif item == 'B':
            return self.B.full_matrix()
        raise Exception("Only 'R', 'G', 'B' indexes are available")


@dataclass
class BMP24Compressor:
    to_svd: ToSVD

    def compress_bmp(self, img: PIL.Image.Image, N: float = -1.) -> SVDPixels:
        def k():
            img_size: int = PIXEL_SIZE * m * n + BMP_HEADER_SIZE
            k_val: int = int(img_size // (N * (m + n + 1) * COLOR_NUMBERS * FLOAT_SIZE))
            if k_val <= 0:
                print(f"Can't compress image in {N} times")
                exit(1)
            return k_val

        m, n = img.size
        lines: int

        if N == 1:
            lines_n = 0
        elif N <= 0:
            lines_n = min(n, m)
        else:
            lines_n = min(n, m, k())

        colors: BMPPixels24 = BMPPixels24()
        colors.from_img(img)
        r, g, b = tuple([self.to_svd.get_svd(lines_n, clr) for clr in colors.RGB])
        return SVDPixels(r, g, b)

    @staticmethod
    def decompress_bmp(svd_colors: SVDPixels) -> PIL.Image.Image:
        colors: BMPPixels24 = BMPPixels24()
        colors.from_air(svd_colors['R'], svd_colors['G'], svd_colors['B'])
        return colors.get_img()


class Serializer:
    MAGIC: str = "CP"  # 2 bytes
    M_size: int = 4  # 4 bytes
    N_size: int = 4  # 4 bytes
    K_size: int = 4  # 4 bytes
    ORDER: str = "RGB"  # 3 bytes
    NAME: str = ".BMP.cp"

    @staticmethod
    def _serialize_color(color: SVD) -> bytearray:
        data: bytearray = bytearray()

        data.extend(color.U.astype(dtype=np.float32).tobytes())
        data.extend(color.S.astype(dtype=np.float32).tobytes())
        data.extend(color.VT.astype(dtype=np.float32).tobytes())

        return data

    def serialize(self, svd_colors: SVDPixels, path: Path):
        data: bytearray = bytearray()

        m, _ = svd_colors.R.U.shape
        k: int = svd_colors.R.S.size
        _, n = svd_colors.R.VT.shape

        # Header
        data.extend(self.MAGIC.encode("ASCII"))
        data.extend(np.uint32(m).tobytes())
        data.extend(np.uint32(n).tobytes())
        data.extend(np.uint32(k).tobytes())
        data.extend(self.ORDER.encode("ASCII"))

        for clr in [svd_colors.R, svd_colors.G, svd_colors.B]:
            a = self._serialize_color(clr)
            data.extend(self._serialize_color(clr))

        with open(path, "wb") as f:
            f.write(data)

    def deserialize(self, path: Path) -> SVDPixels:
        def _get_color(local_offset: int) -> tuple[int, SVD]:
            ofs: int = local_offset

            u: np.matrix = np.matrix(
                np.frombuffer(bts, offset=ofs, dtype=np.float32, count=size_u).reshape((m, k))
            )
            ofs += u.size * FLOAT_SIZE

            s = np.frombuffer(bts, offset=ofs, dtype=np.float32, count=k)
            ofs += k * FLOAT_SIZE

            vt: np.matrix = np.matrix(
                np.frombuffer(bts, offset=ofs, dtype=np.float32, count=size_vt).reshape((k, n))
            )
            ofs += vt.size * FLOAT_SIZE

            return ofs, SVD(u, s, vt)

        with open(path, "rb") as f:
            bts: bytes = f.read()

            if bts[0:2].decode("ASCII") != self.MAGIC:
                print("Cannot recognize the file (MAGIC numbers don't match)")
                exit(1)

            m: int = int(np.frombuffer(bts, offset=2, dtype=np.uint32, count=1)[0])
            n: int = int(np.frombuffer(bts, offset=6, dtype=np.uint32, count=1)[0])
            k: int = int(np.frombuffer(bts, offset=10, dtype=np.uint32, count=1)[0])

            order: str = bts[14:17].decode("ASCII")

            if any(map(lambda clr: clr not in ['R', 'G', 'B'], order)):
                print("Cannot recognize the file (incorrect ORDER)")
                exit(1)

            actual_body_size: int = len(bts) - COMPRESSED_HEADER
            body_size: int = (n + m + 1) * k * COLOR_NUMBERS * FLOAT_SIZE

            if actual_body_size != body_size:
                print("Cannot recognize the file (actual_body_size != body_size)")
                exit(1)

            size_u: int = m * k
            size_vt: int = k * n
            offset: int = COMPRESSED_HEADER

            lst = []
            for _ in range(COLOR_NUMBERS):
                ofs, svd = _get_color(offset)
                offset = ofs
                lst.append(svd)

            return SVDPixels(lst[order.find('R')], lst[order.find('G')], lst[order.find('B')])


MODE_COMPRESS: str = "compress"
MODE_DECOMPRESS: str = "decompress"

SVD_ALGO_NUMPY: str = "numpy"
SVD_ALGO_DUMMY: str = "power"
SVD_ALGO_ADVANCED: str = "power-advanced"


def parse_args() -> dict[str, Any]:
    pars = argparse.ArgumentParser()
    pars.add_argument("--in_file", type=str, default="in", help="Path to source file")
    pars.add_argument("--out_file", type=str, default="out", help="Path to result file")
    pars.add_argument("--mode", type=str, default=MODE_COMPRESS,
                      help=f"Choose: <{MODE_COMPRESS}> or <{MODE_DECOMPRESS}>")
    pars.add_argument(
        "--method",
        type=str,
        default=SVD_ALGO_NUMPY,
        help=f"Choose SVD-algorithm: {SVD_ALGO_NUMPY} or {SVD_ALGO_DUMMY} or {SVD_ALGO_ADVANCED}",
    )
    pars.add_argument("-N", type=int, default=2, help="N = original-img-size / compressed-img-size")
    pars.add_argument("-t", type=int, default=1000, help="Time limit in ms")
    return pars.parse_args().__dict__


def check_img(img: PIL.Image.Image):
    if img.format != "BMP":
        print("Only 24bit / per pixel bmp is supported")
        exit(1)
    if len(img.getpixel((0, 0))) != COLOR_NUMBERS:
        print("The version of .bmp is not supported (only 24bit / per pixel)")
        exit(1)
    if img.palette is not None:
        print("Compression for .bmp with palette is not supported")
        exit(1)


if __name__ == "__main__":
    args = parse_args()

    mode = args["mode"]
    infile = args["in_file"]
    outfile = args["out_file"]
    t = args["t"]

    serializer: Serializer = Serializer()

    if mode == MODE_COMPRESS:
        img: PIL.Image.Image = PIL.Image.open(infile)
        check_img(img)

        svder: ToSVD
        algo_type: str = args["method"]

        if algo_type == SVD_ALGO_NUMPY:
            svder = StandardSVD(t)
        elif algo_type == SVD_ALGO_DUMMY:
            svder = PowerMethodSVD(t)
        elif algo_type == SVD_ALGO_ADVANCED:
            svder = PowerAdvancedMethodSVD(t)
        else:
            print(
                f"SVD-algorithm not supported (choose from: {SVD_ALGO_NUMPY} or {SVD_ALGO_DUMMY} or {SVD_ALGO_ADVANCED})")
            exit(1)

        compressor: BMP24Compressor = BMP24Compressor(svder)
        compressed_img: SVDPixels = compressor.compress_bmp(img, args["N"])
        serializer.serialize(compressed_img, Path(outfile + Serializer.NAME))

        print(f"[COMPRESS] {infile} to {outfile + Serializer.NAME}")

    elif mode == MODE_DECOMPRESS:
        img: PIL.Image.Image = BMP24Compressor.decompress_bmp(serializer.deserialize(Path(infile)))
        img.save(outfile, "BMP")

        print(f"[DECOMPRESS] {infile} to {outfile + ".BMP"}")

    else:
        print(f"MODE={mode} is not supported")
        exit(1)
