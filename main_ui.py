"""Mini NPU Simulator - MAC 연산 시뮬레이터 (모드 1 사용자 입력 / 모드 2 data.json 분석)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time
from typing import Callable, Iterable

DEFAULT_EPSILON = 1e-9
BENCHMARK_ITERATIONS = 10

LABEL_MAP: dict[str, str] = {
    "+": "Cross",
    "cross": "Cross",
    "x": "X",
}

FILTER_SIZE_KEY_RE = re.compile(r"^size_(\d+)$")
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_.+$")


class Matrix:
    """n x n 크기의 2차원 숫자 데이터를 저장하고 위치별 접근을 제공하는 자료구조."""

    def __init__(self, size: int, fill: float = 0.0) -> None:
        if size <= 0:
            raise ValueError(f"행렬 크기는 1 이상이어야 합니다: {size}")
        self.size = size
        val = float(fill)
        self._data: list[list[float]] = [[val for _ in range(size)] for _ in range(size)]

    @classmethod
    def from_rows(cls, rows: Iterable[Iterable[int | float]]) -> Matrix:
        """2차원 시퀀스로부터 정사각 Matrix 인스턴스를 생성한다."""
        parsed_rows = [[float(cell) for cell in row] for row in rows]
        n = len(parsed_rows)
        if n == 0:
            raise ValueError("빈 행렬은 생성할 수 없습니다.")
        for idx, row in enumerate(parsed_rows):
            if len(row) != n:
                raise ValueError(
                    f"{idx}번 행의 열 개수({len(row)})가 행 개수({n})와 일치하지 않습니다."
                )
        matrix = cls(n)
        matrix._data = parsed_rows
        return matrix

    def _validate_index(self, row: int, col: int) -> None:
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError(
                f"인덱스 범위 초과: ({row}, {col}), 허용 범위 0..{self.size - 1}"
            )

    def get(self, row: int, col: int) -> float:
        """지정된 (row, col) 위치의 값을 반환한다."""
        self._validate_index(row, col)
        return self._data[row][col]

    def set(self, row: int, col: int, value: int | float) -> None:
        """지정된 (row, col) 위치에 값을 저장한다."""
        self._validate_index(row, col)
        self._data[row][col] = float(value)

    @property
    def rows(self) -> list[list[float]]:
        """내부 2차원 리스트 복사본을 반환한다."""
        return [row[:] for row in self._data]


def normalize_label(raw_label: str) -> str:
    """다양한 필터/기대값 라벨을 표준 라벨('Cross' 또는 'X')로 정규화한다."""
    if not isinstance(raw_label, str):
        raise ValueError(f"라벨은 문자열이어야 합니다: {raw_label!r}")
    normalized = LABEL_MAP.get(raw_label.strip().lower())
    if normalized is None:
        raise ValueError(f"지원하지 않는 라벨입니다: {raw_label!r}")
    return normalized


def compute_mac(pattern: Matrix, filter_matrix: Matrix) -> float:
    """외부 라이브러리 없이 이중 반복문으로 패턴과 필터의 MAC 점수를 계산한다."""
    if pattern.size != filter_matrix.size:
        raise ValueError(
            f"크기 불일치: 패턴({pattern.size}x{pattern.size}) != "
            f"필터({filter_matrix.size}x{filter_matrix.size})"
        )
    total = 0.0
    n = pattern.size
    for r in range(n):
        for c in range(n):
            total += pattern.get(r, c) * filter_matrix.get(r, c)
    return float(total)


def compare_scores(
    score_a: float,
    score_b: float,
    label_a: str = "A",
    label_b: str = "B",
    tie_label: str = "UNDECIDED",
    epsilon: float = DEFAULT_EPSILON,
) -> str:
    """허용오차(epsilon) 기준으로 두 점수를 비교해 승자 라벨 또는 동점 라벨을 반환한다."""
    if abs(score_a - score_b) < epsilon:
        return tie_label
    return label_a if score_a > score_b else label_b


# ---------------------------------------------------------------------------
# 3단계: 모드 1 콘솔 입력 검증 및 모드 2 data.json 스키마/크기 검증
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedCase:
    """JSON 패턴 케이스의 검증 결과와 정규화된 데이터를 담는다."""

    case_id: str
    size: int | None = None
    pattern: Matrix | None = None
    cross_filter: Matrix | None = None
    x_filter: Matrix | None = None
    expected: str | None = None
    error: str | None = None


def parse_matrix_row(line: str, expected_cols: int = 3) -> list[float]:
    """공백으로 구분된 한 줄 문자열을 검증하고 float 리스트로 변환한다."""
    tokens = line.strip().split()
    format_msg = f"입력 형식 오류: 각 줄에 {expected_cols}개의 숫자를 공백으로 구분해 입력하세요."
    if len(tokens) != expected_cols:
        raise ValueError(format_msg)
    values: list[float] = []
    for token in tokens:
        try:
            num = float(token)
        except ValueError as exc:
            raise ValueError(format_msg) from exc
        if not math.isfinite(num):
            raise ValueError(format_msg)
        values.append(num)
    return values


def read_matrix_from_console(
    title: str,
    size: int = 3,
    input_fn: Callable[[str], str] = input,
) -> Matrix:
    """콘솔에서 size x size 행렬을 한 줄씩 입력받으며 오류 시 재입력을 유도한다."""
    print(f"  > {title} | {size}행 x {size}열 (숫자는 공백으로 구분)")
    rows: list[list[float]] = []
    row_idx = 0
    while row_idx < size:
        raw_line = input_fn(f"    {row_idx + 1:02d}행 > ")
        try:
            parsed_row = parse_matrix_row(raw_line, expected_cols=size)
        except ValueError as exc:
            print(f"{exc} (현재 {row_idx + 1}번째 줄 재입력)")
            continue
        rows.append(parsed_row)
        row_idx += 1
    return Matrix.from_rows(rows)


def _is_valid_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def parse_square_matrix(
    raw_matrix: object,
    expected_size: int | None = None,
    name: str = "행렬",
) -> Matrix:
    """JSON 등에서 읽은 2차원 배열을 검증해 Matrix로 변환한다."""
    if not isinstance(raw_matrix, list) or len(raw_matrix) == 0:
        raise ValueError(f"스키마 오류: {name}은(는) 비어 있지 않은 2차원 배열이어야 합니다.")

    actual_rows = len(raw_matrix)
    target_size = expected_size if expected_size is not None else actual_rows
    if actual_rows != target_size:
        raise ValueError(
            f"크기 불일치: {name} 행 수({actual_rows}) != 기대 크기({target_size})"
        )

    parsed_rows: list[list[float]] = []
    for r_idx, row in enumerate(raw_matrix):
        if not isinstance(row, list):
            raise ValueError(f"스키마 오류: {name} {r_idx}번 행이 배열이 아닙니다.")
        if len(row) != target_size:
            raise ValueError(
                f"크기 불일치: {name} {r_idx}번 행 열 수({len(row)}) != 기대 크기({target_size})"
            )
        parsed_row: list[float] = []
        for c_idx, cell in enumerate(row):
            if not _is_valid_number(cell):
                raise ValueError(
                    f"데이터 오류: {name} ({r_idx}, {c_idx}) 값이 유효한 숫자가 아닙니다: {cell!r}"
                )
            parsed_row.append(float(cell))
        parsed_rows.append(parsed_row)

    return Matrix.from_rows(parsed_rows)


def extract_size_from_key(key: str, pattern: re.Pattern[str]) -> int:
    """'size_N' 또는 'size_N_idx' 형태 키에서 N을 추출한다."""
    if not isinstance(key, str):
        raise ValueError(f"스키마 오류: 키는 문자열이어야 합니다: {key!r}")
    match = pattern.fullmatch(key.strip())
    if not match:
        raise ValueError(f"스키마 오류: 키 형식이 올바르지 않습니다: {key!r}")
    size = int(match.group(1))
    if size <= 0:
        raise ValueError(f"스키마 오류: 크기 N은 1 이상이어야 합니다: {key!r}")
    return size


def load_json_file(json_path: str | Path) -> dict:
    """JSON 파일을 읽어 최상위 dict 구조를 반환한다."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("스키마 오류: JSON 최상위 구조는 객체(dict)여야 합니다.")
    return data


def load_filters(
    raw_filters: object,
) -> tuple[dict[int, dict[str, Matrix]], dict[int, str]]:
    """filters 섹션을 파싱·정규화·검증해 크기별 필터 맵과 오류 맵을 반환한다."""
    if not isinstance(raw_filters, dict):
        raise ValueError("스키마 오류: 'filters' 항목은 객체(dict)여야 합니다.")

    filters_by_size: dict[int, dict[str, Matrix]] = {}
    filter_errors: dict[int, str] = {}

    for size_key, pair in raw_filters.items():
        try:
            n = extract_size_from_key(size_key, FILTER_SIZE_KEY_RE)
        except ValueError:
            # size_N 형식 자체가 잘못된 키는 무시하되 전체 중단은 막음
            continue

        if not isinstance(pair, dict):
            filter_errors[n] = f"스키마 오류: {size_key} 필터 묶음이 객체가 아닙니다."
            continue

        normalized_pair: dict[str, Matrix] = {}
        try:
            for raw_label, raw_mat in pair.items():
                std_label = normalize_label(raw_label)
                normalized_pair[std_label] = parse_square_matrix(
                    raw_mat, expected_size=n, name=f"{size_key}.{raw_label}"
                )
            for required in ("Cross", "X"):
                if required not in normalized_pair:
                    raise ValueError(
                        f"스키마 오류: {size_key}에 '{required}' 필터가 누락되었습니다."
                    )
        except ValueError as exc:
            filter_errors[n] = str(exc)
            continue

        filters_by_size[n] = normalized_pair

    return filters_by_size, filter_errors


def prepare_pattern_cases(
    raw_patterns: object,
    filters_by_size: dict[int, dict[str, Matrix]],
    filter_errors: dict[int, str] | None = None,
) -> list[ValidatedCase]:
    """patterns 각 항목을 검증하고 크기 일치 여부까지 확인해 ValidatedCase 목록을 만든다."""
    if not isinstance(raw_patterns, dict):
        raise ValueError("스키마 오류: 'patterns' 항목은 객체(dict)여야 합니다.")

    err_map = filter_errors or {}
    cases: list[ValidatedCase] = []

    for case_id, item in raw_patterns.items():
        try:
            n = extract_size_from_key(case_id, PATTERN_KEY_RE)
            if not isinstance(item, dict):
                raise ValueError(f"스키마 오류: {case_id} 항목이 객체가 아닙니다.")
            if "input" not in item or "expected" not in item:
                raise ValueError(f"스키마 오류: {case_id}에 'input' 또는 'expected'가 없습니다.")

            expected_label = normalize_label(item["expected"])

            if n in err_map:
                raise ValueError(err_map[n])
            if n not in filters_by_size:
                raise ValueError(f"필터 누락: size_{n} 필터가 정의되어 있지 않습니다.")

            cross_filter = filters_by_size[n]["Cross"]
            x_filter = filters_by_size[n]["X"]
            pattern_matrix = parse_square_matrix(
                item["input"], expected_size=n, name=f"{case_id}.input"
            )

            if pattern_matrix.size != cross_filter.size or pattern_matrix.size != x_filter.size:
                raise ValueError(
                    f"크기 불일치: 패턴({pattern_matrix.size}x{pattern_matrix.size})과 "
                    f"size_{n} 필터 크기가 다릅니다."
                )

            cases.append(
                ValidatedCase(
                    case_id=case_id,
                    size=n,
                    pattern=pattern_matrix,
                    cross_filter=cross_filter,
                    x_filter=x_filter,
                    expected=expected_label,
                )
            )
        except ValueError as exc:
            cases.append(ValidatedCase(case_id=str(case_id), error=str(exc)))

    return cases


# ---------------------------------------------------------------------------
# 4~5단계 + 보너스(2): 패턴 생성기, 성능 분석 및 모드 1 / 모드 2 / 모드 3 실행 흐름
# ---------------------------------------------------------------------------


def generate_cross_pattern(size: int) -> Matrix:
    """크기 N에 맞는 N x N 십자가(Cross) 패턴 행렬을 자동 생성한다."""
    matrix = Matrix(size, fill=0.0)
    center = size // 2
    for idx in range(size):
        matrix.set(center, idx, 1.0)
        matrix.set(idx, center, 1.0)
    return matrix


def generate_x_pattern(size: int) -> Matrix:
    """크기 N에 맞는 N x N X 패턴 행렬을 자동 생성한다."""
    matrix = Matrix(size, fill=0.0)
    for idx in range(size):
        matrix.set(idx, idx, 1.0)
        matrix.set(idx, size - 1 - idx, 1.0)
    return matrix


def format_matrix(matrix: Matrix) -> str:
    """행렬을 콘솔에 보기 좋게 문자열로 포맷한다."""
    lines: list[str] = []
    for row in matrix.rows:
        cells = [str(int(v)) if v.is_integer() else f"{v:g}" for v in row]
        lines.append(" ".join(cells))
    return "\n".join(lines)


def benchmark_mac(
    pattern: Matrix,
    filter_matrix: Matrix,
    iterations: int = BENCHMARK_ITERATIONS,
) -> tuple[float, int]:
    """I/O를 제외하고 compute_mac 호출 구간만 iterations회 반복해 (평균 ms, N^2)를 반환한다."""
    start = time.perf_counter()
    for _ in range(iterations):
        compute_mac(pattern, filter_matrix)
    elapsed_sec = time.perf_counter() - start
    avg_ms = (elapsed_sec / iterations) * 1000.0
    ops = pattern.size * pattern.size
    return avg_ms, ops


def print_performance_table(
    rows: list[tuple[int, float, int]],
    iterations: int = BENCHMARK_ITERATIONS,
    title: str = "[3] 성능 분석",
) -> None:
    """크기별 평균 MAC 연산 시간(ms)과 연산 횟수(N^2) 표를 출력한다."""
    print("\n" + "=" * 58)
    print(f"  {title} | {iterations}회 평균 측정")
    print("-" * 58)
    print(f"{'SIZE':<12} | {'AVG (ms)':>14} | {'MAC OPS':>12}")
    print("-" * 44)
    for size, avg_ms, ops in rows:
        size_label = f"{size}x{size}"
        print(f"{size_label:<12} | {avg_ms:>14.4f} | {ops:>12}")


def _evaluate_and_print_pair(
    filter_a: Matrix,
    filter_b: Matrix,
    pattern: Matrix,
    label_a: str = "A",
    label_b: str = "B",
    tie_label: str = "판정 불가",
) -> None:
    """두 필터와 입력 패턴에 대한 MAC 점수, 연산 시간, 판정 및 단건 성능 표를 출력한다."""
    score_a = compute_mac(pattern, filter_a)
    score_b = compute_mac(pattern, filter_b)
    avg_ms_a, ops = benchmark_mac(pattern, filter_a)
    avg_ms_b, _ = benchmark_mac(pattern, filter_b)
    avg_ms = (avg_ms_a + avg_ms_b) / 2.0

    decision = compare_scores(
        score_a, score_b, label_a=label_a, label_b=label_b, tie_label=tie_label
    )

    print("\n" + "=" * 58)
    print("  STEP [3] MAC 결과")
    print("-" * 58)
    print(f"{label_a} 점수: {score_a}")
    print(f"{label_b} 점수: {score_b}")
    print(f"연산 시간(평균/{BENCHMARK_ITERATIONS}회): {avg_ms:.4f} ms")
    if decision == tie_label:
        print(f"판정 결과 : {tie_label} (|{label_a}-{label_b}| < {DEFAULT_EPSILON:g})")
    else:
        print(f"판정 결과 : {decision}")

    print_performance_table([(pattern.size, avg_ms, ops)], title="성능 분석")


def run_user_input_mode(input_fn: Callable[[str], str] = input) -> None:
    """모드 1: 3x3 필터 2개(A, B)와 패턴을 입력받아 MAC 연산·판정·성능을 출력한다."""
    print("\n" + "=" * 58)
    print("  STEP [1] 필터 입력")
    print("-" * 58)
    filter_a = read_matrix_from_console("필터 A", size=3, input_fn=input_fn)
    print()
    filter_b = read_matrix_from_console("필터 B", size=3, input_fn=input_fn)
    print("\n[OK] 필터 A, B (3x3) 저장 완료")

    print("\n" + "=" * 58)
    print("  STEP [2] 패턴 입력")
    print("-" * 58)
    pattern = read_matrix_from_console("패턴", size=3, input_fn=input_fn)
    print("[OK] 패턴 (3x3) 저장 완료")

    _evaluate_and_print_pair(filter_a, filter_b, pattern)


def _evaluate_valid_case(case: ValidatedCase) -> tuple[bool, str | None]:
    """검증을 통과한 단일 패턴 케이스의 MAC 연산·판정·PASS/FAIL을 출력한다."""
    assert case.pattern is not None
    assert case.cross_filter is not None
    assert case.x_filter is not None
    assert case.expected is not None

    cross_score = compute_mac(case.pattern, case.cross_filter)
    x_score = compute_mac(case.pattern, case.x_filter)
    decision = compare_scores(
        cross_score, x_score, label_a="Cross", label_b="X", tie_label="UNDECIDED"
    )
    is_pass = decision == case.expected
    status = "PASS" if is_pass else "FAIL"

    print(f"\n  CASE | {case.case_id}")
    print("-" * 40)
    print(f"  Cross 점수 : {cross_score}")
    print(f"  X 점수     : {x_score}")
    if is_pass:
        print(f"판정 결과 : {decision} | 기대값: {case.expected} | {status}")
        return True, None

    if decision == "UNDECIDED":
        reason = f"동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| < {DEFAULT_EPSILON:g})"
        print(f"판정 결과 : {decision} | 기대값: {case.expected} | {status} (동점 규칙)")
    else:
        reason = f"예측 불일치 (판정: {decision}, expected: {case.expected})"
        print(f"판정 결과 : {decision} | 기대값: {case.expected} | {status}")
    return False, reason


def _collect_benchmark_samples(
    filters_by_size: dict[int, dict[str, Matrix]],
    cases: list[ValidatedCase],
) -> list[tuple[int, float, int]]:
    """3x3과 로드된 크기(5, 13, 25 등)별 MAC 연산 성능 표 데이터를 수집한다."""
    samples: dict[int, tuple[Matrix, Matrix]] = {
        3: (generate_cross_pattern(3), generate_cross_pattern(3))
    }

    for case in cases:
        if (
            case.error is None
            and case.size is not None
            and case.size not in samples
            and case.pattern is not None
            and case.cross_filter is not None
        ):
            samples[case.size] = (case.pattern, case.cross_filter)

    for size, pair in filters_by_size.items():
        if size not in samples:
            samples[size] = (generate_cross_pattern(size), pair["Cross"])

    rows: list[tuple[int, float, int]] = []
    for size in sorted(samples):
        pattern_mat, filter_mat = samples[size]
        avg_ms, ops = benchmark_mac(pattern_mat, filter_mat)
        rows.append((size, avg_ms, ops))
    return rows


def run_json_analysis_mode(json_path: str | Path) -> None:
    """모드 2: data.json을 로드·검증하고 일괄 판정, 성능 분석, 결과 요약을 출력한다."""
    print("\n" + "=" * 58)
    print("  STEP [1] 필터 로드")
    print("-" * 58)
    try:
        raw_json = load_json_file(json_path)
        filters_by_size, filter_errors = load_filters(raw_json.get("filters"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"JSON 로드 오류: {exc}")
        return

    all_filter_sizes = sorted(set(filters_by_size.keys()) | set(filter_errors.keys()))
    if not all_filter_sizes:
        print("로드할 필터가 없습니다.")
    for size in all_filter_sizes:
        if size in filter_errors:
            print(f"[ERROR] size_{size} 필터 로드   FAIL : {filter_errors[size]}")
        else:
            print(f"[OK] size_{size} 필터 로드 완료 (Cross, X)")

    print("\n" + "=" * 58)
    print("  STEP [2] 패턴 분석 (라벨 정규화 적용)")
    print("-" * 58)
    try:
        cases = prepare_pattern_cases(
            raw_json.get("patterns"), filters_by_size, filter_errors
        )
    except ValueError as exc:
        print(f"패턴 스키마 오류: {exc}")
        return

    passed = 0
    failures: list[tuple[str, str]] = []

    for case in cases:
        if case.error is not None:
            print(f"\n  CASE | {case.case_id}")
            print("-" * 40)
            print(f"판정 결과 : FAIL | 사유: {case.error}")
            failures.append((case.case_id, case.error))
            continue

        is_pass, reason = _evaluate_valid_case(case)
        if is_pass:
            passed += 1
        elif reason is not None:
            failures.append((case.case_id, reason))

    bench_rows = _collect_benchmark_samples(filters_by_size, cases)
    print_performance_table(bench_rows)

    total = len(cases)
    failed = len(failures)
    print("\n" + "=" * 58)
    print("  STEP [4] 결과 요약")
    print("-" * 58)
    print(f"  전체 테스트 : {total}개")
    print(f"  PASS : {passed}개")
    print(f"  FAIL : {failed}개")
    if failures:
        print("\n실패 케이스:")
        for case_id, reason in failures:
            print(f"- {case_id}: {reason}")


def read_positive_int(prompt: str, input_fn: Callable[[str], str] = input) -> int:
    """1 이상의 정수를 입력받을 때까지 검증하며 재입력을 유도한다."""
    while True:
        raw = input_fn(prompt).strip()
        try:
            value = int(raw)
            if value >= 1:
                return value
        except ValueError:
            pass
        print("입력 형식 오류: 1 이상의 정수를 입력하세요.")


def run_pattern_generator_mode(input_fn: Callable[[str], str] = input) -> None:
    """보너스(2): 크기 N을 입력받아 Cross/X 패턴을 자동 생성하고 MAC·성능 분석에 재활용한다."""
    print("\n" + "=" * 58)
    print("  STEP [1] N x N 패턴 자동 생성")
    print("-" * 58)
    size = read_positive_int("생성할 패턴 크기 N 입력 (예: 3, 5, 13, 25): ", input_fn=input_fn)
    cross_mat = generate_cross_pattern(size)
    x_mat = generate_x_pattern(size)

    print(f"\n[생성된 {size}x{size} Cross 패턴]")
    print(format_matrix(cross_mat))
    print(f"\n[생성된 {size}x{size} X 패턴]")
    print(format_matrix(x_mat))

    print("\n" + "=" * 58)
    print("  STEP [2] 생성 패턴 재활용 (MAC 판정)")
    print("-" * 58)
    while True:
        choice = input_fn("입력 패턴 선택 (1: Cross, 2: X): ").strip()
        if choice in {"1", "2"}:
            break
        print("입력 오류: 1 또는 2를 입력하세요.")

    target_pattern = cross_mat if choice == "1" else x_mat
    _evaluate_and_print_pair(
        cross_mat,
        x_mat,
        target_pattern,
        label_a="Cross",
        label_b="X",
        tie_label="UNDECIDED",
    )

    bench_sizes = sorted({3, 5, 13, 25, size})
    bench_rows: list[tuple[int, float, int]] = []
    for s in bench_sizes:
        p = generate_cross_pattern(s)
        f = generate_cross_pattern(s)
        avg_ms, ops = benchmark_mac(p, f)
        bench_rows.append((s, avg_ms, ops))
    print_performance_table(bench_rows, title="생성 패턴 크기별 성능 비교")


def select_mode(input_fn: Callable[[str], str] = input) -> str:
    """사용자로부터 유효한 실행 모드(1, 2 또는 3)를 입력받는다."""
    print("=" * 58)
    print("             MINI NPU | MAC SIMULATOR")
    print("=" * 58)
    print("  실행할 분석 모드를 선택하세요.\n")
    print("  [1] 직접 입력       - 3x3 행렬 연산")
    print("  [2] 파일 분석       - data.json 일괄 검사")
    print("  [3] 패턴 생성       - N x N Cross / X (보너스)")
    print("-" * 58)
    while True:
        choice = input_fn("  모드 번호 > ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("입력 오류: 1, 2 또는 3을 입력하세요.")


def main() -> None:
    """Mini NPU Simulator 콘솔 애플리케이션 진입점."""
    mode = select_mode()
    if mode == "1":
        run_user_input_mode()
    elif mode == "2":
        data_path = Path(__file__).with_name("data.json")
        run_json_analysis_mode(data_path)
    else:
        run_pattern_generator_mode()


if __name__ == "__main__":
    main()
